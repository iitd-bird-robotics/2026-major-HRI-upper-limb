import os
import opensim as osim  # type: ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # type: ignore

from arm26_utilities.actuator_module import actuatorModule
from arm26_utilities.add_actuators_controller import addActuatorsController
from arm26_utilities.add_bodies import addBodies
from arm26_utilities.control_block import controlBlock
from arm26_utilities.inverse_dynamics import IDSolver
from arm26_utilities.Multiplier import multiplier
from arm26_utilities.Optimizer import optimizer
from arm26_utilities.Activation_Dynamics import ExFromAct
from scipy.io import loadmat

import csv

from emg_xlxs import EMGProcessor


# from nsd import train_emg_model
from nsd import test_emg_model,train_emg_model

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ArmAssistSimulation:
    def __init__(self, time, s_angle, e_angle, external_load=0, assisted=0, cs_choice=1, only_elbow=1):
        os.add_dll_directory("C:/OpenSim 4.5/bin")
        os.add_dll_directory("C:/OpenSim 4.5/sdk/Python/opensim")

        self.time = time
        self.s_angle = s_angle*0
        self.s_angle_0 = s_angle
        self.e_angle = e_angle
        self.external_load = external_load
        self.assisted = assisted
        self.cs_choice = cs_choice
        self.only_elbow = only_elbow
        self.signal_c = [[],[],[],[],[],[]]
        self.signal_a = []
        self.reqElbow_tau = []
        self.reqShoulder_tau = []


        self.emg_data()
        self._initialize_models()
        self._initialize_data_storage()

    def _initialize_models(self):
        self.ID_model = osim.Model('arm26.osim')
        if self.only_elbow:
            self.main_model = osim.Model('arm16.osim')
        else:
            self.main_model = osim.Model('arm26.osim')

        path = 'Geometry'
        osim.ModelVisualizer_addDirToGeometrySearchPaths(path)

        addBodies(self.main_model, self.external_load)
        addBodies(self.ID_model, self.external_load)

        self.main_model, self.brain, self.pact_a, self.pact_b = addActuatorsController(self.main_model)
        self.main_model.finalizeConnections()
        self.ID_model.finalizeConnections()

        self.ID_state = self.ID_model.initSystem()
        self.main_model.setUseVisualizer(True)
        self.state = self.main_model.initSystem()

        for mn in range(6):
            self.ID_model.getMuscles().get(mn).setAppliesForce(self.ID_state, False)

        viz = self.main_model.updVisualizer().updSimbodyVisualizer()
        viz.setBackgroundColor(osim.Vec3(1))
        viz.setGroundHeight(-1)

    def _initialize_data_storage(self):
        self.previous_activation = 0.01 * np.ones(6)
        self.MA_el = 0.01 * np.ones(6)
        self.MA_sh = 0.01 * np.ones(6)
        self.Fm = 0.01 * np.ones(6)
        self.T_el_FD = 0.01 * np.ones(6)

        self.m1 = 0.01
        self.m2 = 0.01

        self.measured_shoulder_trajectory = []
        self.measured_trajectory = []
        self.desired_trajectory = []
        self.elbow_mul = []
        self.shoulder_mul = []
        self.desired_act = []
        self.fd_Elbow_torque = []
        self.Assistive_tau = []
        self.total_tau = []
        self.Excitation = []
        self.t = []
        self.MCT = []

    def run_simulation(self):
        break_count= []
        for i in range(1, len(self.e_angle) - 1):
            timed = self.time[i + 1] - self.time[i]

            shoulder_tau, elbow_tau, _, _ = IDSolver(
                self.ID_model, self.ID_state, self.main_model, self.state,
                self.e_angle, self.s_angle, timed, i
            )
            ######
            # elbow_tau+=shoulder_tau*6
            # print(i,self.e_angle[i],self.s_angle[i],shoulder_tau, elbow_tau)
            self.reqShoulder_tau.append(shoulder_tau)
            self.reqElbow_tau.append(elbow_tau)

            angle_e = self.main_model.getCoordinateSet().get(1).getValue(self.state)
            angle_s = self.main_model.getCoordinateSet().get(0).getValue(self.state)

            if self.cs_choice == 1:
                tauA = controlBlock(angle_e, self.external_load)

            T1, T2 = actuatorModule(tauA, self.m1, self.m2)

            aM2, aM1, _, _, _, pass_tau_el, pass_tau_sh = multiplier(self.main_model, self.state)
            self.elbow_mul.append(aM2)
            self.shoulder_mul.append(aM1)

            if self.assisted == 0:
                torque = np.array([elbow_tau - pass_tau_el, shoulder_tau - pass_tau_sh])
            else:
                torque = np.array([elbow_tau - tauA - pass_tau_el, shoulder_tau - pass_tau_sh])

            x, self.previous_activation = optimizer(
                self.main_model, self.state, aM2, aM1, torque, self.only_elbow, i, self.previous_activation
            )
            self.desired_act.append(x)

            ex, self.previous_activation = ExFromAct(self.main_model, x, self.previous_activation, timed, i)

            for muscle, val in zip(["TRIlong", "TRIlat", "TRImed", "BIClong", "BICshort", "BRA"], ex):
                self.brain.prescribeControlForActuator(muscle, osim.StepFunction(0.0, timed, val, val))

            T1_cmd = T1 if self.assisted else 0
            T2_cmd = T2 if self.assisted else 0
            self.brain.prescribeControlForActuator('pact_a', osim.StepFunction(0, timed, T1_cmd, T1_cmd))
            self.brain.prescribeControlForActuator('pact_b', osim.StepFunction(0, timed, T2_cmd, T2_cmd))

            self.state.setTime(0.0)
            manager = osim.Manager(self.main_model, self.state)
            self.state = manager.integrate(timed)
            self.main_model.realizeVelocity(self.state)
            self.main_model.realizeAcceleration(self.state)
            try:
                self.main_model.equilibrateMuscles(self.state)
            except Exception as e:
                break_count.append(i)
                print("braked at ",i)
                print(e)
                # exit()
                continue

            self.m1 = self.pact_a.computeMomentArm(self.state, self.main_model.getCoordinateSet().get(1))
            self.m2 = self.pact_b.computeMomentArm(self.state, self.main_model.getCoordinateSet().get(1))

            tot = 0
            for k in range(6):
                self.previous_activation[k] = self.main_model.getMuscles().get(k).getActivation(self.state)
                self.MA_el[k] = self.main_model.getMuscles().get(k).computeMomentArm(self.state, self.main_model.getCoordinateSet().get(1))
                self.MA_sh[k] = self.main_model.getMuscles().get(k).computeMomentArm(self.state, self.main_model.getCoordinateSet().get(0))
                self.Fm[k] = self.main_model.getMuscles().get(k).getTendonForce(self.state)
                self.T_el_FD[k] = self.MA_el[k] * self.Fm[k]
                tot += self.T_el_FD[k]

            self.fd_Elbow_torque.append(tot)
            assist_tau = T1 * self.m1 + T2 * self.m2
            self.Assistive_tau.append(assist_tau)
            self.total_tau.append(tot + assist_tau)

            self.measured_trajectory.append(angle_e)
            self.measured_shoulder_trajectory.append(angle_s)
            self.desired_trajectory.append(self.e_angle[i])
            self.Excitation.append(ex)
            self.t.append(self.time[i])
            self.MCT.append(self.main_model.getProbeSet().get(0).getProbeOutputs(self.state).get(0))
        print('break_count: ',break_count)


    def plot_results(self):
        plt.figure()
        plt.plot(self.t, self.measured_trajectory, self.t, self.desired_trajectory)
        plt.title(f"Elbow Trajectory for {self.external_load} kg load, with Assistance={self.assisted}")
        plt.xlabel('Time (sec.)')
        plt.ylabel('Elbow Angle (rad.)')
        plt.legend(['measured_trajectory', 'desired_trajectory'])

        plt.figure()
        plt.plot(self.t, self.reqElbow_tau)
        plt.title(f"ID Elbow Torque for {self.external_load} kg load, with Assistance = {self.assisted}")
        plt.xlabel('Time (sec.)')
        plt.ylabel('Torque (Nm)')

        plt.figure()
        plt.plot(self.t, self.fd_Elbow_torque)
        plt.title(f"FD Elbow Torque for {self.external_load} kg load, with Assistance = {self.assisted}")
        plt.xlabel('Time (sec.)')
        plt.ylabel('Torque (Nm)')

        plt.figure()
        plt.plot(self.t, self.fd_Elbow_torque, self.t, self.reqElbow_tau, self.t, self.Assistive_tau, self.t, self.total_tau)
        plt.title(f"Comparison Elbow Torque for {self.external_load} kg load, with Assistance = {self.assisted}")
        plt.xlabel('Time (sec.)')
        plt.ylabel('Torque (Nm)')
        plt.legend(['FD Elbow Torque', 'ID Elbow Torque', 'Assistive_tau', 'Net Elbow tau'])

        plt.figure()
        plt.plot(self.t, [a[4] for a in self.desired_act])
        plt.title(f"Muscle Activations for {self.external_load} kg load, with Assistance = {self.assisted}, Biceps Short")
        plt.xlabel('Time (sec.)')
        plt.ylabel('Muscle Activations')
        plt.legend(['Biceps Short'])

        plt.figure()
        plt.plot(self.t, [a[3] for a in self.desired_act])
        plt.title(f"Muscle Activations for {self.external_load} kg load, with Assistance = {self.assisted}, Biceps Long")
        plt.xlabel('Time (sec.)')
        plt.ylabel('Muscle Activations')
        plt.legend(['Biceps Long'])

        plt.figure()
        plt.plot(self.t, [a[1] for a in self.desired_act])
        plt.title(f"Muscle Activations for {self.external_load} kg load, with Assistance = {self.assisted}, Triceps Lateral")
        plt.xlabel('Time (sec.)')
        plt.ylabel('Muscle Activations')
        plt.legend(['Triceps Lateral'])

        plt.figure()
        plt.plot(self.t, [a[0] for a in self.desired_act])
        plt.title(f"Muscle Activations for {self.external_load} kg load, with Assistance = {self.assisted}, Triceps Long")
        plt.xlabel('Time (sec.)')
        plt.ylabel('Muscle Activations')
        plt.legend(['Triceps Long'])


        plt.figure()
        mct_df = pd.DataFrame(self.MCT).interpolate(method='linear', axis=0).ffill().rolling(8).mean()
        plt.plot(self.t, mct_df)
        plt.title(f"Metabolic Cost for {self.external_load} kg load, with Assistance = {self.assisted}")
        plt.xlabel('Time (sec.)')
        plt.ylabel('Metabolic Cost (J/s)')

        plt.show()

    def save_in_csv(self):
        for i in self.desired_act:
            self.signal_c[0].append(i[4])
            self.signal_c[1].append(i[3])
            self.signal_c[2].append(i[1])
            self.signal_c[3].append(i[0])
            self.signal_c[4].append(i[2])
            self.signal_c[5].append(i[5])


        m_len = min(            len(self.e_angle)
            ,len(self.reqElbow_tau)
            ,len(self.s_angle_0)
            ,len(self.reqShoulder_tau)
            ,len(self.signal_c[0])
            ,len(self.signal_c[1])
            ,len(self.signal_c[2])
            ,len(self.signal_c[3])
            ,len(self.signal_a[0])
            ,len(self.signal_a[1])
            ,len(self.signal_a[2])
            ,len(self.signal_a[3])
)

        print('e_angle',len(self.e_angle)
            ,'e_tau',len(self.reqElbow_tau)
            ,'s_angle',len(self.s_angle_0)
            ,'s_tau',len(self.reqShoulder_tau)
            ,'s_emg1',len(self.signal_c[0])
            ,'s_emg2',len(self.signal_c[1])
            ,'s_emg3',len(self.signal_c[2])
            ,'s_emg4',len(self.signal_c[3])
            ,'s_emg5',len(self.signal_c[4])
            ,'s_emg6',len(self.signal_c[5])
            ,'a_emg1',len(self.signal_a[0])
            ,'a_emg2',len(self.signal_a[1])
            ,'a_emg3',len(self.signal_a[2])
            ,'a_emg4',len(self.signal_a[3]))

        df = pd.DataFrame({
            'e_angle':self.e_angle[len(self.e_angle)-m_len:]
            ,'e_tau':self.reqElbow_tau[len(self.reqElbow_tau)-m_len:]
            ,'s_angle':self.s_angle_0[len(self.s_angle_0)-m_len:]
            ,'s_tau':self.reqShoulder_tau[len(self.reqShoulder_tau)-m_len:]
            ,'s_emg1':self.signal_c[0][len(self.signal_c[0])-m_len:]
            ,'s_emg2':self.signal_c[1][len(self.signal_c[1])-m_len:]
            ,'s_emg3':self.signal_c[2][len(self.signal_c[2])-m_len:]
            ,'s_emg4':self.signal_c[3][len(self.signal_c[3])-m_len:]
            ,'s_emg5':self.signal_c[4][len(self.signal_c[4])-m_len:]
            ,'s_emg6':self.signal_c[5][len(self.signal_c[5])-m_len:]
            ,'a_emg1':self.signal_a[0][len(self.signal_a[0])-m_len:]
            ,'a_emg2':self.signal_a[1][len(self.signal_a[1])-m_len:]
            ,'a_emg3':self.signal_a[2][len(self.signal_a[2])-m_len:]
            ,'a_emg4':self.signal_a[3][len(self.signal_a[3])-m_len:]
        })

        print(df)

        df.to_csv('data/Alok/try.csv',mode='a', header=False, index=False)
# 167.718822500877
# 167.732331405331

        


############## RMS envlop

    def signal_diff(self):
    
        label = ['Biceps short', 'Biceps long', 'Triceps lateral head', 'Triceps long head','biscep short','flexer']
        
        for j in range(4):
            x, y = self.signal_a[j], self.signal_c[j]
            
            plt.figure(figsize=(10, 4))  # Optional: adjust figure size
            plt.plot(x, label='Actual', color='blue')
            plt.plot(y, label='Calculated', color='orange', linestyle='--')
            plt.title(f'Actual vs Calculated - {label[j]}')
            plt.xlabel('Time')
            plt.ylabel('Signal')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        for j in range(4,6):
            x =  self.signal_c[j]
            
            plt.figure(figsize=(10, 4))  # Optional: adjust figure size
            plt.plot(x, label='Actual', color='blue')
            # plt.plot(y, label='Calculated', color='orange', linestyle='--')
            plt.title(f'Actual vs Calculated - {label[j]}')
            plt.xlabel('Time')
            plt.ylabel('Signal')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()


######## Y AB

    def emg_data(self):

        path = r"D:\books\IIT D\3rd\Mtp\Loading_unloading_data\Loading_unloading_data\EMG data\Alok_7.5KG_up.csv"
        processor = EMGProcessor(path)
        signal = processor.run_all()
        df = pd.DataFrame(signal, columns=[f'Muscle{i+1}' for i in range(4)])
        self.signal_a.append(list(df.iloc[:,0])[::32])
        self.signal_a.append(list(df.iloc[:,1])[::32])
        self.signal_a.append(list(df.iloc[:,2])[::32])
        self.signal_a.append(list(df.iloc[:,3])[::32])


######### 1 biscep long,2 biscep short, 3 trisep long ,4 trisep laterla ,5 flexer,6 bricroradilys, 7  
# data = loadmat("D:/books/IIT D/2nd/JRL891/Upper_limb_data/Upper_limb_data/Sub02_Anant/Angle/2023-04-17-17-21_run08_4kg.mat",struct_as_record=False, squeeze_me=True)
# data = data['record_2023_04_17_17_21_run08']
# s_angle=np.subtract(np.deg2rad(data.movements.sources.signals.signal_2.data),0)
# e_angle=np.subtract(np.deg2rad(data.movements.sources.signals.signal_1.data),0)

angle_path = r"D:\books\IIT D\3rd\Mtp\Loading_unloading_data\Loading_unloading_data\Xsens data\Alok_Kumar_7.5KGxsens.xlsx"
angle_df = pd.read_excel(angle_path, sheet_name="Joint Angles ZXY")

e_angle = np.deg2rad(angle_df["Right Elbow Flexion/Extension"].values)
e_angle=np.subtract(e_angle,-0.1)
s_angle = np.deg2rad(angle_df["Right Shoulder Flexion/Extension"].values)


print(max(e_angle))
print(np.where(e_angle == max(e_angle)))
plt.plot(e_angle)
plt.plot(s_angle)
plt.show()
# exit(0)



# for i in range(20):
#     print(s_angle[i],e_angle[i])

# e_angle[282]=167.718822500877
# s_angle = [0]*len(e_angle)
# s_angle = (s_angle - np.max(s_angle)) / (np.max(s_angle) - np.min(s_angle))
time=[i*0.017 for i in range(1,min(len(s_angle),len(e_angle))+1)]

# print(min(s_angle))
# print(max(s_angle))
# print(type(s_angle))
# print(type(time))

# data = pd.read_excel("Trajectory2.xlsx", 'Sheet1')

# time=data.iloc[0:240,0]
# s_angle=data.iloc[0:240,1]
# e_angle=data.iloc[0:240,2]
# print()

# print(min(e_angle))
# print(max(e_angle))
# print(type(s_angle))
# print(type(time))

sim = ArmAssistSimulation(
    time,s_angle,e_angle,only_elbow=1,external_load=7.5,assisted=1
)
sim.run_simulation()
# sim.plot_results()
sim.save_in_csv()
sim.signal_diff()

# train_emg_model(sim)
# train_emg_model(sim)
# test_emg_model(sim)
