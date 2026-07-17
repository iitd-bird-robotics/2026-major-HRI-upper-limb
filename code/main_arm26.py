import os
os.add_dll_directory("C:/OpenSim 4.5/bin")
os.add_dll_directory("C:/OpenSim 4.5/sdk/Python/opensim")

# Importing Opensim and other python packages as required
import opensim as osim # type: ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # type: ignore


# Created a arm26_utilities folder than contains all the child modules
from arm26_utilities.actuator_module import actuatorModule
from arm26_utilities.add_actuators_controller import addActuatorsController
from arm26_utilities.add_bodies import addBodies
from arm26_utilities.control_block import controlBlock
from arm26_utilities.inverse_dynamics import IDSolver
from arm26_utilities.Multiplier import multiplier
from arm26_utilities.Optimizer import optimizer
from arm26_utilities.Activation_Dynamics import ExFromAct


# Setting of load, assistance, control and Degree of freedom
External_load = 0
Assisted = 0 #0 means off/1 means on
cs_choice = 1 #gravity compensation
Only_elbow = 1 #Assistance only at elbow or not

# Extracting data from the excel file and storing
mydata = pd.read_excel("Trajectory2.xlsx",'Sheet1')
time=mydata.iloc[:,0]
s_angle=mydata.iloc[:,1]
e_angle=mydata.iloc[:,2]



# Defining models to use          
ID_model = osim.Model('arm26.osim')
if Only_elbow == 1:
    main_model = osim.Model('arm16.osim')
else:
    main_model = osim.Model('arm26.osim')

# For visualization purposes
path='Geometry'
osim.ModelVisualizer_addDirToGeometrySearchPaths(path)

# Now, adding the load, arm strap, and forearm strap bodies to both the models and define their respective joints.
addBodies(main_model, External_load)
addBodies(ID_model, External_load)

# Add actuators and controller (to give control signals to these actuators, i.e., human brain) to only the with muscles model i.e main_model
main_model,brain,pact_a,pact_b=addActuatorsController(main_model)

# After the model is defined, build the connections between parts of the model and get the initial state of the model.
main_model.finalizeConnections()
ID_model.finalizeConnections()
ID_state = ID_model.initSystem()
main_model.setUseVisualizer(True)
state = main_model.initSystem()


# Setting ID_model's muscle forces to zero
for mn in range(0,6):
    ID_model.getMuscles().get(mn).setAppliesForce(ID_state, False)

# Model Visualization
viz = main_model.updVisualizer().updSimbodyVisualizer()
viz.setBackgroundColor(osim.Vec3(1))
viz.setGroundHeight(-1)

# Initialization
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Exosuit cables' moment arms
m1 = 0.01
m2 = 0.01
# Initializing previous activation of the muscles of the model arm26
previous_activation = 0.01 * np.ones(6)
# Initializing moment arms of arm25 model's muscles crossing elbow and shoulder joints
MA_el = 0.01 * np.ones(6)
MA_sh = 0.01 * np.ones(6)
# Initializing Force to be used to calculate FD torque
Fm = 0.01 * np.ones(6)
# Initializing FD torque
T_el_FD= 0.01 * np.ones(6)
# Initializing lists for storage
measured_shoulder_trajectory=[]
measured_trajectory=[]
desired_trajectory=[]
act=[]
reqElbow_tau=[]
reqShoulder_tau=[]
elbow_mul=[]
shoulder_mul=[]
desired_act=[]
fd_Elbow_torque=[]
Assistive_tau=[]
total_tau=[]
Excitation=[]
t=[]
MCT=[]


# Start of the simulation loop
# Sequence of simulation process flow-----> Inverse Dynamics-(to get the joint torques)---->Calculating assistive tau (tauA) using control scheme with (1) representing Gravity Compensation Control.
# ------>Static Optimization-(to get muscle activations)---->Forward Dynamics-(for forward progression of the simulation and subsequent updation of model's state parameters)

for i in range(1,len(e_angle)-1):
    timed = time[i+1]-time[i]
    
    # Inverse Dynamics
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    shoulder_tau,elbow_tau,elb_accel_new,elb_accel_ref=IDSolver(ID_model, ID_state, main_model, state, e_angle, s_angle, timed, i)
    print(shoulder_tau,elbow_tau)
    reqShoulder_tau.append(shoulder_tau)
    reqElbow_tau.append(elbow_tau)
    
    
    # Control System Block
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    angle_from_sensor = main_model.getCoordinateSet().get(1).getValue(state)                    # Get measured elbow angle value from the state
    velocity_from_sensor = main_model.getCoordinateSet().get(1).getSpeedValue(state)             # Get measured elbow speed value from the state
    acceleration_from_sensor = main_model.getCoordinateSet().get(1).getAccelerationValue(state)    # Get measured acceleration value from the state
    angle_from_sensor2 = main_model.getCoordinateSet().get(0).getValue(state)                   # Get measured shoulder angle value from the state
    velocity_from_sensor2 = main_model.getCoordinateSet().get(0).getSpeedValue(state)              # Get measured shoulder speed value from the state
    acceleration_from_sensor2 = main_model.getCoordinateSet().get(0).getAccelerationValue(state)   # Get measured shoulder acceleration value from the state
    
    # Gravity Compensation Control scheme 
    if cs_choice == 1:                                                                            # Gravity Compensation
        tauA = controlBlock(angle_from_sensor, External_load)
        
    
    # Actuator Module
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # This function calculates the assistive tau produced by the exosuit cables.
    T1,T2 =actuatorModule(tauA, m1, m2) 
    
        
    # Static Optimization
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # This functions calculates joint multipliers which when multiplied by muscle activations give joint torques
    aM2,aM1,Lm,Lm_max,Lm_norm,pass_tau_el,pass_tau_sh =multiplier(main_model,state)
    elbow_mul.append(aM2)
    shoulder_mul.append(aM1)
    
    # Subtraction of passive torques from ID torques to get just the torques produced by the muscles that will produce the corresponding muscle activations    
    if Assisted == 0:
        torque=np.array([(elbow_tau-pass_tau_el), (shoulder_tau-pass_tau_sh)])
    else:
        torque=np.array([(elbow_tau-tauA-pass_tau_el),(shoulder_tau-pass_tau_sh)])
    
    # This function calculates the muscle activations
    x,previous_activation = optimizer(main_model,state, aM2, aM1, torque, Only_elbow,i,previous_activation)
    desired_act.append(x)
    
    # Converting muscle activations to excitations for brain control
    ex,previous_activation = ExFromAct(main_model,x, previous_activation, timed,i)
    
    # Presribed excitations for brain control
    brain.prescribeControlForActuator("TRIlong", osim.StepFunction(0.0, timed, ex[0], ex[0]))       
    brain.prescribeControlForActuator("TRIlat", osim.StepFunction(0.0, timed, ex[1], ex[1]))  
    brain.prescribeControlForActuator("TRImed", osim.StepFunction(0.0, timed, ex[2], ex[2]))
    brain.prescribeControlForActuator("BIClong", osim.StepFunction(0.0, timed, ex[3], ex[3]))
    brain.prescribeControlForActuator("BICshort", osim.StepFunction(0.0, timed, ex[4], ex[4]))
    brain.prescribeControlForActuator("BRA", osim.StepFunction(0.0, timed, ex[5], ex[5]))
    
    # This determines the tension in the exosuit cables depending on the assistance status
    if Assisted == 1:                                                                           # If assistance is 1, the controller will send signal to generate tension in cables
       brain.prescribeControlForActuator('pact_a', osim.StepFunction(0, timed, T1, T1));        # Applying a force of T1 to the actuator pact_a, for time duration of 'timed', using the brain as controller
       brain.prescribeControlForActuator('pact_b', osim.StepFunction(0, timed, T2, T2));
    if Assisted == 0:                                                                           # If assistance is 0, controller won't send any control signal to the path actuators, resulting in no-assistance
        brain.prescribeControlForActuator('pact_a', osim.StepFunction(0, timed, 0, 0))          # 0N force is applied to pact_a actuator
        brain.prescribeControlForActuator('pact_b', osim.StepFunction(0, timed, 0, 0))          # 0N force is applied to pact_b actuator
 
    
    # Forward Dynamics  
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    state.setTime(0.0)
    manager = osim.Manager(main_model,state)                           # Manager will be the forward dynamics integrator
    state = manager.integrate(timed)                                   # Integrate for the time duration of 'timed'
    main_model.realizeVelocity(state)                                  # Realizing the state to velocity and acceleration stage, will ensures that the velocity dependent terms are calculated and stored in state
    main_model.realizeAcceleration(state)
    main_model.equilibrateMuscles(state)
    
    m1 = pact_a.computeMomentArm(state, main_model.getCoordinateSet().get(1));       # Moment arm is computed from OpenSim of flexor cable about the elbow joint
    m2 = pact_b.computeMomentArm(state, main_model.getCoordinateSet().get(1));       # Moment arm is computed from OpenSim of extensor cable about the elbow joint
    
    # Calculating the Forward Dynamics Torque for comparison with ID torque to check for the validity of the simulation flow
    tot=0
    for k in range(6):
          previous_activation[k]= main_model.getMuscles().get(k).getActivation(state)
          MA_el[k]= main_model.getMuscles().get(k).computeMomentArm(state, main_model.getCoordinateSet().get(1)); # Getting moment arm of every muscle about elbow joint
          MA_sh[k] = main_model.getMuscles().get(k).computeMomentArm(state, main_model.getCoordinateSet().get(0)); # Getting moment arm of every muscles about shoulder joint
          Fm[k]=main_model.getMuscles().get(k).getTendonForce(state)
          T_el_FD[k]= MA_el[k]*(Fm[k]);
          tot=tot+T_el_FD[k]
        
    fd_Elbow_torque.append(tot)        # FD torque at every time step stored in a list
    if Assisted==0:                 #When no assistance is required, there will be no tension T1, T2 in the exosuit cables
        T1=0
        T2=0
    
    Assistive_tau.append(T1*m1+T2*m2)  # Assistive torque at every time step stored in a list
    total_tau=[x + y for x, y in zip(fd_Elbow_torque, Assistive_tau)]  # Total torque= FD torque (Human Torque) + Exosuit torque (Assistive torque)
    
    # Storing variables at every time step for plotting and reference
    measured_shoulder_trajectory.append(main_model.getCoordinateSet().get(0).getValue(state))  # Shoulder joint angle after forward dynamics.
    measured_trajectory.append(main_model.getCoordinateSet().get(1).getValue(state))           # Elbow joint angle after forward dynamics.
    desired_trajectory.append(e_angle[i])                                                      # Storing the desired trajectory into a new variable so that the dimension can be compatible
    Excitation.append(ex)                                                                      # Excitations derived from muscle activations
    t.append(time[i])                                                                          # storing time
    
    # Metabolic Cost Calculation
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    MCT.append(main_model.getProbeSet().get(0).getProbeOutputs(state).get(0))                 # Using OpenSim capability to calculate the metabolic cost with and without Assistance                                      
    
# Plotting of the results
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
plt.figure()
plt.plot(t, measured_trajectory, t, desired_trajectory)
plt.title("Elbow Trajectory for {} kg load, with Assistance={}".format(External_load, Assisted))
plt.xlabel('Time (sec.)');
plt.ylabel('Elbow Angle (rad.)');
plt.legend(['measured_trajectory','desired_trajectory'])

# For ID Elbow Torque
plt.figure()
plt.plot(t, reqElbow_tau)
plt.title("ID Elbow Torque for {} kg load, with Assistance = {}".format(External_load, Assisted))
plt.xlabel('Time (sec.)');
plt.ylabel('Torque (Nm)');

# For FD Elbow Torque
plt.figure()
plt.plot(t, fd_Elbow_torque)
plt.title("FD Elbow Torque for {} kg load, with Assistance = {}".format(External_load, Assisted))
plt.xlabel('Time (sec.)');
plt.ylabel('Torque (Nm)');

plt.figure()
plt.plot(t, fd_Elbow_torque,t,reqElbow_tau,t, Assistive_tau,t,total_tau)
plt.title("Comparison Elbow Torque for {} kg load, with Assistance = {}".format(External_load, Assisted))
plt.xlabel('Time (sec.)');
plt.ylabel('Torque (Nm)');
plt.legend(['FD Elbow Torque','ID Elbow Torque','Assistive_tau','Net Elbow tau'])

# # For Muscle Activations
plt.figure()
plt.plot(t, desired_act)
plt.title("Muscle Activations for {} kg load, with Assistance = {}".format(External_load, Assisted))
plt.xlabel('Time (sec.)');
plt.ylabel('Muscle Activations')
plt.legend(['Triceps Long', 'Triceps Lateral', 'Triceps Medial','Biceps Long', 'Biceps Short', 'Brachialis'])


# # For Metabolic Cost
plt.figure()
MCT=pd.DataFrame(MCT)
MCT=MCT.interpolate(method='linear', axis=0).ffill()
MCT= MCT.rolling(8).mean()
plt.plot(t, MCT)
plt.title("Metabolic Cost for {} kg load, with Assistance = {}".format(External_load, Assisted))
plt.xlabel('Time (sec.)')
plt.ylabel('Metabolic Cost (J/s)');


# Combine all the operations and display
plt.show()