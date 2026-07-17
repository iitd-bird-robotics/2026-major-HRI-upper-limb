import numpy as np
def multiplier(main_model,state):
    
    main_model.realizeVelocity(state)
    # Initialization
    aM1=np.zeros(6)
    aM2=np.zeros(6)
    Lm=np.zeros(6)
    Lm_max=np.zeros(6)
    Lm_norm=np.zeros(6)
    F_l_norm=np.zeros(6)
    F_v_norm=np.zeros(6)
    Fm_max=np.zeros(6)
    MA_el=np.zeros(6)
    MA_sh=np.zeros(6)
    Fpe=np.zeros(6)
    eisom=0.6;
    kpe=4
    vel=np.zeros(6)
    Vmax=np.zeros(6)
    vel_norm=np.zeros(6)
    for j in range(6):
          # jth_muscle=main_model.getMuscles().get(j)
           # Muscle length from OpenSim
          Lm[j]=main_model.getMuscles().get(j).getFiberLength(state)
           #Lm.append(j)
  
           # Getting optimal fiber lengths of muscle from OpenSim
          Lm_max[j]=main_model.getMuscles().get(j).getOptimalFiberLength()
           #Lm_max.append(j)
  
           # Normalized Muscle length
          Lm_norm[j]=Lm[j]/Lm_max[j]
  
           # Calculating muscle Force-length multiplier 
          # F_l_norm[j]=np.exp(-((Lm_norm[j]-1)**2)/0.45)
          F_l_norm[j]=main_model.getMuscles().get(j).getActiveForceLengthMultiplier(state)
          # vel[j]=main_model.getMuscles().get(j).getFiberVelocity(state)
          # Vmax[j]=main_model.getMuscles().get(j).getMaxContractionVelocity()
          # vel_norm[j]=vel[j]/(Vmax[j]*Lm_max[j])
           # Getting Muscle velocity-force multiplier
          F_v_norm[j] = main_model.getMuscles().get(j).getForceVelocityMultiplier(state)
          
           # Getting maximum isometric force of muscle from OpenSim
          Fm_max[j]=main_model.getMuscles().get(j).getMaxIsometricForce()                                       # Getting Max Isometric force of every muscles
          
          MA_el[j]=main_model.getMuscles().get(j).computeMomentArm(state, main_model.getCoordinateSet().get(1))*main_model.getMuscles().get(j).getCosPennationAngle(state) # Getting moment arm of every muscle about elbow joint
          MA_sh[j]=main_model.getMuscles().get(j).computeMomentArm(state, main_model.getCoordinateSet().get(0))*main_model.getMuscles().get(j).getCosPennationAngle(state)# Getting moment arm of every muscles about shoulder joint

   
          aM2[j]=MA_el[j]*Fm_max[j]*(F_l_norm[j]*F_v_norm[j])              # This is elbow multiplier which means, if it is multiplied with a vector comprising of acivations of the six muscles, will give the value of elbow joint moment
          aM1[j]=MA_sh[j]*Fm_max[j]*(F_l_norm[j]*F_v_norm[j])             # This is shoulder multiplier which means, if it is multiplied with a vector comprising of acivations of the six muscles, will give the value of shoulder joint moment
          
          # Calculating passive force multiplier
          # if Lm_norm[j] > (1+eisom):
          #       Fpe[j]=Fm_max[j]*(1+(kpe/eisom)*(Lm_norm[j]-(1+eisom)));
          # else:
          #       Fpe[j]=Fm_max[j]*((np.exp(kpe*(Lm_norm[j]-1)/eisom))/np.exp(kpe));
          Fpe[j]=main_model.getMuscles().get(j).getPassiveFiberForceAlongTendon(state)

    pass_tau_el=np.dot(MA_el,Fpe)   # Passive torque at the elbow joint
    pass_tau_sh=np.dot(MA_sh,Fpe)   # Passive torque at the shoulder joint
   
    return [aM2,aM1,Lm,Lm_max,Lm_norm,pass_tau_el,pass_tau_sh];