import numpy as np
def controlBlock(angle_from_sensor, External_load):
    m_e = 1.83
    m_l = External_load                                                                     # Mass of extra load
    l_c = 0.181479                                                             # Center of mass location of the forearm wrt elbow joint
    l_l = 0.35                                                                  # Center of mass location of the extra load wrt elbow joint
    g = 9.81
    # tauA calculation in Gravity Compensation
    tau_a_grav = m_e*g*l_c*(np.sin(angle_from_sensor)) + m_l*g*l_l*(np.sin(angle_from_sensor))          # Gravitational component of total elbow torque
    # tau_a_grav =  m_l*g*l_l*sin(angle_from_sensor);          % Gravitational component of total elbow torque
    tauA = tau_a_grav
    
    return tauA;