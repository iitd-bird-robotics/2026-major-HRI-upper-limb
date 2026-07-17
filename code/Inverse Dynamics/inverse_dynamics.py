import opensim as osim
def IDSolver(ID_model, ID_state, main_model, state, e_angle, s_angle, timed, i):
    
    ID = osim.InverseDynamicsSolver(ID_model)
    
    s_speed = (s_angle[i+1]-s_angle[i])/timed                                          # calculating speed of shoulder joint
    s_alpha = ((s_angle[i+1]+s_angle[i-1])-2*s_angle[i])/(timed**2)                    # calculating acceleration of shoulder joint
    e_speed =(e_angle[i+1]-e_angle[i])/timed                                           # calculating speed of elbow joint
    e_alpha = ((e_angle[i+1]+e_angle[i-1])-2*e_angle[i])/(timed**2)                    # calculating acceleration of elbow joint
    elb_accel_ref=e_alpha
 
    angle_from_sensor = main_model.getCoordinateSet().get('r_elbow_flex').getValue(state)                  # Get current elbow angle value from the state
    velocity_from_sensor = main_model.getCoordinateSet().get('r_elbow_flex').getSpeedValue(state)          # Get current elbow speed value from state
    angle_from_sensor2 = main_model.getCoordinateSet().get('r_shoulder_elev').getValue(state)              # Get current elbow angle value from the state
    velocity_from_sensor2 = main_model.getCoordinateSet().get('r_shoulder_elev').getSpeedValue(state)      # Get current elbow speed value from state
    
    #Putting the angle and velocity values into the current state of the model
    ID_model.getCoordinateSet().get('r_elbow_flex').setValue(ID_state, angle_from_sensor)               # Setting the current angle from FD model of elbow joint to the Inverse Dynamics model
    ID_model.getCoordinateSet().get('r_elbow_flex').setSpeedValue(ID_state, velocity_from_sensor)       # Setting the current velocity from FD model of elbow joint to the Inverse Dynamics model
    ID_model.getCoordinateSet().get('r_shoulder_elev').setValue(ID_state, angle_from_sensor2)           # Setting the current angle from FD model of shoulder joint to the Inverse Dynamics model
    ID_model.getCoordinateSet().get('r_shoulder_elev').setSpeedValue(ID_state, velocity_from_sensor2)   # Setting the current velocity from FD model of shoulder joint to the Inverse Dynamics model

    kp=100
    kv=20
    
    # Error_tau = kp(delta.theta) + kv*(delta.theta.dot)
    elbow_error_acc= kp*(e_angle[i]-angle_from_sensor) + kv*(e_speed-velocity_from_sensor)
    shoulder_error_acc = kp*(s_angle[i]-angle_from_sensor2) + kv*(s_speed-velocity_from_sensor2)
    e_alpha = e_alpha + elbow_error_acc 
    elb_accel_new=e_alpha                                                               # Acceleration = desired_Acceleration + kp(delta.theta) + kv*(delta.theta.dot)
    s_alpha = s_alpha + shoulder_error_acc
    
    #Making a two dimensioanl vector to contain two values : shoulder and elbow acceleration.
    u_dot = osim.Vector(2,0)                                                            #Making a vector with two elements
    u_dot.set(0, s_alpha)                                                               # Putting the calculated value of shoulder acceleration at the first position of the vector
    u_dot.set(1, e_alpha)                                                               # Putting the calculated value of elbow acceleration at the second position of the vector
                                                              
    
    #Providing the inverse dynamics solver with the state and the joint acceleration vector to get joint torque vector
    tau = ID.solve(ID_state, u_dot)
 
    # Getting individual joint torques from the torque vector
    shoulder_tau=(tau.get(0))                                                          # First element of the "tau" vector gives the value of shoulder torque
    elbow_tau=(tau.get(1))                                                             # Second element of the "tau" vector gives the value of elbow torque
    
    return[shoulder_tau, elbow_tau,elb_accel_new,elb_accel_ref];