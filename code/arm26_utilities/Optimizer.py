from scipy.optimize import minimize
def optimizer(main_model,state, aM2, aM1, torque, Only_elbow,i,previous_activation):
    
    def objective(x):
        p = 2
        tot = 0
        for a in range(6):
            tot = tot + x[a] ** p
        return tot
    
    x0 = previous_activation
                                                       
    bound = (0.01, 1)
    bounds = [bound, bound, bound, bound, bound, bound]
    
    if Only_elbow == 1:
        A = aM2                           # Multipliers which when multiplied by activation matrix gives torque
        b = torque[0]                                                               # Torque required at elbow                                                                   
    elif Only_elbow == 0:
        A = [(aM2),(aM1)]                 # Multipliers which when multiplied by activation matrix gives torque
        b = torque                                                                  # Torques required at elbow and shoulder
    
    eq_cons1 =[ {'type': 'eq', 'fun': lambda x: A@x-b}]
    
    res= minimize(objective,x0=x0,bounds=bounds, constraints=eq_cons1)
    
    return [res.x,previous_activation];