import numpy as np
def ExFromAct(main_model,x, previous_activation, timed,i):
       ex=0.0001 * np.ones(6)
       act_der=0.0001 * np.ones(6)
       for a in range(6):
           act_der[a]=(x[a]-previous_activation[a])/timed
           if act_der[a] > 0:                                                      # If rate of change of activation is positive => u > a (since activation follows excitation), so the first condition (from formula given above)
                ex[a]=(act_der[a] * 0.01)*(0.5 + 1.5*x[a]) + x[a]
           else:
                ex[a]=(act_der[a] * 0.04)/(0.5 + 1.5*x[a]) + x[a]  
       return [ex,previous_activation];