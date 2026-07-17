def actuatorModule(tauA, m1, m2):
     if tauA >= 0:
         T1 = tauA/m1        #T1 is force in flexor cable and m1 is moment arm of the flexor cable about the elbow joint
         T2 = 0
     else:
         T1 = 0
         T2 = tauA/m2        #T2 is force in extensor cable and m2 is moment arm of the extensor cable about the elbow joint
 
     return [T1,T2];