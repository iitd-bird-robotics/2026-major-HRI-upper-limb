import numpy as np
import pandas as pd
import tensorflow as tf
# import matplotlib.pyplot as plt
# import time
# from matplotlib.animation import FuncAnimation
from scipy.signal.windows import gaussian, hamming
from scipy.ndimage import filters
def BrainControlledInterface(BCI):

    mydata = pd.read_excel("Trajectory2.xlsx",'Sheet1')
    # myElbow_data = pd.read_excel("C:/Users/Dell/Downloads/Filter_results (1).xlsx",'Sheet4')
        
    if BCI==0:
    
        time=mydata.iloc[0:120,0]
        s_angle=mydata.iloc[0:120,1]
        e_angle=mydata.iloc[0:120,2]
    
    else:
    
 
        # from arm26_utilities import BiCurNet_anant_240_1600
        # def BCI_module(xval,yval,yval_flatten,pcc):
        xval = np.load('BiCurNet_anant_240_1600_xval.npy')

        yval = np.load('BiCurNet_anant_240_1600_yval.npy')
        yval_flatten = np.load('BiCurNet_anant_240_1600_yval_flatten.npy')
        pcc = np.load('BiCurNet_anant_240_1600_pcc.npy')

        BiCurNet_trained = tf.keras.models.load_model('BiCurNet_anant_240_1600')

        # Show the model architecture
        BiCurNet_trained.summary()

        filt_weight = gaussian(10, 8)

        def rescale_array(arr, old_min, old_max, new_min, new_max):
            # Check if any value in the array is outside the old range
            if np.any((arr < old_min) | (arr > old_max)):
                raise ValueError("Input values are not within the old range")

            # Calculate the ratios for rescaling
            ratios = (arr - old_min) / (old_max - old_min)

            # Map the ratios to the new range to get the rescaled array
            rescaled_array = new_min + ratios * (new_max - new_min)
            return rescaled_array

        # Define the old range and new range
        old_min = -0.15
        old_max = 0.15
        new_min = 0.0
        new_max = 0.9
        predicted_angle=[]
        comp_yval=[]
        t=[]
        time=[]
        for frame in range(1,2):
            a=xval[frame]
            b=a.reshape(1,a.shape[0],a.shape[1])
            # Evaluate the restored model
            output = BiCurNet_trained.predict(b)
            updated_angle = output.transpose()
            filt_angle = filters.convolve1d(updated_angle, filt_weight / filt_weight.sum())
            # scaled_angle = rescale_array(filt_angle, old_min, old_max, new_min, new_max)
            # ratios = (filt_angle - old_min) / (old_max - old_min)

            # # Map the ratios to the new range to get the rescaled array
            # rescaled_array = new_min + ratios * (new_max - new_min)
            fa=filt_angle.reshape(filt_angle.shape[0])
            predicted_angle.extend(fa)
            comp_yval.extend(yval[frame])
            frame += 1

        time=np.zeros(200)
        for t in range(1,len(time)-1):
            time[t+1]=time[t]+(9.6/1200)
        
        # time=np.arange(filt_angle.shape[0]*(4.8/800))
        ref_ang=np.array(comp_yval)
        pred=predicted_angle-np.max(predicted_angle)
        ref=ref_ang-np.max(ref_ang)
        pred_angle=pred*(-1)
        ref=ref_ang*(-1)
        #Un-normalizing both ref and predicted data
        Pred_unNorm=pred_angle*(2.18166)+0.436332
        Ref_unNorm=ref*(2.18166)+0.436332
        # # Show the plot
        # plt.xlabel("X-axis")
        # plt.ylabel("Y-axis")
        # plt.title("Updating Plot Data with Matplotlib")
        # plt.grid(False)
        # plt.show()
        # pred_ang=np.array(predicted_angle)
        # ref_ang=np.array(comp_yval)
        # time=x_data
        s_angle=np.zeros(200)
        e_angle=Pred_unNorm   
        # pred_ang=np.array(predicted_angle)
        # ref_ang=np.array(comp_yval)
    return[s_angle,e_angle,time]
