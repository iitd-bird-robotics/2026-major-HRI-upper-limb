import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
from scipy.ndimage import convolve1d
from matplotlib import animation
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

        filt_weight = gaussian(50, 8)

    # for frame in range(xval.shape[0]):
    #     a=xval[frame]
    #     b=a.reshape(1,a.shape[0],a.shape[1])
    #     # Evaluate the restored model
    #     output = BiCurNet_trained.predict(b)
    #     updated_angle = output.transpose()
    #     filt_angle = filters.convolve1d(updated_angle, filt_weight / filt_weight.sum())
    #     # Update the Y-axis data with new values (example: sinusoidal function with a phase shift)
    #     # new_y_data = updated_angle(x_data + frame)
    #     # line.set_ydata(new_y_data)
        
    #     time.sleep(1.5)


        # Parameters
        n_frames = 200  # Total number of frames
        frame_duration_ms = 1000  # Time interval between updates in milliseconds

        # Initialize data and index
        x_data = np.arange(n_frames)
        y_data = np.zeros(n_frames)  # Initialize with zeros, this will be updated

        # Create a figure and axis for plotting
        fig, ax = plt.subplots()
        ax.set_ylim(0, 0)
        line, = ax.plot(x_data, y_data)

        # Initialize a counter variable to keep track of frames
        frame_counter = 0
        predicted_angle=[]
        comp_yval=[]
        t=[]
        time=[]
        # Function to update the plot
        # for frame in range(53):
            #     global frame_counter  # Access the counter variable
        
        for frame_counter in range(8):
            # Get the frame data from xval using the counter (replace with your data)
                a = xval[frame_counter]
                b = a.reshape(1, a.shape[0], a.shape[1])
                output = BiCurNet_trained.predict(b)
                updated_angle = output.transpose()
                filt_angle = convolve1d(updated_angle, filt_weight / filt_weight.sum(), axis=-1, mode='constant')
            
                # Update the plot data
                line.set_ydata(filt_angle.squeeze())  # Update the Y-axis data
                # plt.pause(frame_duration_ms / 1000)  # Pause for the specified duration
                fa=filt_angle.reshape(filt_angle.shape[0])
                predicted_angle.extend(fa)
                comp_yval.extend(yval[frame_counter])
                # time.extend(x_data*(1.6/(4*20)))
                # plt.plot(filt_angle)
                # Increment the counter
                frame_counter += 1
        # return [np.array(predicted_angle),comp_yval];
        # Create a timer to update the plot periodically
    # ani = animation.FuncAnimation(fig, update, frames=n_frames, repeat=False, interval=frame_duration_ms)
    #Flipping output and un-normalizing it
    #converting list into array
    # pred_angle,ref_angle=update
        time=np.zeros(1200)
        for t in range(1,len(time)-1):
            time[t+1]=time[t]+(9.6/1200)
        
        # time=np.arange(filt_angle.shape[0]*(4.8/800))
        pred_angle=np.array(predicted_angle)
        ref_ang=np.array(comp_yval)
        #flipping both ref and predicted output
        pred=pred_angle-np.max(pred_angle)
        ref=ref_ang-np.max(ref_ang)
        pred=pred*(-1)
        ref=ref*(-1)
        #Un-normalizing both ref and predicted data
        Pred_unNorm=pred*(2.18166)+0.436332
        Ref_unNorm=ref*(2.18166)+0.436332
        # # Show the plot
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Updating Plot Data with Matplotlib")
        plt.grid(True)
        plt.show()
        # pred_ang=np.array(predicted_angle)
        # ref_ang=np.array(comp_yval)
        # time=x_data
        s_angle=np.zeros(1200)
        e_angle=Pred_unNorm    
        # pred_ang=np.array(predicted_angle)
        # ref_ang=np.array(comp_yval)
    return[s_angle,e_angle,time]
