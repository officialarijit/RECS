import socket
import pickle

#============================
# Import important libraries
#============================
import pandas as pd 
import numpy as np
import pywt
from river import metrics
import time
import datetime
import matplotlib.pyplot as plt

from mlxtend.plotting import plot_confusion_matrix
from window_slider import Slider
from sklearn import preprocessing

from CustomLogisticRegression import LogisticRegression
from EEGFeature import WaveletEntropyFeatures


##========================================
# Initial parameters
##========================================
participant = 32 #participants
num_videos = 40 # total number of videos
segment_in_sec = 60 #time segment 
overlap_count = 0
classifier = 'logistic regression-SGD'
init_i = 0
lr = 0.05 #Learning rate
epoch = 1 #epoch is 1 because the model will be trained only once

start_time = time.time() #timer start now

all_Results =pd.DataFrame([])

#================================================
# Performance matric declaration here
#================================================

acc_val = metrics.Accuracy() #Accuracy
f1m_val = metrics.F1() #F1 measure 
mse_val = metrics.MSE() #MSE error  
acc_aro = metrics.Accuracy() #Accuracy
f1m_aro = metrics.F1() #F1 measure
mse_aro = metrics.MSE() #MSE error  


HEADERSIZE = 10

port = 1245

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), port))

start_time = time.time() #timer start now


while True:
    full_msg = b''
    new_msg = True
    while True:
        msg = s.recv(1280) #total number of times we are going to receive the data

        if new_msg:
            # print("new msg len:",msg[:HEADERSIZE])
            msglen = int(msg[:HEADERSIZE])
            new_msg = False

        # print(f"full message length: {msglen}")

        full_msg += msg

        # print(len(full_msg))

        if len(full_msg)-HEADERSIZE == msglen:
            print("full msg recvd")
            # print(full_msg[HEADERSIZE:])
            # print(
            rcv_data = pickle.loads(full_msg[HEADERSIZE:]) #received data from buffer
            pvc = rcv_data["pvc"] #which video which participants which channel
            data = rcv_data["data"] #signal recordings received from server
            emo_label = rcv_data["emotion"] #emotion labels

            p = pvc.iloc[0] #participant
            v = pvc.iloc[1] #video

            ptr = "Data received for participant: "+str(p)+" video:"+str(v)
            print(ptr)
            

            new_msg = True
            full_msg = b""

            #===================================================================
            # ML approach starts from here 
            #===================================================================

            #==================================================================
            # Useful For binary-class classification
            #==================================================================

            val = emo_label.iloc[0]
            aro = emo_label.iloc[1]

            #valence emotion maping 0-> low valence and 1-> high valence

            if (val >=5):
                vl = 1 #high valence
            else:
                vl = 0 #low valence

            #arousal emotion maping 0-> low arousal and 1-> high high arousal
            if (aro >=5):
                al = 1 #high arousal
            else:
                al = 0 #low arousal

            y_act_aro = al
            y_act_val = vl
            #==================================================================

            #=================================================
            # Feature extraction from EEG
            #=================================================
            features = WaveletEntropyFeatures(np.array(data), level = 5, wavelet = 'db4')
            features = np.array([features]) #ERG feature vector
#                 features = np.array(preprocessing.normalize(features)) #EEG normalized features

            #===============================================================
            # Emotion Classification --> Valence and Arousal
            #===============================================================

            if init_i == 0: #For the first time model will return 9 or None

                #=======================================
                # Model LR regression model declaration
                #=======================================
                
                model_val = LogisticRegression(features.shape[1],lr,epoch)
                model_aro = LogisticRegression(features.shape[1],lr,epoch)
                print('Model initialization done..!')
        
                y_pred_val = 3
                y_pred_aro = 3
                
                model_val.fit_once(features,y_act_val) #valence classifier 
                model_aro.fit_once(features,y_act_aro) #arousal classifier

                init_i += 1

            else:

                y_pred_val = 1 if model_val.predict_once(features) >0.5 else 0
                y_pred_aro = 1 if model_aro.predict_once(features) >0.5 else 0

                model_val.fit_once(features,y_act_val) #valence classifier 
                model_aro.fit_once(features,y_act_aro) #arousal classifier


            
            acc_val.update(y_act_val, y_pred_val)  # update the accuracy metric
            f1m_val.update(y_act_val, y_pred_val) #update f1 measure metric
            mse_val.update(y_act_val, y_pred_val) #update mse error

            acc_aro.update(y_act_aro, y_pred_aro)  # update the accuracy metric
            f1m_aro.update(y_act_aro, y_pred_aro) #update f1 measure metric
            mse_aro.update(y_act_aro, y_pred_aro) #update mse error

            window_number +=1

            tmp_results = pd.DataFrame([{
                'person':p,
                'video':v,
                'window_number': window_number,
                'acc_val':acc_val.get(),
                'f1m_val':f1m_val.get(),
                'mse_val':mse_val.get(),
                'acc_aro':acc_aro.get(),
                'f1m_aro':f1m_aro.get(),
                'mse_aro':mse_aro.get()
            }])

            all_Results = pd.concat([all_Results,tmp_results], axis=0)
            
                
elapsed = time.time() - start_time



SIZE = 14
plt.rc('font', size=SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)    # legend fontsize
plt.rc('figure', titlesize=SIZE)  # fontsize of the figure title
# plt.rc('font', weight='bold')


fig, (ax1, ax2) = plt.subplots(figsize=(11, 5), nrows=2,  sharex=True)

# plt.subplot(3, 1, 1)
ax1.plot(range(0,all_Results.shape[0]), all_Results.acc_val)
ax1.plot(range(0,all_Results.shape[0]), all_Results.acc_aro)
# Function add a legend
ax1.legend(["valence accuacy", "arousal accuracy"], loc="lower right")
ax1.set_ylabel('accuracy')
ax1.set_xlim(0,all_Results.shape[0])


# plt.subplot(3, 1, 2)
ax2.plot(range(0,all_Results.shape[0]), all_Results.f1m_val, color="blue")
ax2.plot(range(0,all_Results.shape[0]), all_Results.f1m_aro, color="red")
# Function add a legend
ax2.legend(["valence f1-score", "arousal f1-score"], loc="lower right")
ax2.set_ylabel('f1-score')
ax2.set_xlabel('number of instances')
ax2.set_xlim(0,all_Results.shape[0])
    
# # plt.subplot(3, 1, 3)
# ax3.plot(range(0,all_Results.shape[0]), all_Results.mse_val,color='cyan')
# ax3.plot(range(0,all_Results.shape[0]), all_Results.mse_aro, color='magenta')
# # Function add a legend
# ax3.legend(["valence mse", "arousal mse"], loc="upper right")

# plt.show()

plt.savefig(str(segment_in_sec)+'_sec_RECS-performance.png', dpi=300, bbox_inches='tight')

all_Results.to_csv(str(segment_in_sec)+'_sec_RECS-2024-updated-work.csv', index=None)

