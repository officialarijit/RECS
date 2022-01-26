import socket
import pickle

import pandas as pd 
import numpy as np
import pywt

import time


from creme import metrics
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report

class logistic_regression_val:
    
    def __init__(self,iterations,alpha):
        self.iterations=iterations  #choosing the number of iterations (Hyperparameter)
        self.alpha=alpha       #choosing alpha(Hyperparameter) 
    
    def sigmoid(self,z):
        return(1/(1+np.exp(-z)))    #sigmoid (Link Fucntion)
    
    def fit_once(self,x,y):              #(X-data for training, y - Output) 
        m=x.shape[0]                
        self.w=np.random.randn(x.shape[1],1)  #Initializing the weight
        
        cost_vals=[] 
        for i in range(2):     #For each number of iterations
            a= np.dot(x,self.w)            #multiplying the weights with the Feature values and summing them up
            z=self.sigmoid(a)         #Using link function to transform the data
            
            cost = (-1/m) *( np.dot(y,np.log(z))+(np.dot((1-y),np.log(1-z))))  #Calculating the cost function
            
            cost_vals.append(cost)        #Creating a list with all cost values for each iteration
            
            dw = np.dot(x.T,z-np.array([y])).mean()  #Calculating the gradient
            
            self.w=self.w-(self.alpha*dw)         #updating the weights
#             print(self.w.shape)
        return self
    
    def predict_once(self,x,threshold,c):
        
        if c == 0: #the model is first visiting the data
            return (0)
        else:
            probability=self.sigmoid(np.dot(x,self.w))  #predicting a new set of values based on the training 

            if(probability>threshold):
                return (1)
            else:
                return (0)


class logistic_regression_aro:
    
    def __init__(self,iterations,alpha):
        self.iterations=iterations  #choosing the number of iterations (Hyperparameter)
        self.alpha=alpha       #choosing alpha(Hyperparameter) 
    
    def sigmoid(self,z):
        return(1/(1+np.exp(-z)))    #sigmoid (Link Fucntion)
    
    def fit_once(self,x,y):              #(X-data for training, y - Output) 
        m=x.shape[0]                
        self.w=np.random.randn(x.shape[1],1)  #Initializing the weight
        
        cost_vals=[] 
        for i in range(2):     #For each number of iterations
            a= np.dot(x,self.w)            #multiplying the weights with the Feature values and summing them up
            z=self.sigmoid(a)         #Using link function to transform the data
            
            cost = (-1/m) *( np.dot(y,np.log(z))+(np.dot((1-y),np.log(1-z))))  #Calculating the cost function
            
            cost_vals.append(cost)        #Creating a list with all cost values for each iteration
            
            dw = np.dot(x.T,z-np.array([y])).mean()  #Calculating the gradient
            
            self.w=self.w-(self.alpha*dw)         #updating the weights
#             print(self.w.shape)
        return self
    
    def predict_once(self,x,threshold,d):
        
        if d == 0: #the model is first visiting the data
            return (0)
        else:
            probability=self.sigmoid(np.dot(x,self.w))  #predicting a new set of values based on the training 

            if(probability>threshold):
                return (1)
            else:
                return (0)

HEADERSIZE = 10

port = 1245

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), port))

start_time = time.time() #timer start now


#==========================
#  Feature extraction EEG
#==========================

# wavelet entropy python
def WE(y, level = 5, wavelet = 'db4'):
    from math import log
    fv = []
    n = len(y)

    sig = y

    ap = {}

    for lev in range(0,level):
        (y, cD) = pywt.dwt(y, wavelet)
        ap[lev] = y

    # Energy

    Enr = np.zeros(level)
    for lev in range(0,level):
        Enr[lev] = np.sum(np.power(ap[lev],2))/n

    Et = np.sum(Enr)

    Pi = np.zeros(level)
    for lev in range(0,level):
        Pi[lev] = Enr[lev]/Et

    we = - np.sum(np.dot(Pi,np.log(Pi)))

    return np.mean(Enr), np.mean(Pi), we


#================================
# Optimizer 
#================================
l_rate = 0.05 #Learning rate
n_epoch = 1 #epoch is 1 because the model will be trained only once
c = 0
d = 0

#=======================================
# Model LR regression model declaration
#=======================================
model_val = logistic_regression_val(n_epoch,l_rate) # model creation
model_aro = logistic_regression_aro(n_epoch,l_rate) # model creation
classifier = 'logistic regression-SGD'

#================================================
# Performance matric declaration here
#================================================

acc_val = metrics.Accuracy() #Accuracy

f1m_val = metrics.F1() #F1 measure  

acc_aro = metrics.Accuracy() #Accuracy

f1m_aro = metrics.F1() #F1 measure


#=================================================
#===Data reading and main program start===========
#=================================================

global aro_actual_class_label, aro_predicted_class_labels, val_actual_class_label, val_predicted_class_labels, dframe_aro, dframe_val

participant = 32 #participants

aro_actual_class_label = []
aro_predicted_class_labels = []

val_actual_class_label = []
val_predicted_class_labels = []

dframe_val = []
dframe_aro = []


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

            #===========================================================
            # Feature extraction from EEG using Wavelet Transformation #
            #===========================================================

            tmpfeature = []
            features = []
            for i in range(data.shape[0]):
                we,m_wer,m_enr = WE(data.iloc[i,range(0,8064)])

                features = features+[we,m_wer,m_enr]

            tmpfeature.append(features) 
            
            #=================================================
            #emotion labels (valence, arousal) mapping 0-1
            #=================================================
            val = df.iloc[1,8067]
            aro = df.iloc[1,8068]

            #valence emotion maping 0-> low valence and 1-> high valence

            if (val >5):
                vl = 1 #high valence
            else:
                vl = 0 #low valence

            #arousal emotion maping 0-> low arousal and 1-> high high arousal
            if (aro >5):
                al = 1 #high arousal
            else:
                al = 0 #low arousal


            #############################################
            # Valence classification model
            #############################################
            x = np.array(tmpfeature) #feature vector 

            y_act_val = vl

            #Test the model first 
            if c ==0:
                y_pred_val = model_val.predict_once(x,0.5,c)
            else:
                y_pred_val = model_val.predict_once(x,0.5,c)


            #Train the model once
            model_val.fit_once(x,y_act_val)

            acc_val = acc_val.update(y_act_val, y_pred_val)  # update the accuracy metric

            f1m_val = f1m_val.update(y_act_val, y_pred_val) #update f1 measure metric

            pr1_val = "Valence Accuracy:" + str(acc_val.get())
            pr2_val = "Valence F1 score:"+str(f1m_val.get())

            scr_val = np.array([p,v,acc_val.get(), f1m_val.get(), y_act_val, y_pred_val]) #storing ACC anf F1 results

            dframe_val.append(scr_val)

            #==============================================================
            # Storing actual and predicted valence classification results
            #==============================================================
            val_actual_class_label.append(y_act_val)
            val_predicted_class_labels.append(y_pred_val)

            #############################################
            # Arousal classification model
            #############################################

            y_act_aro = al

            #Test the model first 
            if d ==0:
                y_pred_aro = model_aro.predict_once(x,0.5,d)
            else:
                y_pred_aro = model_aro.predict_once(x,0.5,d)

            #Train the model once 
            model_aro.fit_once(x,y_act_aro)

            print(y_act_aro)
            print(y_pred_aro)


            acc_aro = acc_aro.update(y_act_aro, y_pred_aro)  # update the accuracy metric

            f1m_aro = f1m_aro.update(y_act_aro, y_pred_aro) #update f1 measure metric

            pr1_aro = "Arousal Accuracy:" + str(acc_aro.get())
            pr2_aro = "Arousal F1 score:"+str(f1m_aro.get())

            scr_aro = np.array([p,v,acc_aro.get(), f1m_aro.get(), y_act_aro, y_pred_aro]) #storing ACC anf F1 results

            dframe_aro.append(scr_aro)

            #==============================================================
            # Storing actual and predicted valence classification results
            #==============================================================

            aro_actual_class_label.append(y_act_aro)
            aro_predicted_class_labels.append(y_pred_aro)

            print(p_v)
            print(pr1_val)
            print(pr2_val)

            print(pr1_aro)
            print(pr2_aro)


            print('-----------------------------------------------')
            v = v+1
            c = c+1
            d = d+1

elapsed = time.time() - start_time
print('Elapsed time:',elapsed)
fname_aro = 'TEST_12_JAN_2020_valence_emo_all_person_'+'_' +classifier+'_results.csv'
np.savetxt(fname_aro,dframe_val, delimiter ="\t", fmt =['%d', '%d', '%f', '%f', '%s', '%s'], 
    header='Person, Video, Acc, F1, y_act_val, y_pred_val')

fname_aro = 'TEST_12_JAN_2020_arousal_emo_all_person_'+'_' +classifier+'_results.csv'
np.savetxt(fname_aro,dframe_aro, delimiter ="\t", fmt =['%d', '%d', '%f', '%f', '%s', '%s'], 
    header='Person, Video, Acc, F1, y_act_val, y_pred_val')


#========================================================
# Classifiers Report Showing
# Performance metrics
#========================================================

#============================================
# Valence Classification Report
#============================================
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


y_true = val_actual_class_label #Acutal class labels

y_pred = val_predicted_class_labels #Predicted Class labels

cm = confusion_matrix(y_true, y_pred) # Confusion Matrix

target_names = ['Low','High'] # Class names

c_report = classification_report(y_true, y_pred, target_names=target_names) #Classification report

acc_score = balanced_accuracy_score(y_true, y_pred) #Balanced accuracy Score calculation

f1_scr = f1_score(y_true, y_pred)

print('Valence accuracy:')
print(acc_score)

print(' ')
print('Valence F1 Score')
print(f1_scr)

print(' ')

print('Valence Confiusion matric')
print(cm)

print(' ')

# print('Accuracy score', acc_score)

print('Valence Classification Report')
print(c_report)

from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

class_names = target_names

## Plot Confusion matric Valence 
## ================================
fig1, ax1 = plot_confusion_matrix(conf_mat=cm, show_absolute=True,
                                show_normed=True,
                                colorbar=True,
                                  class_names=class_names)
plt.figure(1)
# plt.show()

fname = 'LR-SGD valence.jpeg'

plt.savefig(fname, bbox_inches='tight')

#============================================
# Arousal Classification Report
#============================================
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


y_true = aro_actual_class_label #Acutal class labels

y_pred = aro_predicted_class_labels #Predicted Class labels

cm = confusion_matrix(y_true, y_pred) # Confusion Matrix

target_names = ['Low','High'] # Class names

c_report = classification_report(y_true, y_pred, target_names=target_names) #Classification report


acc_score = balanced_accuracy_score(y_true, y_pred) #Balanced accuracy Score calculation

f1_scr = f1_score(y_true, y_pred)

print('Arousal accuracy:')
print(acc_score)

print(' ')
print('Arousal F1 Score')
print(f1_scr)

print(' ')

print('Arousal Confiusion matric')
print(cm)

print(' ')

# print('Accuracy score', acc_score)

print('Arousal classification Report')
print(c_report)

from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

class_names = target_names

## Plot Confusion matric Valence 
## ================================
fig1, ax1 = plot_confusion_matrix(conf_mat=cm, show_absolute=True,
                                show_normed=True,
                                colorbar=True,
                                  class_names=class_names)
plt.figure(1)
# plt.show()

fname = 'LR-SGD arousal.jpeg'

plt.savefig(fname, bbox_inches='tight')
