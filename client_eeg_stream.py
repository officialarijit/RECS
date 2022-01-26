import socket
import time
import pickle
import pandas as pd

HEADERSIZE = 10
port = 1245

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), port))
s.listen(5)

while True:
    # now our endpoint knows about the OTHER endpoint.
    clientsocket, address = s.accept()
    print(f"Connection from {address} has been established.")
    p = 32 #participants
    for i in range (0,p):
        i = i+1
        filename = '../05_JAN_2021_Final_EEG_Stream_socket_stream/data/'+str(i)+'_data_DEAP'+'.csv'
        v = 1
        for df in pd.read_csv(filename,sep=',', header = None, chunksize=32):
            # ptr = "Sending video:"+str(v)+" data for participant:"+str(i)
            # print(ptr)
            pvc = df.iloc[1,range(0,3)] #person+video+channel
            
            data = df.iloc[:,range(3,8067)] #Data signals
            
            emo_label = df.iloc[1,[8067,8068]] #emotion labels
            
            di ={"pvc":pvc, "data":data, "emotion":emo_label} # making a dictionary.

            msg = pickle.dumps(di) # maing a pickle dump
            msg = bytes(f"{len(msg):<{HEADERSIZE}}", 'utf-8')+msg
            # print(msg)
            clientsocket.send(msg) #Sending the message through socket
            ptr = "Data sent for video:"+str(v)+" and participant:"+str(i)
            print(ptr)

            v=v+1

            time.sleep(1) # wait for 1sec # For speeding up the process. But in real case it will be for 60 sec.  
    
