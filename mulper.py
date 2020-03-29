#The main alg

import numpy as np
import math
import random
import matplotlib 
import matplotlib.pyplot as plt
import glob
import tkinter as tk
from tkinter import filedialog
from tkinter import *
import glob
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
dt=""
def inter():
    interface=tk.Tk()
    interface.title('多層感知機')
    interface.geometry('1100x1000')
    
    def selectfile():
            global dt
            file=tk.filedialog.askopenfilename()
            file=open(file,'r')
            data=file.read()
           
            dt=data.split('\n')
            dt=[i.split(' ')for i in dt]
            dt=np.array(dt)
            return dt
        
    LRlabel=tk.Label(interface,text="Learnig Rate",bg='red',font=('Arial',12),width=10,height=1)
    LRlabel.grid(row=1,sticky=W)
    LRentry=tk.Entry(interface)
    LRentry.grid(row=2,sticky=W)
    #######################################Learning rate
    EPlabel=tk.Label(interface,text="Epoch",bg='red',font=('Arial',12),width=10,height=1)
    EPlabel.grid(row=3,sticky=W)
    Epochentry=tk.Entry(interface)
    Epochentry.grid(row=4,sticky=W)
    ALLentry=tk.Entry(interface)
    ALLentry.grid(row=6,sticky=W)
    alllabel=tk.Label(interface,text="train all or not",bg='red',font=('Arial',12),width=10,height=1)
    alllabel.grid(row=5,sticky=W)
    FileButton=tk.Button(interface,text="fileselect",command=selectfile)
    FileButton.grid(row=0,sticky=W)
    
    def normalize(d):
        num=[]
        for i in range(len(d)):
            if(int(d[i])==1):
               num.append(1.0)
            else:
               num.append(0.0)
        return num,2

    def _initial_():
        global dt
        
        col=len(dt[0])
        data=[]
        
        for i in range (len(dt)):
            dj=[]
            if(len(dt[i])>2):
                for j in range( col):
                
                    dj.append(float(dt[i][j]))#turn str to float
                data.append(dj)
        row=len(data)
        d=[]

        data=np.array(data)
        data = np.hsplit(data, [col - 1])
        ipt=data[0]
        d=data[1]
        outputy=[]
        for j in range(row):
                outputy.append(float(d[j]))           

        wi1=[]
        wi2=[]
        for i in range (col-1):
            num=np.random.randn(1,1)
            wi1.append(num)
        for i in range (col-1):
            num=np.random.randn(1,1)
            wi2.append(num)
    
        #########################################classify
        clas=[]
        clas.append(dt[0][col-1])
        return row,col,wi1,wi2,ipt,outputy
    def sigmoid(x):
        r=1.000/(1+math.pow(math.e,-x))
       
        return r
    def press():
        epoch=1#initial
        LR=str(LRentry.get())
        LR=float(LR)
        epoch=int(Epochentry.get())
        ALL=int(ALLentry.get())
        train(epoch,LR,ALL)
    def train(epoch,LR,ALL):
        
        row,col,wi1,wi2,inputx,d=_initial_()
        w1out=np.random.rand(1,1)
        w2out=np.random.rand(1,1)
        Biasp=-1
        Biasi1=np.random.randn(1,1)
        Biasi1=float(Biasi1)
        Biasi2=np.random.randn(1,1)
        Biasi2=float(Biasi2)
        Biask=np.random.randn(1,1)
        Biask=float(Biask)
        wi1=list(wi1)
        wi2=list(wi2)
        
        d,interval=normalize(d)
#       
        if (ALL==0):
            testindex=[]
            trainindex=np.random.choice(row,size=int(row*2/3)+1,replace=False)
            for i in range(row):
                testindex.append(i)
            testindex=set(testindex)-set(trainindex)
            
            testindex=list(testindex)
#            print(testindex,"testind")
        else:
            trainindex=[]
            for i in range(row):
                trainindex.append(i)
            testindex=np.random.choice(row,size=int(row/3),replace=False)
        trainre=[]
        #random choice 1/3
        j=0
        while(j<epoch):
            
            MSE=0.0
            trainacr=0.0
            a=0
            while(a<len(trainindex)):
                inpnode=inputx[int(trainindex[a])]
                su=0.0
                for i in range(len(wi1)):
                    su+=float((inpnode[i])*(wi1[i]))
                su+=Biasi1*Biasp
                su=float(su)
                hnode1=sigmoid(su)
                su=0.0
                for i in range(len(wi2)):
                    su+=float((inpnode[i])*(wi2[i]))
                su+=Biasi2*Biasp
                hnode2=sigmoid(su)
                su=0.0
                su=(hnode1)*(w1out)+(hnode2)*(w2out)+Biask*Biasp
                su=float(su)
                outnode=sigmoid(su)
                if(j==(epoch-1)):
                    gp=[inpnode,outnode]
                    trainre.append(gp)
                    if(outnode<=0.50000 and d[trainindex[a]]==0):
                        trainacr+=1
                    elif(outnode>0.50000 and d[trainindex[a]]==1):
                        trainacr+=1
                #back propagation
                err=d[trainindex[a]]-outnode
                MSE+=err*err
                deltak=0.0
                deltak=(d[trainindex[a]]-outnode)*outnode*(1-outnode)
                w1out=float(w1out+LR*float(deltak)*hnode1)
                w2out=float(w2out+LR*float(deltak)*hnode2)
                Biask+=LR*deltak*(Biasp)
                delta1=0.0
                delta1=hnode1*(1-hnode1)*deltak*w1out               
                for i in range(len(wi1)):
                    wi1[i]=float(float(wi1[i])+float(LR*delta1)*float(inpnode[i]))
                Biasi1+=float(float(delta1)*float(LR)*float(Biasp))
                delta2=0.0
                delta2=hnode2*(1-hnode2)*deltak*w2out
                for i in range(len(wi2)):
                    wi2[i]=float(float(wi2[i])+float(LR)*float(delta2)*float(inpnode[i]))
                Biasi2+=float(float(delta2)*float(LR)*float(Biasp))
                a=a+1
               
            trainacr=trainacr*100/len(trainindex)
            
#           
            j+=1
        
        ##########################################test
        
        testre=[]
        testacr=0
        if(len(testindex)>=1):
            ind=0
            while(ind<len(testindex)):
                inputnode=inputx[testindex[ind]]
                su=0.0
                for k in range(len(wi1)):
                       su+=float((inputnode[k])*(wi1[k]))
                su+=Biasi1*Biasp
                su=float(su)
                        
                hnode1=sigmoid(su)
                       # print(hnode1,"hnode1")
                su=0.0
                for k in range(len(wi2)):
                    su+=float((inputnode[k])*(wi2[k]))
                su+=Biasi2*Biasp
                        
                hnode2=sigmoid(su)
                        #print(hnode2,"hnode2")
                su=0.0
                su=(hnode1)*(w1out)+(hnode2)*(w2out)+Biask*Biasp
                su=float(su)
                outnode=sigmoid(su)
                if(ALL==0):
                    MSE+=(d[testindex[ind]]-outnode)*(d[testindex[ind]]-outnode)
                gp=[inputnode,outnode]
                testre.append(gp)
                   
                if(outnode<=0.50000 and d[testindex[ind]]<0.25):
                    testacr+=1
                elif(outnode>0.50000 and d[testindex[ind]]>0.8):
                    testacr+=1
                ind+=1
        testacr=float(testacr)*100.0/float(len(testindex))
        MSE=MSE/row
        MSE=math.pow(MSE,0.5)
        
        def draw():
            
             f =Figure(figsize=(5,5), dpi=100)
             a=f.add_subplot(111)
               
             canvas =FigureCanvasTkAgg(f, master=interface)
             
             for i in range(len(trainre)):
                 if(trainre[i][1]>0.50000):
                     a.plot(trainre[i][0][0],trainre[i][0][1],'go')
                 else:
                     a.plot(trainre[i][0][0],trainre[i][0][1],'ro')
             for i in range(len(testre)):
#                
                 if(testre[i][1]>0.50000):
                    a.plot(testre[i][0][0],testre[i][0][1],'bx')
                 else:
                    a.plot(testre[i][0][0],testre[i][0][1],'mx')
             
             
             MSEL="RMSE :"+str(MSE)
             MSEBL=tk.Label(interface,text=MSEL,bg='white',font=('Arial',12),width=50,height=2)
             MSEBL.grid(row=3,column=2)
             TRACRL="Training accuracy :"+str(trainacr)+"%"
             TRACRLBL=tk.Label(interface,text=TRACRL,bg='white',font=('Arial',12),width=50,height=2)
             TRACRLBL.grid(row=1,column=2)
             TsACRL="Testing accuracy :"+str(testacr)+"%"
             TsACRLBL=tk.Label(interface,text=TsACRL,bg='white',font=('Arial',12),width=50,height=2)
             TsACRLBL.grid(row=2,column=2)
            

             canvas.get_tk_widget().grid(row=8)
             canvas._tkcanvas.grid(row=8)
             print("draw success")
        draw()
        
        

    trainButton=tk.Button(interface,text="train",command=press)
    trainButton.grid(row=7,sticky=W)
    interface.mainloop()      
inter()

