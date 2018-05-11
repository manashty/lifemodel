# coding: utf-8

script_version='2.6'
date='April 8th, 2018, UNB'
#in 15th folder, we can find the code before refactoring


import gzip
import queue

import pandas
import numpy

#def read_values():
#    return sth

method1_1="LifeModel_Binary"
method1_2="Fixed_Binary"
method1_3="Reg_Binary"

method2_1="LifeModel_Multiple"
method2_2="Fixed_Multiple"
method2_3="Reg_Multiple"

method3_1="LifeModel_Accel" #size(70, 129) Binary
method3_2="Fixed_Accel"     #size(70, 129) Binary

method4_1="reg_LifeModel32_Accel_AutoShift" #(, 130)
method4_2="reg_Fixed32_Accel_AutoShift"     #(, 130)
method4_3="LifeModelForecast_SingleRegression4_mortality"
method4_4="RegularForecast_SingleRegression4_mortality"

method5_1 = "LifeModel_Seq2Seq"
method5_2 = "Regular_Seq2Seq"

#method selection
method = method5_1


seqToSeq = True

input_file_variable_name = "Input"
#Possible values:
#1  "_50Miss" #Fall Regression
#2  "_10Miss" #Fall Regression
#3  "" #Fall Regression no missing values
#4  "Input" # Mortality
 
mortality = True
#Possible values:
#1  True  --> mortality dataset
#2  False --> fall detection

detection_mode = 1
#Possible values:
#1  1  --> one step (all people including who has not fallen are considered)
#2  2  --> two step (only people who has fallen are considered)

auto_shifted = False
#Possible values:
#1  True  --> autoshifted fallDetection dataset
#2  False --> not autoshifted fallDetection dataset

binaryP = "Binary classification problem"
multiClassP = "MultiClass classification problem"
regressionP = "Regression problem"

problem_type = regressionP

#if it is future prediction or binary  classification
omit=True
#Possible values:
#1  True  --> prediction
#2  False --> binary

if(method==method3_1 or
   method==method3_2 or
   method==method4_1 or
   method==method4_2):
    mortality=False

#for TE lost function
tolerance = 4096
#tolerance = 83000
toleranceIndex = 23

#value we assign to people who has not fallen
not_fallen = -1

#input and output declaration   
X=[]
Y=[]

if(method==method3_1):
    seqToSeq = False
    input_file_variable_name = ""
    mortality = False
    detection_mode = 1
    auto_shifted = False
    problem_type = binaryP
    omit=False
elif(method==method3_2):
    seqToSeq = False
    input_file_variable_name = ""
    mortality = False
    detection_mode = 1
    auto_shifted = False
    problem_type = binaryP
    omit=False
elif(method==method4_1):
    seqToSeq = False
    input_file_variable_name = "_50Miss" # "" or "_10Miss" or "_50Miss"
    mortality = False
    detection_mode = 2 # 1 or 2 
    auto_shifted = True
    problem_type = regressionP
    omit=True
elif(method==method4_2):
    seqToSeq = False
    input_file_variable_name = "_50Miss" # "" or "_10Miss" or "_50Miss"
    mortality = False
    detection_mode = 2 # 1 or 2 
    auto_shifted = True
    problem_type = regressionP
    omit=True
elif(method==method4_3):
    seqToSeq = False
    input_file_variable_name = "Input" 
    mortality = True
    detection_mode = 2
    auto_shifted = False
    problem_type = regressionP
    omit=True
elif(method==method4_4):
    seqToSeq = False
    input_file_variable_name = "Input" 
    mortality = True
    detection_mode = 2
    auto_shifted = False
    problem_type = regressionP
    omit=True
elif(method==method5_1):
    seqToSeq = True
    input_file_variable_name = "Input" 
    mortality = True
    detection_mode = 1
    auto_shifted = False
    problem_type = regressionP
    omit=True
elif(method==method5_2):
    seqToSeq = True
    input_file_variable_name = "Input" 
    mortality = True
    detection_mode = 1
    auto_shifted = False
    problem_type = regressionP
    omit=True

if(not omit):
    limit = max = 5200
else:
    limit = max = 34100

if(method==method4_3 or method==method4_4):
    limit = max = 5200
#network parameters:
#values for LifeModel of mortaliy dataset
F=8371 #dimentionality of each timestep in each sequence for each sample
K=32   #sequence length

if(not mortality):
    #values for Accel data (fall detection)
    if(auto_shifted):
        limit = max = 500
    else:
        limit = max = 70
    F = 4
    K = 32

#???
current_total=0
y_current_total=0

#C:\Users\manashty\Documents\LifeModelForecasting\LifeModelForecasting
curDir=""

if(method==method1_1):#LifeModel
    filename=curDir+'MIMICIII_Diag_Proc_LM_{0}12-9-2017 7-10-28 PM.csv'
elif (method==method1_2):##Fixed
    filename=curDir+'MIMICIII_Diag_Proc_Fixed_{0}12-10-2017 8-41-34 PM.csv'
elif (method==method1_3):##Reg
    filename=curDir+'All Patients Reg Fixed\MIMICIII_Diag_Proc_Reg_{0}1-9-2018 3-51-30 PM.csv'

##Omit Life Model multiple output
#??? by "Multiple output"you mean sequene?
if(method==method2_1):#LifeModel
    filename=curDir+"MIMICIII_Diag_Proc_LM_Omit_{0}1-10-2018 2-20-50 PM.csv"
elif (method==method2_2):##Fixed
    filename=curDir+""
elif (method==method2_3):##Reg
    filename=curDir+'\All Patients Reg Fixed\MIMICIII_Diag_Proc_Reg_{0}1-9-2018 3-51-30 PM.csv'

if(method==method3_1):#LifeModel
    filename=curDir+"FallDataAvgLifeModel32_{}2018-03-10  15-17-39.csv"
elif (method==method3_2):#Fixed
    filename=curDir+"FallDataAvgFixed32_{}2018-03-10  15-17-39.csv"

if(method==method4_1):
    filename=curDir+'FallDataAvgLifeModel32AutoShift{0}_2018-03-10  15-17-39.csv'
elif(method==method4_2):
    filename=curDir+'FallDataAvgFixed32AutoShift{0}_2018-03-10  15-17-39.csv'
elif(method==method4_3):
    filename=curDir+'MIMICIII_Diag_Proc_LM_Forecast_SingleRegression4_{}3-13-2018 7-19-42 PM.csv'
elif(method==method4_4):
    filename=curDir+'MIMICIII_Diag_Proc_Reg_Forecast_SingleRegression4_{}3-28-2018 5-10-01 AM.csv'

if(method==method5_1):
    filename=curDir+'MIMICIII_Diag_Proc_LM_Forecast_Seq2SeqDiagnosisLM4_{}3-28-2018 4-01-38 AM.csv'
elif(method==method5_2):
    filename=curDir+'MIMICIII_Diag_Proc_Reg_Forecast_Seq2SeqDiagnosisReg4_{}3-28-2018 4-01-38 AM.csv'

        
print("Opened input file and ready for reading from {0}".format(filename.format(input_file_variable_name)))

  
if(not mortality):
    file_in=open(curDir+filename.format(input_file_variable_name))
else:#Normal Mortality
    file_in=gzip.open(curDir+filename.format(input_file_variable_name)+'.gz')
numberOfSamples = sum(1 for line in file_in)

if(method==method4_3 or method==method4_4):
    numberOfSamples = limit

print("Read input samples: {0} samples from {1}".format(numberOfSamples,filename.format(input_file_variable_name)))



if(seqToSeq):
    file_out=gzip.open(curDir+filename.format("Output")+'.gz')
    
# Function to read output: Y
def file_read_output(fname, future=None, accel=False):
    global numberOfSamples
    if(seqToSeq):
        fname = fname + '.gz'
    if(not accel):        
        with open(fname, encoding="utf8") as f: #    encoding="utf8" added for seq2seq
        #with open(fname) as f:     
                counter=0
                if future:
                    for line in f:
                        if(not omit):
                            #Y.append(int(line.split(',')))#Check if this is not correct
                            Y.append(int(line))
                        else:
                            '''
                            lines=line.split(',') 
                            
                            lines=list(map(lambda x: int(x),lines))
                            #Y.append(lines)
                            Y.append(sum(lines)/5)#Make it a 5 category thing, regression
                            '''
                            #mehrdad                            
                            #Y.append((int(line) if int(line)>0 else int(line)*12)+5)#Change -1 to -10 (*10)
                            Y.append((int(line)))# if int(line)>0 else int(line)*12)+5)#Change -1 to -10 (*10)
                            #-7,,,,,,,,,,5,6,7,8,9
                        counter+=1
                        if(counter==limit):
                            break
                        #lines=line.split(',')                                                
                        #lines=list(map(lambda x: float(x),lines))
                        #Y.append([lines[i*F:(i+1)*F] for i in range(K)])
                        
                    f.close()#Y.append(data[:])
                else: #History                    
                    for line in f:  
                        lines=line.split(',')                                                
                        lines=list(map(lambda x: float(x),lines))
                        X.append([lines[i*F:(i+1)*F] for i in range(K)])
                        counter+=1
                        print("Reading patient "+str(counter))
                        if(counter==limit):
                            break
                        #print("X Shape "+str(len(X[0][0])))
                    #X.append(data[:])
                    #print(len(X))
                        #Y.append([int(j == int(lines[0])) for j in range(diagnosis_classes)])
                        #Y.append(lines[0])
                        #X.append([lines[1+i*25:1+(i+1)*25] for i in range(25)])
        
    else:
        if(auto_shifted):
            df = numpy.array(pandas.read_csv(fname, usecols=[0,1], header=None))
            #numberOfSamples = df.shape[0]
            for index, value in enumerate(df[:,1].tolist()):
                if(omit):
                    if(detection_mode==2):
                        if(df[index,0]==0.0):
                            Y.append(value)
                    else:
                        if(df[index,0]==0.0):
                            Y.append(value)# :(
                        else:
                            Y.append(not_fallen)
                else:
                    Y.append(value)
            numberOfSamples = len(numpy.array(Y))
        else:
            df = numpy.array(pandas.read_csv(fname, usecols=[0], header=None))
            numberOfSamples = df.shape[0]
            for index, value in enumerate(df[:,0].tolist()):
                Y.append(value)
            
if(not seqToSeq):
    if (mortality):
        file_read_output(filename.format("Output"),True, False) #Replace Type with output for reading future
        print("Read output samples: {0} samples from {1}".format(len(Y),filename.format("Output")))
    else:
        if(omit):
            file_read_output(filename.format(input_file_variable_name),True, True) 
            print("Read output samples: {0} samples from {1}".format(len(Y),filename.format(input_file_variable_name)))
        else:
            file_read_output(filename.format(input_file_variable_name),False, True) 
            print("Read output samples: {0} samples from {1}".format(len(Y),filename.format(input_file_variable_name)))
else:
    Y = []

'''
def batch_read(size=1):    
    global current_total, X, file_in
    X=[]
    counter=0    
    for c in range(size):
        if(current_total<max):
            line=next(file_in)
            if(not mortality):
                linesB=line                
            else:#Normal
                linesB=line.decode('UTF-8')            
            lines=linesB.split(',')                
            lines=list(map(lambda x: float(x),lines[:-1]))
            #print("Len of lines: {0}".format(len(lines)))
            X.append([lines[i*F:(i +1)*F] for i in range(K)])
            #print("Len of X: {0}".format(len(X)))
            counter+=1
            print("Reading patient "+str(counter),end="\r",flush=True)
            if(counter==limit):
                break
            current_total+=1
        else:
            print('Can not read more than {0} samples.'.format(current_total))
            return X
            break
    print("**Finished Reading {0} Samples**".format(size))
    return X
'''


def batch_read_thread(q, size=2):  

    global current_total, X, file_in
    X=[]
    counter=0
    c = 0
    while c < size:
        c = c + 1
        if(current_total<max):
            line=next(file_in)
            #if (c<3):                
             #   print('Line {0} of current batch file offset {1}'.format(str(c), str(file_in.tell())))


            if(not mortality):
                linesB=line                
            else:#Normal #mehrdad
                linesB=line.decode('UTF-8')  
                   
            lines=linesB.split(',')
            if(mortality):
                #mehrdad
                if(not seqToSeq and method==method4_3):
                    lines = lines[:-1] #there is a non-integer character in last column with value '/r/n'
                    #if we get error of size and dimension in LSTM, it's because of the line above!
            # 267872 / 32 = 8371 
            lines=list(map(lambda x: float(x),lines))
            
            if(mortality):
                X.append([lines[i*F:(i +1)*F] for i in range(K)])
                #print(numpy.array(X).shape)
            else:
                if(not omit):
                    X.append([lines[i*F:(i +1)*F] for i in range(K)])
                else:
                    if(detection_mode==2):
                        print(lines[0])
                        if(lines[0]==0):
                            X.append([lines[i*F+2:(i +1)*F+2] for i in range(K)])
                        else:
                            c = c-1
                    else:
                        X.append([lines[i*F+2:(i +1)*F+2] for i in range(K)])

            print("  *******I/O("+str(counter+1)+'/'+str(size)+'->'+str(current_total+1),end=")***\r",flush=True)
            counter+=1
            if(counter==limit):
                break
            current_total+=1
        else:
            print('Can not read more than {0} samples.'.format(current_total))
            X2=X
            q.put(X2)
            return X2
            break
    print("**Finished Reading {0} from {1} to {2} Samples***".format(size,current_total-size,current_total))
    X2=X
    q.put(X2)
    return X2


# In[8]:


def reset():
    global current_total, X, file_in
    file_in.close()
    if(not mortality):
        file_in=open(curDir+filename.format(input_file_variable_name))
    else:
        file_in=gzip.open(curDir+filename.format("Input")+'.gz')
    current_total=0
    y_current_total=0
    X=[]
    counter=0

def Y_batch_read_thread(q2, size=2):  

    global y_current_total, Y, file_out
    Y=[]
    counter=0
    c = 0
    while c < size:
        c = c + 1
        if(y_current_total<max):
            line=next(file_out)

            linesB=line.decode('UTF-8')  
                   
            lines=linesB.split(',')
            if(not seqToSeq):
                lines = lines[:-1] #there is a non-integer character in last column with value '/r/n'
            # 267872 / 32 = 8371 
            lines=list(map(lambda x: float(x),lines))
            
            Y.append([lines[i*F:(i +1)*F] for i in range(K)])

            print("  *******I/O("+str(counter+1)+'/'+str(size)+'->'+str(y_current_total+1),end=")***\r",flush=True)
            counter+=1
            if(counter==limit):
                break
            y_current_total+=1
        else:
            print('Can not read more than {0} samples.'.format(y_current_total))
            Y2=Y
            q2.put(Y2)
            return Y2
            break
    print("**Finished Reading {0} from {1} to {2} Samples (Y)***".format(size,y_current_total-size,y_current_total))
    Y2=Y
    q2.put(Y2)
    return Y2
