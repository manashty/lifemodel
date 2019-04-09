script_version='2.0'
import gzip
import queue
method1_1="LifeModel_Binary"
method1_2="Fixed_Binary"

method=method1_1#Life Model Binary
#method=method1_2

X=[]
Y=[]
limit=34100
current_total=0
max=34100
#data=[]
F=8371
K=32
#diagnosis_classes=10
#curDir='C:\\Users\\manas\\OneDrive\\@Dev\\SP4\\PythonCNTKVS2017SP4\\PythonCNTKVS2017SP4\\'
curDir=""
if(method==method1_1):#LifeModel
    filename=curDir+'MIMICIII_Diag_Proc_LM_{0}12-9-2017 7-10-28 PM.csv'
elif (method==method1_2):##Fixed
    filename=curDir+'MIMICIII_Diag_Proc_Fixed_{0}12-10-2017 8-41-34 PM.csv'


#file_read(filename.format("Input")) #Replace Type with input for reading history
file_in=gzip.open(curDir+filename.format("Input"))
print("Opened input file and ready for reading from {0}".format(filename.format("Input")))
#print("Read input samples: {0} samples from {1}".format(len(X),filename.format("Input")))

def file_read_output(fname, future=None):     
        with open(fname) as f: 
                #data.clear()
                #Content_list is the list that contains the read lines.       
                #for line in f:  
                        #lines=line.split(',')                        
                        
                        #data.append([lines[i*F:(i+1)*F] for i in range(32)])
                counter=0
                if future:
                    for line in f:  
                        Y.append(int(line))
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
 
file_read_output(filename.format("Output"),True) #Replace Type with output for reading future
#file_out=open(filename.format("Output"))
print("Read output samples: {0} samples from {1}".format(len(Y),filename.format("Output")))


def batch_read(size=100):    
    '''
    Reads the next set of samples and puts them in the X.
    X is a list (size) of list (K) of numbers (F)
    '''
    global current_total, X, file_in
    X=[]
    counter=0    
    for c in range(size):
        if(current_total<max):
            line=next(file_in)
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


def batch_read_thread(q, size=100):    
    global current_total, X, file_in
    X=[]
    counter=0    
    for c in range(size):
        if(current_total<max):
            line=next(file_in)
            linesB=line.decode('UTF-8')            
            lines=linesB.split(',')                
            lines=list(map(lambda x: float(x),lines[:-1]))
            #print("Len of lines: {0}".format(len(lines)))
            X.append([lines[i*F:(i +1)*F] for i in range(K)])
            #print("Len of X: {0}".format(len(X)))            
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


#print(X)

def reset():
    global current_total, X, file_in
    file_in.close()
    file_in=gzip.open(curDir+filename.format("Input"))
    current_total=0
    X=[]
    counter=0
    


