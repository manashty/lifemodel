X=[]
Y=[]
limit=100
#data=[]
F=8372
K=32
#diagnosis_classes=10
def file_read(fname, future=None):          
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
                    #Y.append(data[:])
                else: #History                    
                    for line in f:  
                        lines=line.split(',')                                                
                        lines=list(map(lambda x: float(x),lines))
                        print("Len of lines: {0}".format(len(lines)))
                        X.append([lines[i*F:(i+1)*F] for i in range(K)])                        
                        print("Len of X: {0}".format(len(X)))
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
 
#def batch_read(size=100):

#5 Class
#filename='db-s10000aps50inj0.15-0.1w400v5randTruedrange10000.csv'
#diagnosis_classes=10
#10 Class
#filename='db-s10000aps50inj0.15-0.05w400v5randTruedrange10000.csv'
#10 Class 10000 Rand

#101 samples
#filename='db-{0}-LM-f8372v2n5rand101FixedFalse-2017-12-08_13-42-29.csv'

#1001 samples
filename='db-{0}-LM-f8372v2n5rand1001FixedFalse-2017-12-08_14-29-13.csv'


file_read(filename.format("Input")) #Replace Type with input for reading history
print("Read input samples: {0} samples from {1}".format(len(X),filename.format("Input")))

file_read(filename.format("Output"),True) #Replace Type with output for reading future
print("Read output samples: {0} samples from {1}".format(len(Y),filename.format("Output")))

#print(X)