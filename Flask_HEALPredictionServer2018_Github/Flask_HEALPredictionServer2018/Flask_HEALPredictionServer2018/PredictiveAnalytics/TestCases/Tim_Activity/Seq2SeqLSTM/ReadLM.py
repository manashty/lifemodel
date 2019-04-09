X=[]
Y=[]
#data=[]
F=17
K=32
#diagnosis_classes=10
def file_read(fname, future=None):          
        with open(fname) as f: 
                #data.clear()
                #Content_list is the list that contains the read lines.       
                #for line in f:  
                        #lines=line.split(',')                        
                        
                        #data.append([lines[i*F:(i+1)*F] for i in range(32)])
                if future:
                    for line in f:  
                        lines=line.split(',')                                                
                        lines=list(map(lambda x: float(x),lines))
                        Y.append([lines[i*F:(i+1)*F] for i in range(K)])
                    #Y.append(data[:])
                else: #History
                    for line in f:  
                        lines=line.split(',')                                                
                        lines=list(map(lambda x: float(x),lines))
                        X.append([lines[i*F:(i+1)*F] for i in range(K)])
                    #X.append(data[:])
                    #print(len(X))
                        #Y.append([int(j == int(lines[0])) for j in range(diagnosis_classes)])
                        #Y.append(lines[0])
                        #X.append([lines[1+i*25:1+(i+1)*25] for i in range(25)])
 
#5 Class
#filename='db-s10000aps50inj0.15-0.1w400v5randTruedrange10000.csv'
#diagnosis_classes=10
#10 Class
#filename='db-s10000aps50inj0.15-0.05w400v5randTruedrange10000.csv'
#10 Class 10000 Rand
filename='db-{0}-LM-f17v1n5rand258FixedFalse.csv'

file_read(filename.format("Input")) #Replace Type with input for reading history
print("Read input samples: {0} samples from {1}".format(len(X),filename.format("Input")))

file_read(filename.format("Output"),True) #Replace Type with output for reading future
print("Read output samples: {0} samples from {1}".format(len(Y),filename.format("Output")))

#print(X)