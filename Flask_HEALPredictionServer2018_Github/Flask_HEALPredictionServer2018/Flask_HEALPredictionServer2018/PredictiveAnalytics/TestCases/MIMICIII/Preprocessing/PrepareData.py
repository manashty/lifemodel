from PredictiveAnalytics.LifeModel.IntensityTemporalSequence import WriteSample
from PredictiveAnalytics.TestCases.MIMICIII.Preprocessing.ReadData import F, Z
from PredictiveAnalytics.LifeModel import LifeModel
import datetime, time
today=datetime.datetime.today() 
timestr = time.strftime("%Y-%m-%d_%H-%M-%S")

fixedSizeTesting=False#Fixed size array for life model for testing and comparision purposes. Default is false
#If true->ITS with Fixed Window Times
#If false->LM with exponential values
V=2#0 or 1
binaryFeaturesEnabled=True
#Create a dataset of Life Model mappings of Z
writeToFileOnly=True#Do not store in memory and write later
n=5
if(writeToFileOnly):
    file=open("db-Input-LM-f{f}v{v}n{n}rand{r}Fixed{x}-{t}.csv".format(v=V,f=F,n=n,r="NA",x=fixedSizeTesting,t=timestr),'w')
D=[]
for slice in range(len(Z)):
    lmHistory=LifeModel(F, V,n=5, binaryFeatures=binaryFeaturesEnabled)
    #lmFuture=LifeModel(F, V,n=5,future=True)    
    D.append((lmHistory.GetMSS(Z[slice][0],fixedSizeTesting),Z[slice][1]))#lmFuture.GetMSS(Z[slice:],fixedSizeTesting)))
    if(writeToFileOnly):
        if( not binaryFeaturesEnabled):
                WriteSample(file,D[-1][0],F,V)
        else:
                WriteSample(file,D[-1][0],F,1)
        print(slice,len(D[-1][0]))
        D[-1]=(None, D[-1][1])#Clear the memory of the actual data
    else:
        print(slice,len(D[-1][0]))
    
print("Dataset length: {0}".format(len(D)))
#print(D)

n=5



#Writing the LM dataset to file ready for ReadLM.py and then SequenceClassification.py
#Writing two separate files for input and output
if(not writeToFileOnly):
    with open("db-Input-LM-f{f}v{v}n{n}rand{r}Fixed{x}-{t}.csv".format(v=V,f=F,n=n,r=len(D),x=fixedSizeTesting,t=timestr),'w') as file:
        for d in D:        
            if( not binaryFeaturesEnabled):
                WriteSample(file,d[0],F,V)
            else: 
                WriteSample(file,d[0],F,1)

with open("db-Output-LM-f{f}v{v}n{n}rand{r}Fixed{x}-{t}.csv".format(v=V,f=F,n=n,r=len(D),x=fixedSizeTesting,t=timestr),'w') as file:
    for d in D:        
        #WriteSample(file,d[1],F,V)
        a=file.write(str(d[1])+"\n")



