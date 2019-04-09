from PredictiveAnalytics.LifeModel.IntensityTemporalSequence import WriteSample
from PredictiveAnalytics.TestCases.Tim_Activity.Preprocessing.ReadData import F, V, Z
from PredictiveAnalytics.LifeModel import LifeModel

fixedSizeTesting=True #Fixed size array for life model for testing and comparision purposes. Default is false
#If true->ITS with Fixed Window Times
#If false->LM with exponential values

#Create a dataset
D=[]
for slice in range(1,len(Z)):
    lmHistory=LifeModel(F, V,n=5)
    lmFuture=LifeModel(F, V,n=5,future=True)    
    D.append((lmHistory.GetMSS(Z[:slice],fixedSizeTesting),lmFuture.GetMSS(Z[slice:],fixedSizeTesting)))
    print(slice,len(D[-1][0]))
    
print("Dataset length: {0}".format(len(D)))
print(D)

n=5

#Writing the LM dataset to file ready for ReadLM.py and then SequenceClassification.py
#Writing two separate files for input and output
import datetime, time
today=datetime.datetime.today()
timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
with open("db-Input-LM-f{f}v{v}n{n}rand{r}Fixed{x}-{t}.csv".format(v=V,f=F,n=n,r=len(D),x=fixedSizeTesting,t=timestr),'w') as file:
    for d in D:        
        WriteSample(file,d[0],F,V)

with open("db-Output-LM-f{f}v{v}n{n}rand{r}Fixed{x}-{t}.csv".format(v=V,f=F,n=n,r=len(D),x=fixedSizeTesting,t=timestr),'w') as file:
    for d in D:        
        WriteSample(file,d[1],F,V)



