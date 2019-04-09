from PredictiveAnalytics.TemporalModelling import TemporalAbstraction
import numpy as np
class LifeModel:
    """Creates a LifeModel mapping of an MSS Z."""
    def __init__(self, F, V, n=None, future=False, binaryFeatures=False):        
        """Number of variables (F), and the number of value abstractions (V), predefined n (LM produces 2^n elements), future? (Mapping starts from 0 instead of ending in 0), whether number of value abstractions represent a single value (V=2=> Off, On will be changed to Off/On with V=1"""
        self.n=n
        self.F=F
        self.V=V
        self.history=not future
        self.binaryFeatures=binaryFeatures

    def GetMSS (self, Z, fixedSize=False):
        """Get the Life Model representation (mapping) of given MSS. 
        FixedSize:
        Fixed size determines whether this method should use Life Model mapping (exponential) or fixed size ITS 
        (whole period is divided into fixed-size windows).
        Used for comparision between LM and regular window sizes. Fixed size window sizes similar to LM (both 32 for example)        
        """
        self.Z=Z

        if self.history:
            sequenceLength=max(E.e for E in Z)#Ending point -> Should become zero
            Z_norm=list(map(lambda x:TemporalAbstraction(x.f,x.v,x.s-sequenceLength,x.e-sequenceLength),Z))
            DeltaT=-min(E.s for E in Z_norm)#If not sorted, the length of the MSS
        else: #Future
            sequenceLength=min(E.s for E in Z) # Starting point - > Should become zero
            Z_norm=list(map(lambda x:TemporalAbstraction(x.f,x.v,x.s-sequenceLength,x.e-sequenceLength),Z))
            DeltaT=max(E.e for E in Z_norm)#If not sorted, the length of the MSS

        #print ("Z:")
        #print (Z)
        #print ("Z norm:")
        #print (Z_norm)


        #Finding k so 
        if(self.n is None):
            n=0
            k=1
            delta=1#delta
            while DeltaT>=2**(delta*k):
                n+=1
                k=2**n
        else:
            n=self.n
            k=2**self.n


        #print ("n= {0}, k= {1}".format(n,k))

        #Create a flag for comparison with regular windowing
        
        
        if self.history==True:
            p=[(-2**(k-i)+1,-2**(k-(i+1))+1) for i in range(0,k)]
        else:#future
            p=[(2**i-1,2**(i+1)-1) for i in range(0,k)]

        #Used for comparision between LM and regular window sizes
        #Fixed size window sizes similar to LM (both 32 for example)
        if fixedSize:
            elementdeltaT=DeltaT/k;# K equal window size
            if self.history==True:
                p=[(-elementdeltaT*(k-i)+1,-elementdeltaT*(k-(i+1))+1) for i in range(0,k)]
            else:#future
                p=[(elementdeltaT*i-1,elementdeltaT**(i+1)-1) for i in range(0,k)]
            
            

        #print (p)


        Matrix = [[[0 for v in range(self.V)] for u in range(self.F)] for m in range(0,k)]

        #find list
        for i in range(0,k):    
            In=list(filter(lambda x: ( p[i][0]<=x.s<p[i][1] or p[i][0]<=x.e<p[i][1]), Z_norm))#create in list for all temporal states that either start or end inside current period p_i using "interval comparisson"
            Out=list(filter(lambda x: ( x.s<p[i][0] and x.e>=p[i][1]), Z_norm))#create in list for all temporal states that are outside of the current period p_i using "interval comparisson"
            pLength=p[i][1]-p[i][0]
            #print("In")
            #print(p[i])
            #print(In)
            #print("Out")
            #print(p[i])
            #print(Out)
            #print("##################################")
            for Ej in In:
                if p[i][0]<=Ej.s<p[i][1] and p[i][0]<=Ej.e<p[i][1]:
                    #print("Condition In 1: i={0}, Ej.f={1}, Ej.v={2}".format(i,Ej.f,Ej.v))
                    Matrix[i][Ej.f][Ej.v]+=Ej.Length
                    Z_norm.remove(Ej)
                elif p[i][0]<=Ej.s<p[i][1]:#Ej only starts in p_i
                    #print("Condition In 2: i={0}, Ej.f={1}, Ej.v={2}", i,Ej.f,Ej.v)
                    Matrix[i][Ej.f][Ej.v]+=p[i][1]-Ej.s            
                elif p[i][0]<=Ej.e<p[i][1]:#Ej only ends in p_i
                    #print("Condition In 2: i={0}, Ej.f={1}, Ej.v={2}", i,Ej.f,Ej.v)
                    Matrix[i][Ej.f][Ej.v]+=Ej.e-p[i][0]            
                    Z_norm.remove(Ej)#Will not be useful for further periods
    
            for Ej in Out:
                #print("Condition Out: i={0}, Ej.f={1}, Ej.v={2}", i,Ej.f,Ej.v)
                Matrix[i][Ej.f][Ej.v]+=p[i][1]-p[i][0]#Add pi.length 
            #print("S{0}=".format(i))    
            #print(Matrix[i]/pLength)
            #Matrix_norm=list(map(lambda x: [matr][] ,Matrix[i]))
            mat=np.matrix(Matrix[i])
            mat=mat/pLength
            Matrix[i]=mat.tolist()
            #print(Matrix[i])

        if(self.binaryFeatures):#This removes the first column (not(binaryFeature)) which is redundant (off, on) [0.0, 1.0] =>[1.0] (on)
            #Remove the first column of ITS elements in the matrix                                                           
            Matrix=[[[x[1]] for x in y] for y in Matrix]
            


        return Matrix

def TestLifeModel(F, V, n=5, future=False):
    """-----------WORKS PERFECT FOR TESTING LM-------------"""
    lm=LifeModel(F, V,n=5,future=True)
    ITS=lm.GetMSS(Z)

    for i,S in enumerate(ITS):
        print("S[{0}] is {1}".format(i,S))
#-----------WORKS PERFECT FOR TESTING LM-------------END




