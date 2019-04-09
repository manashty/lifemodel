using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

/*
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




 
  
*/
namespace MIMICIIIReader.LifeModel
{
    public class Point
    {
        public TimeSpan TimeStamp { get; set; }

        public List<double> values;

        internal Point Normalize(TimeSpan max)
        {
            TimeStamp = TimeStamp.Subtract(max).Subtract(new TimeSpan(1));//A little before zero to be included in the Life Model
            return this;
            //double max = values.Max();
            //values = values.Select(x => x - max).ToList();
        }
    }


    public class LifeModel
    {

        int F, V, n;
        bool future = false;
        bool binaryFeatures = false;
        public LifeModel(int F, int V, int n = -1, bool future = false, bool binaryFeatures = false)
        {
            this.F = F;
            this.V = V;
            this.n = n;
            this.future = future;
            this.binaryFeatures = binaryFeatures;
        }
        public float[,,] GetITS(List<TemporalAbstraction> MSS, bool fixedSize = false)
        {
            List<TemporalAbstraction> Z_norm = new List<TemporalAbstraction>();
            Dictionary<int, TemporalAbstraction> Z_norm_Dic = new Dictionary<int, TemporalAbstraction>();
            float sequenceLength = 0;
            float DeltaT = 0;
            if (!future)//History
            {
                //sequenceLength = MSS.Max(x => x.e);
                sequenceLength = MSS.Max(x => x.e);//Look for intervals now! Faster!
                for (int i = 0; i < MSS.Count; i++)
                {
                    Z_norm_Dic.Add(i, new TemporalAbstraction(MSS[i].f, MSS[i].v, MSS[i].s - sequenceLength, MSS[i].e - sequenceLength));
                }
                DeltaT = -Z_norm_Dic.Min(x => x.Value.s);//Earliest start means the length (because end is now zero)                
            }
            else//Future
            {
                sequenceLength = MSS.Min(x => x.s);
                for (int i = 0; i < MSS.Count; i++)
                {
                    Z_norm_Dic.Add(i, new TemporalAbstraction(MSS[i].f, MSS[i].v, MSS[i].s - sequenceLength, MSS[i].e - sequenceLength));
                }
                DeltaT = Z_norm_Dic.Max(x => x.Value.e);//Earliest start means the length (because end is now zero)                
            }

            //Finding K and N
            int k = 0;

            if (n < 0)//it is not defined
            {
                n = 0;
                k = 1;
                int delta = 1;
                while (DeltaT >= Math.Pow(2, delta * k))
                {
                    n++;
                    k = (int)Math.Pow(2, n);
                }
            }
            else//n was provided
                k = (int)Math.Pow(2, n);




            //Create periods
            List<Tuple<int, int>> p = new List<Tuple<int, int>>();
            for (int i = 0; i < k; i++)
            {
                if (!future)//History                            
                    p.Add(new Tuple<int, int>(-(int)Math.Pow(2, k - i) + 1, -(int)Math.Pow(2, k - (i + 1)) + 1));
                else
                    p.Add(new Tuple<int, int>((int)Math.Pow(2, i) - 1, (int)Math.Pow(2, (i + 1)) - 1));
            }

            //Fixed Sized periods for testing and comparison between LifeModel mapping and a fixed Window mapping
            if (fixedSize)
            {
                int elementDeltaT = (int)Math.Ceiling(DeltaT / k);
                for (int i = 0; i < k; i++)
                {
                    if (!future)//History                            
                        p.Add(new Tuple<int, int>(-elementDeltaT * (k - i) + 1, -elementDeltaT * (k - (i + 1)) + 1));
                    else
                        p.Add(new Tuple<int, int>(elementDeltaT * (i) - 1, -elementDeltaT * ((i + 1)) - 1));
                }

            }

            // List<List<List<float>>> matrix = new List<List<List<float>>>();
            float[,,] matrix = new float[k, F, V];

            for (int i = 0; i < k; i++)
            {
                //Create IN and OUT list
                var IN = Z_norm_Dic.Where(x =>
                  ((x.Value.s >= p[i].Item1) && (x.Value.s < p[i].Item2) || ((x.Value.e >= p[i].Item1) && (x.Value.e < p[i].Item2)))).ToList();//.ToDictionary(x => x.Key, x => x.Value);//.ToDictionary (.ToDictionary<int,TemporalAbstraction>( ;//.Select(x=>x.Value).ToList();

                var OUT = Z_norm_Dic.Where(x =>
                   ((x.Value.s < p[i].Item1) && (x.Value.e >= p[i].Item2))).ToList();//.ToDictionary(x=>x.Key,x=>x.Value);//.Select(x => x.Value).ToList();

                float pLength = p[i].Item2 - p[i].Item1;



                foreach (var Ej in IN)
                {
                    if (((p[i].Item1 <= Ej.Value.s) && (Ej.Value.s < p[i].Item2)) &&
                        ((p[i].Item1 <= Ej.Value.e) && (Ej.Value.e < p[i].Item2)))//Start and End inside p[i]
                    {
                        matrix[i, Ej.Value.f, Ej.Value.v] += Ej.Value.Length;
                        Z_norm_Dic.Remove(Ej.Key);
                    }
                    else if ((p[i].Item1 <= Ej.Value.s) && (Ej.Value.s < p[i].Item2))//Only if starts in p[i]
                    {
                        matrix[i, Ej.Value.f, Ej.Value.v] += p[i].Item2 - Ej.Value.s;
                        //Do not remove, let it be there for next elements
                    }
                    else if ((p[i].Item1 <= Ej.Value.e) && (Ej.Value.e < p[i].Item2))//Only if ends in p[i]
                    {
                        matrix[i, Ej.Value.f, Ej.Value.v] += Ej.Value.e - p[i].Item1;
                        Z_norm_Dic.Remove(Ej.Key);
                    }
                }

                foreach (var Ej in OUT)
                {
                    matrix[i, Ej.Value.f, Ej.Value.v] += pLength;
                }

                //Normalize


                //Modify for binary feature


            }

            return matrix;
        }

        /// <summary>
        /// LifeModel mapping using intervals of temporal abstractions. Should be tons faster than regular.
        /// </summary>
        /// <param name="temporalAbstractionIntervalMSS"></param>
        /// <param name="fixedSize"></param>
        /// <returns></returns>
        public double[,,] GetITS(List<TemporalAbstractionInterval> MSS, bool fixedSize = false)
        {
            //List<TemporalAbstraction> Z_norm = new List<TemporalAbstraction>();
            Dictionary<int, TemporalAbstractionInterval> Z_norm_Dic = new Dictionary<int, TemporalAbstractionInterval>();
            float sequenceLength = 0;
            float DeltaT = 0;

            //Calculating sequence length for adjustment (reseting to zero)
            if (!future)
                sequenceLength = MSS.Max(x => x.interval.e);//Look for intervals now! Faster!
            else
                sequenceLength = MSS.Min(x => x.interval.s);//Not actually used for future

            for (int i = 0; i < MSS.Count; i++)
            {
                if (!future)
                    Z_norm_Dic.Add(i, new TemporalAbstractionInterval(MSS[i].variables, MSS[i].interval.s - sequenceLength, MSS[i].interval.e - sequenceLength));
                else
                    Z_norm_Dic.Add(i, new TemporalAbstractionInterval(MSS[i].variables, MSS[i].interval.s, MSS[i].interval.e));
            }

            //Calculating Delta
            if (!future)//History                        
                DeltaT = -Z_norm_Dic.Min(x => x.Value.interval.s);//Earliest start means the length (because end is now zero)                            
            else//Future
                DeltaT = Z_norm_Dic.Max(x => x.Value.interval.e);//Earliest start means the length (because end is now zero)                


            //Finding K and N
            int k = 0;

            if (n < 0)//it is not defined
            {
                n = 0;
                k = 1;
                int delta = 1;
                while (DeltaT >= Math.Pow(2, delta * k))
                {
                    n++;
                    k = (int)Math.Pow(2, n);
                }
            }
            else//n was provided
                k = (int)Math.Pow(2, n);

            List<(double start, double end)> p = CreatePeriods(k, future, fixedSize, DeltaT);



            // List<List<List<float>>> matrix = new List<List<List<float>>>();

            double[,,] matrix = new double[k, F, V];


            for (int i = 0; i < k; i++)
            {
                //Create IN and OUT list
                var IN = Z_norm_Dic.Where(x =>
                  ((x.Value.interval.s >= p[i].Item1) && (x.Value.interval.s < p[i].Item2) || ((x.Value.interval.e >= p[i].Item1) && (x.Value.interval.e < p[i].Item2)))).ToList();//.ToDictionary(x => x.Key, x => x.Value);//.ToDictionary (.ToDictionary<int,TemporalAbstraction>( ;//.Select(x=>x.Value).ToList();

                var OUT = Z_norm_Dic.Where(x =>
                   ((x.Value.interval.s < p[i].Item1) && (x.Value.interval.e >= p[i].Item2))).ToList();//.ToDictionary(x=>x.Key,x=>x.Value);//.Select(x => x.Value).ToList();

                double pLength = p[i].Item2 - p[i].Item1;

                foreach (var Ej in IN)
                {
                    if (((p[i].Item1 <= Ej.Value.interval.s) && (Ej.Value.interval.s < p[i].Item2)) &&
                        ((p[i].Item1 <= Ej.Value.interval.e) && (Ej.Value.interval.e < p[i].Item2)))//Start and End inside p[i]
                    {
                        foreach (var fieldValue in Ej.Value.variables)
                            matrix[i, fieldValue.f, fieldValue.v] += Ej.Value.Length;


                        Z_norm_Dic.Remove(Ej.Key);
                    }
                    else if ((p[i].Item1 <= Ej.Value.interval.s) && (Ej.Value.interval.s < p[i].Item2))//Only if starts in p[i]
                    {
                        foreach (var fieldValue in Ej.Value.variables)
                            matrix[i, fieldValue.f, fieldValue.v] += p[i].Item2 - Ej.Value.interval.s;


                        //Do not remove, let it be there for next elements
                    }
                    else if ((p[i].Item1 <= Ej.Value.interval.e) && (Ej.Value.interval.e < p[i].Item2))//Only if ends in p[i]
                    {
                        foreach (var fieldValue in Ej.Value.variables)
                            matrix[i, fieldValue.f, fieldValue.v] += Ej.Value.interval.e - p[i].Item1;

                        Z_norm_Dic.Remove(Ej.Key);
                    }
                }

                foreach (var Ej in OUT)
                {
                    foreach (var fieldValue in Ej.Value.variables)
                        matrix[i, fieldValue.f, fieldValue.v] += pLength;
                }


                //Normalize
                //i = k
                //if (!future)
                    for (int f = 0; f < F; f++)
                    {
                        for (int v = 0; v < V; v++)
                        {
                            matrix[i, f, v] = matrix[i, f, v] / pLength;
                        }
                    }




                /*
                mat = np.matrix(Matrix[i])
            mat = mat / pLength
            Matrix[i] = mat.tolist()
            #print(Matrix[i])

        if (self.binaryFeatures):#This removes the first column (not(binaryFeature)) which is redundant (off, on) [0.0, 1.0] =>[1.0] (on)
            #Remove the first column of ITS elements in the matrix                                                           
            Matrix =[[[x[1]] for x in y] for y in Matrix]
            */
            }
            //Modify for binary feature
            double[,,] matrixBinary = new double[k, F, 1];
            if (binaryFeatures)
            {
                for (int k2 = 0; k2 < k; k2++)
                {
                    for (int f = 0; f < F; f++)
                    {
                        matrixBinary[k2, f, 0] = matrix[k2, f, 1];
                    }
                }
                matrix = matrixBinary;
            }

            return matrix;
        }

        /// <summary>
        /// Creates LifeModel periods for 
        /// </summary>
        /// <param name="k"></param>
        /// <param name="fixedSize"></param>
        /// <param name="DeltaT"></param>
        /// <returns></returns>
        public static List<(double start, double end)> CreatePeriods(int k, bool future = false, bool fixedSize = false, double DeltaT = 32)
        {
            //Create periods
            List<(double start, double end)> p = new List<(double start, double end)>();

            //int elementDeltaT = (int)Math.Ceiling(DeltaT / k);//For possible fixed size only
            double elementDeltaT = (DeltaT / k);//For possible fixed size only
            for (int i = 0; i < k; i++)
            {
                if (!fixedSize)//Life Model
                {
                    if (!future)//History                            
                        p.Add((-Math.Pow(2, k - i) + 1, -Math.Pow(2, k - (i + 1)) + 1));
                    else
                        p.Add((Math.Pow(2, i) - 1, Math.Pow(2, (i + 1)) - 1));
                }
                else if (fixedSize) //Fixed Sized periods for testing and comparison between LifeModel mapping and a fixed Window mapping
                {
                    if (!future)//History                            
                        p.Add((-elementDeltaT * (k - i), -elementDeltaT * (k - (i + 1))));
                    else
                        p.Add((elementDeltaT * (i), -elementDeltaT * ((i + 1))));
                }
            }

            return p;
        }

        void TestLifeModel(int F, int V, int n = 5, bool future = false)
        {
            /*
             * 
'''--------------------------------------Reading sample from thesis
F, V= (3,5)
#C, G, B= (0, 1, 2)
#VL, L, N, H, VH= (0, 1, 2, 3, 4)
Z=[TemporalAbstraction(1,3,1,5),#(G,H,1,4)
   TemporalAbstraction(0,2,2,14),#(C,N,2,14)
   TemporalAbstraction(2,2,4,20),#(B,N,4,20)
   TemporalAbstraction(1,2,6,9),#(G,N,6,9)
   TemporalAbstraction(1,3,10,13),#(G,H,10,13)
   TemporalAbstraction(0,3,15,24),#(C,H,15,24)
   TemporalAbstraction(1,4,16,23)#(G,VH,16,23)
   ]
'''

             * */
            F = 2;
            V = 2;
            // int C=0, G=1, 
            List<TemporalAbstraction> MSS = new List<TemporalAbstraction>();
            for (int i = 0; i < 50; i++)
                MSS.Add(new TemporalAbstraction(i % 2, 1 - i % 2, i, i + 2));
            //Z =[TemporalAbstraction(i % 2, (1 - i % 2), i, i + 2) for i in range(50)]



            LifeModel lm = new LifeModel(F, V, n, future);
            var ITS = lm.GetITS(MSS);


        }


        public List<List<Point>> MapMultivariateTimeSeries(List<Point> mts, TimeSpan delta, bool fixedIntervals = false)
        {
            var result = new List<List<Point>>();

            var periods = LifeModel.CreatePeriods(32, fixedSize: fixedIntervals, DeltaT: delta.TotalSeconds);//32 seconds by default


            //Normalize values to zero (Maximum is zero, everything else is negative)
            var max = mts.Max(x => x.TimeStamp);

            //mts.ForEach(x => );
            mts = mts.Select(x => x.Normalize(max)).ToList();

            //And it is about history



            //foreach (var point in mts)
            {
                for (int i = 0; i < periods.Count; i++)
                {
                    List<Point> intervalPoints = new List<Point>();
                    if (!fixedIntervals)//Life Model->Millisecond periods
                        intervalPoints = mts.Where(x =>
                    ((x.TimeStamp.TotalMilliseconds >= periods[i].start) && (x.TimeStamp.TotalMilliseconds < periods[i].end))).ToList();//.ToDictionary(x => x.Key, x => x.Value);//.ToDictionary (.ToDictionary<int,TemporalAbstraction>( ;//.Select(x=>x.Value).ToList();
                    else//Fixed intervals are second in fractions
                        intervalPoints = mts.Where(x =>
((x.TimeStamp.TotalSeconds >= periods[i].start) && (x.TimeStamp.TotalSeconds < periods[i].end))).ToList();//.ToDictionary(x => x.Key, x => x.Value);//.ToDictionary (.ToDictionary<int,TemporalAbstraction>( ;//.Select(x=>x.Value).ToList();

                    result.Add(intervalPoints);

                    //Binary search later!! or direct finding!
                    //if(point.TimeStamp.)    

                }
            }


            //find each point s locatio in each period and add it to thaf list


            //Give each mappmapped List of points to a mapper function
            // Mapped should contain the MIM mapped interval matrix. and the result is mapped interval sequrnce MIS
            return result;
        }

        public double[,,] GetAverageMappedIntervalSequence(List<Point> mts, Mapper mapper, TimeSpan delta, bool fixedIntervals = false)
        {
            if (mapper == null)
                mapper = new AverageMapper();

            var mappedPeriods = MapMultivariateTimeSeries(mts, delta: delta, fixedIntervals: fixedIntervals);
            F = 4;//No of variables (4 for accelerometers)
            V = 1;//No of values (1 for average)
            var result = new double[mappedPeriods.Count, F, V];

            for (int i = 0; i < mappedPeriods.Count; i++)
            {
                var mappedIntervalPeriods = mapper.Map(mappedPeriods[i], F, V);
                for (int j = 0; j < F; j++)
                {
                    for (int k = 0; k < V; k++)
                    {
                        result[i, j, k] = mappedIntervalPeriods.mappedIntervalMatrixMIM[j, k];
                    }
                }
            }

            return result;
        }

    }

    public abstract class Mapper
    {
        public abstract (double[,] mappedIntervalMatrixMIM, int F, int V) Map(List<Point> period, int F = 0, int V = 0);
    }

    public class AverageMapper : Mapper
    {
        public override (double[,] mappedIntervalMatrixMIM, int F, int V) Map(List<Point> period, int F = 4, int V = 1)
        {
            double[,] mappedIntervalMatrixMIM = new double[F, V];
            if (period.Count > 0) F = period[0].values.Count;

            for (int i = 0; i < F; i++)
            {
                if (period.Count > 0)
                    mappedIntervalMatrixMIM[i, 0] = period.Select(x => x.values[i]).Average();
                else
                    mappedIntervalMatrixMIM[i, 0] = 0;
            }




            return (mappedIntervalMatrixMIM, F, V);// result.Select(x=>x.ToArray()).ToArray();

        }
    }


    public class TemporalAbstraction
    {
        public int f, v;
        public float s, e;
        public TemporalAbstraction(int f, int v, float s, float e)
        {

            this.f = f;
            this.v = v;
            this.s = s;
            this.e = e;
        }

        public float Length { get => e - s; }
        public override string ToString()
        {
            return $"({f}, {v}, {s}, {e})";
        }
    }

    public class TemporalAbstractionInterval
    {
        public struct Variable
        {
            public int f, v;
            public Variable(int fieldVariable, int value)
            {
                this.f = fieldVariable;
                this.v = value;
            }
        }
        //public int f, v;

        public struct Interval
        {
            public float s, e;
            public Interval(float start, float end)
            {
                s = start;
                e = end;
            }
        }
        public List<Variable> variables;

        public Interval interval;


        //public float s, e;

        public TemporalAbstractionInterval(int f, int v, float s, float e)
        {
            variables = new List<Variable>() { new Variable(f, v) };
            interval = new Interval(s, e);
            //this.f = f;
            //this.v = v;
            //this.s = s;
            //this.e = e;
        }

        public TemporalAbstractionInterval(List<Variable> variables, float s, float e)
        {
            this.variables = variables;
            this.interval = new Interval(s, e);
        }

        public TemporalAbstractionInterval(List<Variable> variables, Interval interval)
        {
            this.variables = variables;
            this.interval = interval;
        }

        //public float Length { get => e - s; }
        public float Length { get => interval.e - interval.s; }

        public List<TemporalAbstraction> ToTemporalAbstactions()
        {
            return variables.Select(x => new TemporalAbstraction(x.f, x.v, interval.s, interval.e)).ToList();
        }

        public static List<TemporalAbstractionInterval> FromTemporalAbstractions(List<TemporalAbstraction> multivariateStateSequence)
        {
            Dictionary<Interval, TemporalAbstractionInterval> intervals = new Dictionary<Interval, TemporalAbstractionInterval>();

            foreach (var temporalAbstraction in multivariateStateSequence)
            {
                Interval interval = new Interval(temporalAbstraction.s, temporalAbstraction.e);
                if (intervals.ContainsKey(interval))
                    intervals[interval].variables.Add(new Variable(temporalAbstraction.f, temporalAbstraction.v));
                else
                    intervals[interval] = new TemporalAbstractionInterval(temporalAbstraction.f, temporalAbstraction.v, temporalAbstraction.s, temporalAbstraction.e);// .variables.Add(new Variable(temporalAbstraction.f, temporalAbstraction.v));
            }


            return intervals.Values.ToList();
        }

        public override string ToString()
        {
            string result = $"{interval.s}->{interval.e}=({variables.Count}){{{String.Join(", ", variables.Select(x => $"({x.f}, {x.v})"))}}}";
            return result;
            //return $"({f}, {v}, {s}, {e})";
        }
    }
}
