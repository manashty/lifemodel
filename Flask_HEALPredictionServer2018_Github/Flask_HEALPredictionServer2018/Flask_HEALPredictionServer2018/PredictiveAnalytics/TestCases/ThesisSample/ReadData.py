"""
Recreates the sample from thesis and generates the MSS in variable 'Z'. MSS is a list of temporal abstractions.
"""
from PredictiveAnalytics.TemporalModelling import TemporalAbstraction

#-----------------------Reading Tim van Kastern's Data------------------
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
#-----------------------Reading Tim van Kastern's Data------------------

