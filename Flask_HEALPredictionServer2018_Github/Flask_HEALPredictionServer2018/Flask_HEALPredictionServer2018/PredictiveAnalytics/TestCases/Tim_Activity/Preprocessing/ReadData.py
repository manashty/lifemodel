"""
Reads the InputSequence.csv file  and generates a MSS in variable 'Z'. MSS is a list of temporal abstractions.
It is only for one house.
"""
from PredictiveAnalytics.TemporalModelling import TemporalAbstraction
import csv

#-----------------------Reading Tim van Kastern's Data------------------
Z_house=[]
f = open('PredictiveAnalytics/TestCases/Tim_Activity/'+ 'InputSequence.csv', 'rt')
try:
    reader = csv.reader(f)
    next(reader, None)#Skip the header
    for row in reader:
        Z_house.append(TemporalAbstraction(int(row[0])-1,0,float(row[1]),float(row[2])))
finally:
    f.close()
print(Z_house)
F, V= (max(e.f for e in Z_house)+1,1)
Z=Z_house
#-----------------------Reading Tim van Kastern's Data------------------