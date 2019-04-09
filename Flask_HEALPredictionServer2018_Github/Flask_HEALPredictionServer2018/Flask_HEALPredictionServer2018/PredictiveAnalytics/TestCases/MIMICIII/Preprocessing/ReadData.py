"""
Reads the InputSequence.csv file  and generates a MSS in variable 'Z'. MSS is a list of temporal abstractions.
It is only for one house.
"""
from PredictiveAnalytics.TemporalModelling import TemporalAbstraction
import csv

#-----------------------Reading MIMIC III Mortality Data------------------
#N patients
#Each patient has M admissions
Z=[]#Dataset of all patients

f = open('PredictiveAnalytics/TestCases/MIMICIII/'+ 'MIMICIII_Diag_Proc_12-7-2017 10-08-34 AM.csv', 'rt')
try:
    reader = csv.reader(f)
    #next(reader, None)#Skip the header
    
    while True:
        #First is a patient, with a 0 or 1 indicating morality, which is the target
        target=next(reader)
        
        if(target is None):#If last line
            break
        
        #Get the mortality target
        patientMortality=int(target[0]) 
        Z_patient=[]#Each patient admissions

        #Reading admissions
        while True:
            nextLine=next(reader)
            if len(nextLine) is 0:#If next line is empty
                break;
            else:
                start=nextLine[0]
                end=nextLine[1]
                values=nextLine[3:]#Feature value (0 or 1)
                F=len(values)
                features=range(F)#Feature number
                for i in features:
                    if(values[i]==''):
                        continue;#The empty string between Diagnosis and Procedure Codes
                    Z_patient.append(TemporalAbstraction(int(features[i]),int(values[i]),float(start),float(end)))#Add all features of the admission
                    #Disasterous Memory management! Very large DB it will be
        #Add patient to dataset            
        Z.append((Z_patient, patientMortality))#Patient and target (Mortality)        
        print ("Patients processed: {p}".format(p=len(Z)))
        if(len(Z)>1000): 
            break
        #print(D[0])
        

                
finally:
    f.close()

#print(Z[0])
#F, V= (max(e.f for e in Z_house)+1,1)

#-----------------------Reading Tim van Kastern's Data------------------