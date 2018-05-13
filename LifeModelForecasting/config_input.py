from enum import Enum

class Method(Enum):
    LifeModel_Binary_11=11
    Fixed_Binary_12=12
    Reg_Binary_13=13
    LifeModel_Multiple_21=21
    Fixed_Multiple_22=22
    Reg_Multiple_23=23
    LifeModel_Accel_31=31#size(70, 129) Binary
    Fixed_Accel_32=32#size(70, 129) Binary
    reg_LifeModel32_Accel_AutoShift_41=41#(, 130)
    reg_Fixed32_Accel_AutoShift_42=42#(, 130)
    LifeModelForecast_SingleRegression4_mortality_43=43
    RegularForecast_SingleRegression4_mortality_44=44
    LifeModel_Seq2Seq_51=51
    Regular_Seq2Seq_52=52

class InputType(Enum):
    Input=0
    _50Miss=1
    _10Miss=2  

#method selection
method =  Method.LifeModel_Seq2Seq_51

if(method==Method.LifeModel_Seq2Seq_51 or method==Method.Regular_Seq2Seq_52):
    seqToSeq = True


input_file_variable_name = "Input"
#Possible values:
#1  "_50Miss" #Fall Regression
#2  "_10Miss" #Fall Regression
#3  "" #Fall Regression no missing values
#4  "Input" # Mortality
 
mortality = True
#Possible values:
#1  True  --> mortality dataset
#2  False --> fall detection

detection_mode = 1
#Possible values:
#1  1  --> one step (all people including who has not fallen are considered)
#2  2  --> two step (only people who has fallen are considered)

auto_shifted = False
#Possible values:
#1  True  --> autoshifted fallDetection dataset
#2  False --> not autoshifted fallDetection dataset

binaryP = "Binary classification problem"
multiClassP = "MultiClass classification problem"
regressionP = "Regression problem"

problem_type = regressionP

#if it is future prediction or binary  classification
omit=True
#Possible values:
#1  True  --> prediction
#2  False --> binary


#for TE lost function
tolerance = 4096
#tolerance = 83000
toleranceIndex = 23

#value we assign to people who has not fallen
not_fallen = -1


if(method==Method.LifeModel_Accel_31 or
   method==Method.Fixed_Accel_32 or
   method==Method.reg_LifeModel32_Accel_AutoShift_41 or
   method==Method.reg_Fixed32_Accel_AutoShift_42):
    mortality=False
    
if(method==Method.LifeModel_Accel_31 or method==Method.Fixed_Accel_32):
    seqToSeq = False
    input_file_variable_name = ""
    mortality = False
    detection_mode = 1
    auto_shifted = False
    problem_type = binaryP
    omit=False

elif(method==Method.reg_LifeModel32_Accel_AutoShift_41 or method==Method.reg_Fixed32_Accel_AutoShift_42):
    seqToSeq = False
    input_file_variable_name = "_50Miss" # "" or "_10Miss" or "_50Miss"
    mortality = False
    detection_mode = 2 # 1 or 2 
    auto_shifted = True
    problem_type = regressionP
    omit=True

elif(method==Method.LifeModelForecast_SingleRegression4_mortality_43 or method==Method.RegularForecast_SingleRegression4_mortality_44):
    seqToSeq = False
    input_file_variable_name = "Input" 
    mortality = True
    detection_mode = 2
    auto_shifted = False
    problem_type = regressionP
    omit=True
elif(method==Method.LifeModel_Seq2Seq_51 or method==Method.Regular_Seq2Seq_52):
    seqToSeq = True
    input_file_variable_name = "Input" 
    mortality = True
    detection_mode = 1
    auto_shifted = False
    problem_type = regressionP
    omit=True


if(not omit):
    limit = max = 5200
else:
    limit = max = 34100

if(method==Method.LifeModelForecast_SingleRegression4_mortality_43 or method==Method.RegularForecast_SingleRegression4_mortality_44):
    limit = max = 5200
#network parameters:
#values for LifeModel of mortaliy dataset
F=8371 #dimentionality of each timestep in each sequence for each sample
K=32   #sequence length

if(not mortality):
    #values for Accel data (fall detection)
    if(auto_shifted):
        limit = max = 500
    else:
        limit = max = 70
    F = 4
    K = 32

#C:\Users\manashty\Documents\LifeModelForecasting\LifeModelForecasting
curDir=""

if(method==Method.LifeModel_Binary_11):#LifeModel
    filename=curDir+'MIMICIII_Diag_Proc_LM_{0}12-9-2017 7-10-28 PM.csv'
elif (method==Method.Fixed_Binary_12):##Fixed
    filename=curDir+'MIMICIII_Diag_Proc_Fixed_{0}12-10-2017 8-41-34 PM.csv'
elif (method==Method.Reg_Binary_13):##Reg
    filename=curDir+'All Patients Reg Fixed\MIMICIII_Diag_Proc_Reg_{0}1-9-2018 3-51-30 PM.csv'

##Omit Life Model multiple output
#??? by "Multiple output"you mean sequene?
if(method==Method.LifeModel_Multiple_21):#LifeModel
    filename=curDir+"MIMICIII_Diag_Proc_LM_Omit_{0}1-10-2018 2-20-50 PM.csv"
elif (method==Method.Fixed_Multiple_22):##Fixed
    filename=curDir+""
elif (method==Method.Reg_Multiple_23):##Reg
    filename=curDir+'\All Patients Reg Fixed\MIMICIII_Diag_Proc_Reg_{0}1-9-2018 3-51-30 PM.csv'

if(method==Method.LifeModel_Accel_31):#LifeModel
    filename=curDir+"FallDataAvgLifeModel32_{}2018-03-10  15-17-39.csv"
elif (method== Method.Fixed_Accel_32):#Fixed
    filename=curDir+"FallDataAvgFixed32_{}2018-03-10  15-17-39.csv"

if(method==Method.reg_LifeModel32_Accel_AutoShift_41):
    filename=curDir+'FallDataAvgLifeModel32AutoShift{0}_2018-03-10  15-17-39.csv'
elif(method==Method.reg_Fixed32_Accel_AutoShift_42):
    filename=curDir+'FallDataAvgFixed32AutoShift{0}_2018-03-10  15-17-39.csv'
elif(method==Method.LifeModelForecast_SingleRegression4_mortality_43):
    filename=curDir+'MIMICIII_Diag_Proc_LM_Forecast_SingleRegression4_{}3-13-2018 7-19-42 PM.csv'
elif(method==Method.RegularForecast_SingleRegression4_mortality_44):
    filename=curDir+'MIMICIII_Diag_Proc_Reg_Forecast_SingleRegression4_{}3-28-2018 5-10-01 AM.csv'

if(method==Method.LifeModel_Seq2Seq_51):
    filename=curDir+'MIMICIII_Diag_Proc_LM_Forecast_Seq2SeqDiagnosisLM4_{}3-28-2018 4-01-38 AM.csv'
elif(method==Method.Regular_Seq2Seq_52):
    filename=curDir+'MIMICIII_Diag_Proc_Reg_Forecast_Seq2SeqDiagnosisReg4_{}3-28-2018 4-01-38 AM.csv'

