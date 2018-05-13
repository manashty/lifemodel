import numpy
import keras.backend as Kernel
import tensorflow
import math
from config import toleranceIndex, tolerance

def accForReg(y_test_all, result):
    # y_test_all: True values
    y_test_all = numpy.array(y_test_all).reshape((len(y_test_all),1))
    # result: Predicted values
    countOfAllSamples = len(y_test_all)
    correctCounter = 0
    for i in range(countOfAllSamples):
        if(round(y_test_all[i][0]) == round(result[i][0])):
            correctCounter = correctCounter + 1
    return ((correctCounter / countOfAllSamples) * 100)

def TE(y_true, y_pred):
    # y_true: True values
    y_true = numpy.array(y_true).reshape((len(y_true),1))
    # y_pred: Predicted values
    countOfAllSamples = len(y_true)
    sum = 0

    for i in range(countOfAllSamples):
        sum = sum + math.sqrt(math.pow((math.pow(2, y_true[i][0]) - math.pow(2, y_pred[i][0])) / tolerance, 2))
        
    return (sum)

def TE_Loss(y_true, y_pred):
    return TE(y_true, y_pred)

def Mortality_Seq2seq_Metric(y_true, y_pred):
    # y_true: True values - 3D - samples*seqLength*features n*K*F
    
    y_true = numpy.array(y_true)
    # y_pred: Predicted values - 3D - samples*seqLength*features n*K*F
    print('Shape Y_True={0}, Shape Y_pred={1}'.format(y_true.shape,y_pred.shape))

    countOfAllSamples = y_true.shape[0]
    countOfAllSamplesN = y_true.shape[0]
    countOfAllSamplesK = y_true.shape[1]
    countOfAllSamplesF = y_true.shape[2]
    #print('Count N={0}, CountK={1},
    #CountF={2}'.format(countOfAllSamplesN,countOfAllSamplesK,countOfAllSamplesF))

    outersum = 0

    tensorflow.cast(y_true, tensorflow.float64)
    tensorflow.cast(y_pred, tensorflow.float64)

    for n in range(countOfAllSamples):
        for i in range(countOfAllSamplesF):
            innerSum = 0
            for j in range(countOfAllSamplesK):
                #innerSum = innerSum +
                #Kernel.pow((Kernel.abs(y_true[n][i][j]-y_pred[n][i][j]))*(Kernel.pow(2,j)/(Kernel.pow(2,tolerance))),2)
                #print('Just before index n={0} of {1}, j(K)={2} of {3}, i
                #(F)={4} of {5}'.format(n, countOfAllSamples,
                #j,countOfAllSamplesK, i,countOfAllSamplesF))
                diffPart = math.fabs(y_true[n][j][i] - y_pred[n][j][i])
                #print('DiffPart={0}'.format(diffPart))
                sum_part = math.pow(diffPart * (math.pow(2,j) / (math.pow(2,toleranceIndex))),2)
                #sum_part=math.ldexp(diffPart,j-toleranceIndex)
                #(math.pow(2,j)/(math.lp.pow(2,toleranceIndex))),2)
                #print('Sum_part={0}'.format(sum_part))
                innerSum = innerSum + sum_part
                #print('Inner Sum={0}'.format(innerSum))
                
                #innerSum = innerSum +
                #math.pow((math.fabs(y_true[n][i][j]-y_pred[n][i][j]))*(math.pow(2,j)/(math.pow(2,tolerance))),2)
                #innerSum = innerSum +
                #(Kernel.abs(y_true[i][j][p]-y_pred[i][j][p]))
                # I didn't use the power of 2 in the formula.  Mehrdad
            outersum = outersum + innerSum            
            #print('OuterSum={0}'.format(outersum))
    return (math.sqrt(outersum))#/Kernel.pow(2, tolerance))


#Trying to make it work with Keras kernel functions only
m = [2 ** i for i in range(32)]
m2 = [m for i in range(8371)]
m2tran = numpy.transpose(numpy.array(m2))

#Mean Tolerance Error
def MTE(y_true, y_pred):
    # y_true: True values - 3D - samples*seqLength*features n*K*F
    #y_true=Kernel.eval(y_true)
    #print(y_true)
    #y_true = numpy.array(y_true)
    # y_pred: Predicted values - 3D - samples*seqLength*features n*K*F
    #print('Shape Y_True={0}, Shape Y_pred={1},
    #K={3}'.format(Kernel.int_shape(y_true),Kernel.int_shape(y_pred),0))#Kernel.int_shape(y_true)[1]))
    
    #M=Kernel.var
    #K=Kernel.variable(Kernel.int_shape(y_true)[1])
    #m = [2 ** i for i in range(32)]
    #m2 = [m for i in range(8371)]
    #m2tran = numpy.transpose(numpy.array(m2))
    m3 = Kernel.constant(m2tran)
    
    #K = Kernel.variable(32)
    #M = Kernel.ones((1,K)) * 2
    #M2=Kernel.
    step1 = y_true - y_pred
    step2 = Kernel.abs(step1)
    step3 = step2 * m3#Kernel.variable(m)
    #step3 = step2
    step4 = Kernel.pow(step3,2)
    step5 = Kernel.sum(step4)
    step6 = Kernel.sqrt(step5)
    
    return step6/ 1000000
        
    ####countOfAllSamples = Kernel.int_shape(y_true)[0]
    ####countOfAllSamplesN = Kernel.int_shape(y_true)[0]
    ####countOfAllSamplesK = Kernel.int_shape(y_true)[1]
    ####countOfAllSamplesF = Kernel.int_shape(y_true)[2]
    ####print('Count N={0}, CountK={1},
    ####CountF={2}'.format(countOfAllSamplesN,countOfAllSamplesK,countOfAllSamplesF))

    
    ####outersum = 0.

    #####tensorflow.cast(y_true, tensorflow.float64)
    #####tensorflow.cast(y_pred, tensorflow.float64)
    ####if(countOfAllSamples is not None):
    ####    for n in range(countOfAllSamples):
    ####        for i in range(countOfAllSamplesF):
    ####            innerSum = 0.
    ####            for j in range(countOfAllSamplesK):
    ####                #innerSum = innerSum +
    ####                Kernel.pow((Kernel.abs(y_true[n][i][j]-y_pred[n][i][j]))*(Kernel.pow(2,j)/(Kernel.pow(2,tolerance))),2)
    ####                print('Just before index n={0} of {1}, j(K)={2} of {3},
    ####                i (F)={4} of {5}'.format(n, countOfAllSamples,
    ####                j,countOfAllSamplesK, i,countOfAllSamplesF))
    ####                diffPart=Kernel.abs(y_true[n][j][i]-y_pred[n][j][i])
    ####                #print('DiffPart={0}'.format(diffPart))
    ####                sum_part=Kernel.pow(Kernel.dot(diffPart,Kernel.pow(2,Kernel.pow(Kernel.variable(j-toleranceIndex),2))),2)
    ####                #sum_part=math.ldexp(diffPart,j-toleranceIndex)
    ####                (math.pow(2,j)/(math.lp.pow(2,toleranceIndex))),2)
    ####                #print('Sum_part={0}'.format(sum_part))
    ####                innerSum=innerSum+sum_part
    ####                #print('Inner Sum={0}'.format(innerSum))
                    
    ####                #innerSum = innerSum +
    ####                math.pow((math.fabs(y_true[n][i][j]-y_pred[n][i][j]))*(math.pow(2,j)/(math.pow(2,tolerance))),2)
    ####                #innerSum = innerSum +
    ####                (Kernel.abs(y_true[i][j][p]-y_pred[i][j][p]))
    ####                # I didn't use the power of 2 in the formula.  Mehrdad
    ####            outersum = outersum + innerSum
    ####            #print('OuterSum={0}'.format(outersum))
    ####return (Kernel.sqrt(Kernel.variable(outersum)))#/Kernel.pow(2,
    ####tolerance))

def Mortality_Seq2seq_Metric_Loss(y_true, y_pred):
    # y_true: True values - 3D - samples*seqLength*features n*K*F
    #y_true=Kernel.eval(y_true)
    #print(y_true)
    #y_true = numpy.array(y_true)
    # y_pred: Predicted values - 3D - samples*seqLength*features n*K*F
    print('Shape Y_True={0}, Shape Y_pred={1}'.format(Kernel.int_shape(y_true),Kernel.int_shape(y_pred)))

    
    countOfAllSamples = Kernel.int_shape(y_true)[0]
    countOfAllSamplesN = Kernel.int_shape(y_true)[0]
    countOfAllSamplesK = Kernel.int_shape(y_true)[1]
    countOfAllSamplesF = Kernel.int_shape(y_true)[2]
    print('Count N={0}, CountK={1}, CountF={2}'.format(countOfAllSamplesN,countOfAllSamplesK,countOfAllSamplesF))

    
    outersum = 0.

    #tensorflow.cast(y_true, tensorflow.float64)
    #tensorflow.cast(y_pred, tensorflow.float64)
    if(countOfAllSamples is not None):
        for n in range(countOfAllSamples):
            for i in range(countOfAllSamplesF):
                innerSum = 0.
                for j in range(countOfAllSamplesK):
                    #innerSum = innerSum +
                    #Kernel.pow((Kernel.abs(y_true[n][i][j]-y_pred[n][i][j]))*(Kernel.pow(2,j)/(Kernel.pow(2,tolerance))),2)
                    print('Just before index n={0} of {1}, j(K)={2} of {3}, i (F)={4} of {5}'.format(n, countOfAllSamples, j,countOfAllSamplesK, i,countOfAllSamplesF))
                    diffPart = Kernel.abs(y_true[n][j][i] - y_pred[n][j][i])
                    #print('DiffPart={0}'.format(diffPart))
                    sum_part = Kernel.pow(Kernel.dot(diffPart,Kernel.pow(2,Kernel.pow(Kernel.variable(j - toleranceIndex),2))),2)                    
                    #sum_part=math.ldexp(diffPart,j-toleranceIndex)
                    #(math.pow(2,j)/(math.lp.pow(2,toleranceIndex))),2)
                    #print('Sum_part={0}'.format(sum_part))
                    innerSum = innerSum + sum_part
                    #print('Inner Sum={0}'.format(innerSum))
                    
                    #innerSum = innerSum +
                    #math.pow((math.fabs(y_true[n][i][j]-y_pred[n][i][j]))*(math.pow(2,j)/(math.pow(2,tolerance))),2)
                    #innerSum = innerSum +
                    #(Kernel.abs(y_true[i][j][p]-y_pred[i][j][p]))
                    # I didn't use the power of 2 in the formula.  Mehrdad
                outersum = outersum + innerSum            
                #print('OuterSum={0}'.format(outersum))
    return (Kernel.sqrt(Kernel.variable(outersum)))#/Kernel.pow(2, tolerance))

