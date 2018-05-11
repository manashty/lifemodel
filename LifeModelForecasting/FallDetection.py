#for floating point division
from __future__ import division

script_version = '2.6'
date = 'March 15th, 2018, Tehran'

import keras.backend as Kernel

import seq2seq
from seq2seq.models import SimpleSeq2Seq, Seq2Seq

import numpy
import sys
import math



from ReadBatchZipLM_Omit_Accel_Readdata_reg import seqToSeq,binaryP, multiClassP, regressionP, problem_type, K, F, numberOfSamples, Y, batch_read_thread, reset, method, filename, omit, mortality, input_file_variable_name, max, tolerance, detection_mode, Y_batch_read_thread, toleranceIndex

#USE CPU ONLY
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"       #'-1' for cpu and [0,1,2,3,] for gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        #https://goo.gl/bZ6eiD
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, f1_score, classification_report, recall_score, brier_score_loss, precision_score, regression
    
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from keras import metrics
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, RepeatVector, LSTM, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.callbacks import TensorBoard
from keras.preprocessing import sequence
import tensorflow
import threading
import queue
import pickle
import jsonpickle
import logging
import matplotlib.pyplot as plt
#from keras import callbacks
class Epoch(object):
    def __init__(self, e, **kwargs):
        self.number = e
        self.data = dict()
        self.input_output = dict()


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
def Mortality_Seq2seq_Metric_Loss_Kernel(y_true, y_pred):
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
    
    return step6
        
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



# fix random seed for reproducibility
numpy.random.seed(7)

sequence_length = K
n_features = F

model = Sequential()


if(seqToSeq):
    if(False):
        model = Seq2Seq(input_shape = (sequence_length, n_features), hidden_dim = 25,
                    output_length = sequence_length, output_dim = n_features,
                    depth = 1, peek = True # where the decoder gets a 'peek' at the context vector at every timestep
                    )
        model.compile(#loss = 'mse' ,
                    optimizer = 'rmsprop'#metrics=[TE]
                      ,loss=Mortality_Seq2seq_Metric_Loss
                      , metrics=['mean_squared_error'])

    

    #totalSamples, minibatchSize, testPercentage, epochs, minibatch_epochs =
    #(40, 20, 0.1, 1, 1)
    totalSamples, minibatchSize, testPercentage, epochs, minibatch_epochs = (100, 20, 0.1, 2, 1)
    #totalSamples, minibatchSize, testPercentage, epochs, minibatch_epochs
    #=(1000, 100, 0.1, 10, 1)
    #totalSamples, minibatchSize, testPercentage, epochs, minibatch_epochs
    #=(1000, 100, 0.1, 10, 2)
    #totalSamples, minibatchSize, testPercentage, epochs, minibatch_epochs
    #=(1000, 100, 0.2, 50, 2)
    #totalSamples, minibatchSize, testPercentage, epochs, minibatch_epochs
    #=(1000, 10, 0.2, 10, 2)
    #totalSamples, minibatchSize, testPercentage, epochs, minibatch_epochs
    #=(5000, 100, 0.2, 50, 2)

    no_mini_batches = int(totalSamples / minibatchSize)    

    X_test_all = []
    y_test_all = []

    import datetime
    dateformatFile = '%Y-%m-%d  %H-%M-%S'

    directory = "Results " + str(datetime.datetime.now().strftime(dateformatFile)) + ' Samples-{0}_MB-{1}_TestPerc{2}_{3}'.format(totalSamples,minibatchSize, int(testPercentage * 100),method)

    print(directory)

    if not os.path.exists(directory):
        os.makedirs(directory)


    #An empty file with the name of the input file!
    #open(directory+'/'+filename.format("Input"),'wb').close()
    logFile = open(directory + '/' + 'log.txt','wt')

    logging.basicConfig(filename=directory + '/' + 'log.txt',format='%(message)s', level=logging.DEBUG)

    #Adding log to console as well
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logging.Formatter('*\t%(message)s'))
    logging.getLogger().addHandler(consoleHandler)
    dateformat = "%Y/%m/%d  %H:%M:%S"
    logging.info("Script version: " + script_version)
    logging.info("Log file created at " + datetime.datetime.now().strftime("%Y/%m/%d  %H:%M:%S"))
    logging.info("Directory: {0}".format(directory))
    logging.info("Filename Template: {0}".format(filename.format(input_file_variable_name)))
    logging.info("Filename Input: {0}".format(filename.format(input_file_variable_name)))
    logging.info("Filename Output: {0}".format(filename.format("Output")))

    tensorboard = TensorBoard(log_dir=directory, write_graph=True)

    startTime = datetime.datetime.now()
    logging.info('Start time: ' + str(startTime))    
    logging.info((model.summary()))

    accuracy = []
    mean_square_error = []

    metricsReports = []
    q = queue.Queue(1)
    q2 = queue.Queue(1)
    epochsData = []

    #tensorflow.get_default_graph().finalize()
    #to solve this problem: GraphDef cannot be larger than 2GB

    if(True):
        #mehrdad
        sequence_length = K
        n_features = F
        model = Sequential()
        model.add(LSTM(100, input_shape=(sequence_length, n_features)))
        model.add(RepeatVector(sequence_length))
        model.add(LSTM(50, return_sequences=True))
        model.add(TimeDistributed(Dense(n_features
                                       #,activation='sigmoid'
                                        )))
        # relu --> worse result
        model.compile(loss=Mortality_Seq2seq_Metric_Loss_Kernel, optimizer='adam', metrics=['mean_squared_error',Mortality_Seq2seq_Metric_Loss_Kernel])
        #model.compile(loss='mse', optimizer='adam',
        #metrics=['mean_squared_error'])

    for e in range(epochs):

        epoch_start_time = datetime.datetime.now()
        logging.info("\t##########################################################################")
        logging.info("\t############# Epoch {0} started @ {1} ####################".format(e + 1,str(epoch_start_time.strftime(dateformat))))
        logging.info("\t##########################################################################")
    
        epoch = Epoch(e)
    
        X_test_all = []
        y_test_all = []
        X2 = []
        Y2 = []
    
        if e == 0:#No need for reseting the file if it is not the first minibatch
            #print("Reading Thread Starting (Epoch Thread)")
            reset()
            readingThread = threading.Thread(target=batch_read_thread,args=(q,minibatchSize))
            readingThread.start()
            readingThread2 = threading.Thread(target=Y_batch_read_thread,args=(q2,minibatchSize))
            readingThread2.start()


        if e > 0:#Should not be necessary anymore
            readingThread.join()
            readingThread2.join()

    
        batch_data = []
        epoch.input_output['all_batches_training_history_loss'] = []
        epoch.input_output['all_batches_training_history_accuracy'] = []
        epoch.input_output['all_batches_training_history_mse'] = []

        #tbCallBack = TensorBoard(log_dir='./Graph',
        #                                         histogram_freq=0,
        #                                         write_graph=True,
        #                                         write_images=True)
    
        for minibatch in range(no_mini_batches):
            #Read samples and prepare data
            minibatch_start_time = datetime.datetime.now()
            logging.info("*********Epoch {2} of {3}, Minibatch {0} of {1} @ {4}********".format(minibatch + 1,no_mini_batches,e + 1,epochs,str(minibatch_start_time.strftime(dateformat))))
            logging.info("Samples:{0}, {3}, MBatch: {1}, Test {2}, Total {4}, MB:{5}".format(totalSamples,minibatchSize,testPercentage,method, int(testPercentage * totalSamples),int(testPercentage * minibatchSize)))
      
            #readingThread.start()
            readingThread.join()
            readingThread2.join()

            #print("Finished!")
            X2 = q.get()
            Y2 = q2.get()
            X = X2
            Y = Y2

            #print(numpy.array(X).shape)
            #print(numpy.array(Y).shape)

            if(mortality == False and detection_mode == 2):
                te = list(numpy.random.randint(0, numberOfSamples,size=int((numberOfSamples * testPercentage))))#Rand 10 percent for testing
                tr = set(list(range(0, numberOfSamples))) - set(te)#The remaining 90 percent for training
            else:
                te = list(numpy.random.randint(0,len(X),size=int((len(X) * testPercentage))))#Rand 10 percent for testing
                tr = set(list(range(0,len(X)))) - set(te)#The remaining 90 percent for training

                #y_te = list(numpy.random.randint(0,len(Y),size=int((len(Y) *
                #testPercentage))))#Rand 10 percent for testing
                #y_tr = set(list(range(0,len(Y)))) - set(y_te)#The remaining 90
                #percent for training
                #print(tr)
                #print(te)
                #print(y_tr)
                #print(y_te)
                #print(len(tr))
                #print(len(y_tr))
        
            offset = minibatchSize * minibatch

            X_train = numpy.array([X[i] for i in tr])
            y_train = numpy.array([Y[i] for i in tr])
            #y_train = numpy.array([Y[offset + i] for i in tr])
            X_test = [X[i] for i in te]
            y_test = [Y[i] for i in te]
            #y_test = [Y[offset + i] for i in te]

            X_test_all.extend(X_test)
            y_test_all.extend(y_test)       

            if(minibatch == (no_mini_batches - 1)):#If last minibatch, reset to read the first batch for next epoch
                reset()

            readingThread = threading.Thread(target=batch_read_thread,args=(q,minibatchSize))
            readingThread.start()
            
            readingThread2 = threading.Thread(target=Y_batch_read_thread,args=(q2,minibatchSize))
            readingThread2.start()

            #print("Xtrain {}".format(X_train.shape))
            #print("Ytrain {}".format(y_train.shape))
            #print("Xtest {}".format(numpy.array(X_test).shape))
            #print("Ytest {}".format(numpy.array(y_test).shape))

            #tbCallBack = TensorBoard(log_dir='./Graph',
            #                                     histogram_freq=0,
            #                                    write_graph=True,
            #                                     write_images=True)

            history = model.fit(X_train, 
                                y_train,
                                #X_train,
                                epochs=minibatch_epochs, batch_size=1
                                #,callbacks=[tensorboard,
                                #           tbCallBack
                                #           ]
                                )

            logging.info("Batch History: ")
            logging.info(str(history.history.items()))
        
            scores = model.evaluate(numpy.array(X_test), numpy.array(X_test))#, verbose=0)
            epoch.input_output['all_batches_training_history_loss'].append((history.history['loss']))

            epoch.input_output['all_batches_training_history_loss'].append((history.history['mean_squared_error']))

            logging.info("**END****Epoch {2} of {3}, Minibatch {0} of {1} in {4} ********".format(minibatch + 1,no_mini_batches,e + 1,epochs,str((datetime.datetime.now() - minibatch_start_time))))
            logging.info('')
        
        # Final evaluation of the model
        result = model.predict_on_batch(numpy.array(X_test_all))
        train_result = model.predict_on_batch(numpy.array(X_train))

        scores = model.evaluate(numpy.array(X_test_all), numpy.array(X_test_all), verbose=0)
    

        epochsData.append(epoch)
   
        #logging.info(result_both)
        #logging.info("Predictions on all:")
    
        logging.info("")
        logging.info("*****************************************")
        logging.info("***Epoch Summary***")
        logging.info("Parameters: Samples:{0}, Minibatch Size: {1}, Test ratio: {2} ".format(totalSamples,minibatchSize,testPercentage))    
        logging.info(str(epoch.data))
   

        logging.info("***END OF EPOCH {0} of {1}** @ {2} in {3}*".format(e + 1, epochs,datetime.datetime.now().strftime(dateformat),str((datetime.datetime.now() - epoch_start_time))))
        logging.info("*****************************************")
        logging.info("")

        pickleFile = open(directory + '/' + 'data{0}.bin'.format(e + 1),'wb')
        jsonFile = open(directory + '/' + 'data{0}.txt'.format(e + 1),'wt')
        epochFile = open(directory + '/' + 'epoch{0}.txt'.format(e + 1),'wt')
    
        model.save(directory + '/' + 'model_keras{0}.h5'.format(e + 1))


        import json
        pickle.dump(epoch,pickleFile)
        pickleFile.flush()
        pickleFile.close()

        epochFile.write(str(epoch.number))
        epochFile.write(str(epoch.data))
        epochFile.write(str(epoch.input_output))
        epochFile.close()

        jsonFile.write(jsonpickle.encode(epoch))        
        #json.dump([epoch.number,epoch.data, epoch.input_output],jsonFile)
        jsonFile.flush()
        jsonFile.close()


        print("Test Mortality_Seq2seq_Metric for seq2seq: ")
        print(Mortality_Seq2seq_Metric(y_test_all,result))
else:
    # Model Definition
    if(seqToSeq == False):
        model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(sequence_length,n_features), name="L1_LSTM"))
        custom_loss = False

        if(mortality):
            if(not omit):   #Binary
                output_layer = 1 
                model.add(Dense(output_layer, activation='sigmoid', name="Output_Dense"))
                #mehrdad.  change it to binary_crossentropy for regression
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])   
            else:   #Forecasting
                #output_layer=4
                output_layer = 1
                model.add(Dense(output_layer, name="Output_Dense"))
                model.compile(loss='mean_squared_error' if  not custom_loss else TE, optimizer='adam', metrics=['mean_squared_error', 'accuracy'])        
        else:
            if (not omit):
                output_layer = 1
                model.add(Dense(output_layer, activation='sigmoid', name="Output_Dense_binary"))
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            else:
                output_layer = 1
                model.add(Dense(output_layer, name="Output_Dense_reg"))
                model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

    #print((model.summary()))
    if(mortality):    
        #totalSamples, minibatchSize, testPercentage, epochs, minibatch_epochs
        #= (40, 20, 0.1, 2, 1)
        #totalSamples, minibatchSize, testPercentage, epochs, minibatch_epochs
        #=(1000, 100, 0.1, 2, 1)
        #totalSamples, minibatchSize, testPercentage, epochs, minibatch_epochs
        #=(1000, 100, 0.1, 10, 2)
        #totalSamples, minibatchSize, testPercentage, epochs, minibatch_epochs
        #=
        #(numberOfSamples, 100, 0.1, 2, 2)
        totalSamples, minibatchSize, testPercentage, epochs, minibatch_epochs = (numberOfSamples, 100, 0.2, 50, 1)   
    else:
        totalSamples, minibatchSize, testPercentage, epochs, minibatch_epochs = (numberOfSamples, numberOfSamples, 0.2, 50, 2)

    no_mini_batches = int(totalSamples / minibatchSize)    

    X_test_all = []
    y_test_all = []

    import datetime
    dateformatFile = '%Y-%m-%d  %H-%M-%S'

    directory = "Results " + str(datetime.datetime.now().strftime(dateformatFile)) + ' Samples-{0}_MB-{1}_TestPerc{2}_{3}'.format(totalSamples,minibatchSize, int(testPercentage * 100),method)

    print(directory)

    if not os.path.exists(directory):
        os.makedirs(directory)


    #An empty file with the name of the input file!
    #open(directory+'/'+filename.format("Input"),'wb').close()
    logFile = open(directory + '/' + 'log.txt','wt')

    logging.basicConfig(filename=directory + '/' + 'log.txt',format='%(message)s', level=logging.DEBUG)

    #Adding log to console as well
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logging.Formatter('*\t%(message)s'))
    logging.getLogger().addHandler(consoleHandler)
    dateformat = "%Y/%m/%d  %H:%M:%S"
    logging.info("Script version: " + script_version)
    logging.info("Log file created at " + datetime.datetime.now().strftime("%Y/%m/%d  %H:%M:%S"))
    logging.info("Directory: {0}".format(directory))
    logging.info("Filename Template: {0}".format(filename.format(input_file_variable_name)))
    logging.info("Filename Input: {0}".format(filename.format(input_file_variable_name)))
    logging.info("Filename Output: {0}".format(filename.format("Output")))

    tensorboard = TensorBoard(log_dir=directory, write_graph=True)

    startTime = datetime.datetime.now()
    logging.info('Start time: ' + str(startTime))    
    logging.info((model.summary()))

    accuracy = []
    mean_square_error = []

    metricsReports = []
    q = queue.Queue(1)
    #it was 1, for seq2seq has changed to 2
    epochsData = []

    for e in range(epochs):

        epoch_start_time = datetime.datetime.now()
        logging.info("\t##########################################################################")
        logging.info("\t############# Epoch {0} started @ {1} ####################".format(e + 1,str(epoch_start_time.strftime(dateformat))))
        logging.info("\t##########################################################################")
    
        epoch = Epoch(e)
    
        X_test_all = []
        y_test_all = []
        X2 = []
    
        if e == 0:#No need for reseting the file if it is not the first minibatch
            #print("Reading Thread Starting (Epoch Thread)")
            reset()
            readingThread = threading.Thread(target=batch_read_thread,args=(q,minibatchSize))
            readingThread.start()

        if e > 0:#Should not be necessary anymore
            readingThread.join()

    
        batch_data = []
        epoch.input_output['all_batches_training_history_loss'] = []
        epoch.input_output['all_batches_training_history_accuracy'] = []
        epoch.input_output['all_batches_training_history_mse'] = []
    
        for minibatch in range(no_mini_batches):
            #Read samples and prepare data
            minibatch_start_time = datetime.datetime.now()
            logging.info("*********Epoch {2} of {3}, Minibatch {0} of {1} @ {4}********".format(minibatch + 1,no_mini_batches,e + 1,epochs,str(minibatch_start_time.strftime(dateformat))))
            logging.info("Samples:{0}, {3}, MBatch: {1}, Test {2}, Total {4}, MB:{5}".format(totalSamples,minibatchSize,testPercentage,method, int(testPercentage * totalSamples),int(testPercentage * minibatchSize)))
      
            #readingThread.start()
            readingThread.join()

            #print("Finished!")
            X2 = q.get()

            X = X2

            print("size of X in every batch: {}".format(numpy.array(X).shape))

            if(mortality == False and detection_mode == 2):
                te = list(numpy.random.randint(0, numberOfSamples,size=int((numberOfSamples * testPercentage))))#Rand 10 percent for testing
                tr = set(list(range(0, numberOfSamples))) - set(te)#The remaining 90 percent for training
            else:
                te = list(numpy.random.randint(0,len(X),size=int((len(X) * testPercentage))))#Rand 10 percent for testing
                tr = set(list(range(0,len(X)))) - set(te)#The remaining 90 percent for training
        
            offset = minibatchSize * minibatch

            X_train = numpy.array([X[i] for i in tr])
            y_train = numpy.array([Y[offset + i] for i in tr])
            X_test = [X[i] for i in te]
            y_test = [Y[offset + i] for i in te]

            X_test_all.extend(X_test)
            y_test_all.extend(y_test)       

            if(minibatch == (no_mini_batches - 1)):#If last minibatch, reset to read the first batch for next epoch
                reset()

            #mehrad : if we add offset to this function., problem will be
            #solved
            readingThread = threading.Thread(target=batch_read_thread,args=(q,minibatchSize))
            readingThread.start()
        

            history = model.fit(X_train, y_train,
                                epochs=minibatch_epochs, batch_size=1,
                                callbacks=[tensorboard])
            logging.info("Batch History: ")
            logging.info(str(history.history.items()))
        
            scores = model.evaluate(numpy.array(X_test), numpy.array(y_test))#, verbose=0)
            epoch.input_output['all_batches_training_history_loss'].append((history.history['loss']))
        
        
            if(problem_type != regressionP):
                epoch.input_output['all_batches_training_history_accuracy'].append(history.history['acc'])                        
                logging.info("Test Accuracy: %.3f%%" % (scores[1] * 100))            
            else:
                epoch.input_output['all_batches_training_history_mse'].append(history.history['mean_squared_error'])
                logging.info("Test MSE: %.3f" % (scores[1]))
    
        
            logging.info("**END****Epoch {2} of {3}, Minibatch {0} of {1} in {4} ********".format(minibatch + 1,no_mini_batches,e + 1,epochs,str((datetime.datetime.now() - minibatch_start_time))))
            logging.info('')
        
        # Final evaluation of the model
        logging.info("For {0} test samples".format(len(y_test_all)))
        logging.info("True Values")
        logging.info(y_test_all)
        logging.info("Predicted Values")
        result = model.predict_on_batch(numpy.array(X_test_all))
        train_result = model.predict_on_batch(numpy.array(X_train))
        logging.info(result)
        logging.info("Comparison on all:")
        result_both = [(float(result[i]),round(float(result[i])), y_test_all[i]) for i in range(len(result))]
        if(not omit):
            y_score = [round(float(y)) for y in result]
        else:
            y_score = result
     
        scores = model.evaluate(numpy.array(X_test_all), numpy.array(y_test_all), verbose=0)
        accuracy.append((scores[1] * 100, scores[0]))

        if(problem_type != regressionP):
            accuracy.append((scores[1] * 100, scores[0]))
            epoch.data['accuracy'] = scores[1]
        else:
            mean_square_error.append((scores[1], scores[0]))
            epoch.data['mse'] = scores[1]
    
        epoch.data['loss'] = scores[0]
    
        logging.info(result_both)
    
        if(problem_type != regressionP):
            epoch.input_output['classification_report'] = classification_report(y_test_all, y_score)
            metricsReports.append(epoch.input_output['classification_report'])

        if problem_type == multiClassP:
            epoch.data['average_precision'] = average_precision_score(y_test_all, y_score)
            #Mehrdad.  I commented the line below because when all of 7 test
            #samples are 1, it causes an error.
            #https://stackoverflow.com/questions/39018097/sklearn-auc-valueerror-only-one-class-present-in-y-true
            #epoch.data['roc_auc']=roc_auc_score(y_test_all, y_score)
            epoch.input_output['roc_curve'] = roc_curve(y_test_all, y_score)
            epoch.data['f1'] = f1_score(y_test_all, y_score)
            epoch.data['recall'] = recall_score(y_test_all, y_score)
            epoch.data['precision'] = precision_score(y_test_all, y_score)
        
            #Calculate Brier Score
            if hasattr(model, "predict_proba"):
                prob_pos = model.predict_proba(numpy.array(X_test_all))#[:, 1]
            else:  # use decision function
                prob_pos = model.decision_function(numpy.array(X_test_all))
                prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

            epoch.input_output['prob_pos'] = prob_pos

            epoch.input_output['Y_test_all'] = y_test_all
            epoch.input_output['y_predictions'] = y_score

            clf_score = brier_score_loss(y_test_all, prob_pos, pos_label=1)
            brier = clf_score 

            epoch.data['brier'] = clf_score

            logging.info('Brier: {0:0.2f}'.format(clf_score))
                #print(':')
        

        else:
            #Calculating Accuracy for Regression and Tolerance Error if
            if(mortality):  #Change to mortality
                epoch.data['accForReg'] = accForReg(y_train,train_result)
                logging.info("")
                logging.info("Train accuracy for Regression: %.3f%%" % (epoch.data['accForReg']))
                if(detection_mode == 2):
                    epoch.data['Tolerance_Error'] = TE(y_train,train_result)               
                    logging.info("Train TE for regression: ")
                    logging.info((str(epoch.data['Tolerance_Error'])))
                    logging.info("Tolerance Error for train: %.3f%%" % (epoch.data['Tolerance_Error'] * 100))

        
        epochsData.append(epoch)
   
        #logging.info(result_both)
        #logging.info("Predictions on all:")
    
        logging.info("")
        logging.info("*****************************************")
        logging.info("***Epoch Summary***")
        logging.info("Parameters: Samples:{0}, Minibatch Size: {1}, Test ratio: {2} ".format(totalSamples,minibatchSize,testPercentage))    
        logging.info(str(epoch.data))
        logging.info("Loss: %.2f%%" % (scores[0]))
    
        if problem_type == multiClassP:
            logging.info('Average precision-recall: {0:0.2f}'.format(epoch.data['average_precision']))
            logging.info('AU-ROC: {0:0.2f}'.format(epoch.data['roc_auc']))
            logging.info('F1: {0:0.2f}'.format(epoch.data['f1']))    
            logging.info('Recall: {0:0.2f}'.format(epoch.data['recall']))
            logging.info('Precision: {0:0.2f}'.format(epoch.data['precision']))
            logging.info("Accuracy: %.2f%%" % (scores[1] * 100))    
            logging.info("Brier: %.2f%%" % (brier))    
            logging.info("*****************************************")
            logging.info("***Classification Reports***")
    
        if problem_type == multiClassP:
            for ep in range(e + 1):
                logging.info("Epoch {0} accuracy:{1}, loss:{2}".format(ep + 1,accuracy[ep][0],accuracy[ep][1]))
                if(not omit):
                    logging.info(metricsReports[ep])
    
        #print("***Metrics***")
        logging.info("***Metrics***")
        for ep in range(e + 1):
            if (problem_type == regressionP):
                logging.info("Epoch {0} MSE:{1}, loss:{2}".format(ep + 1,accuracy[ep][0],accuracy[ep][1]))
            else:
                logging.info("Epoch {0} accuracy:{1}, loss:{2}".format(ep + 1,accuracy[ep][0],accuracy[ep][1]))

            #print("Epoch {0} metrics".format(ep+1))
            logging.info(epochsData[ep].data)

            #print([str(met) for met in metricsAll[e]])
        logging.info("***Results***")
    
        if (problem_type == regressionP):
            for ep in range(e + 1):
                logging.info("Epoch {0} MSE:{1}, Loss:{2}".format(ep + 1,mean_square_error[ep][0],accuracy[ep][1]))
        else:
            for ep in range(e + 1):
                logging.info("Epoch {0} Accuracy:{1}, Loss:{2}".format(ep + 1,accuracy[ep][0],accuracy[ep][1]))
        
        logging.info("Parameters: Samples:{0}, Method={3}, Minibatch Size: {1}, Test: Ratio: {2}, Total {4}, MB:{5}".format(totalSamples,minibatchSize,testPercentage, method, int(testPercentage * totalSamples),int(testPercentage * minibatchSize)))
        if (problem_type == regressionP):
            logging.info("MSE: %.2f" % (scores[1]))
        else:
            logging.info("Accuracy: %.2f%%" % (scores[1] * 100))
    
        logging.info("***END OF EPOCH {0} of {1}** @ {2} in {3}*".format(e + 1, epochs,datetime.datetime.now().strftime(dateformat),str((datetime.datetime.now() - epoch_start_time))))
        logging.info("*****************************************")
        logging.info("")
    
        if(problem_type != regressionP):
            acc_train = [epoch_iter.input_output['all_batches_training_history_accuracy'][-1][-1] for epoch_iter in epochsData]
            acc = [epoch_iter.data['accuracy'] for epoch_iter in epochsData]
        else:
            mse_train = [epoch_iter.input_output['all_batches_training_history_mse'][-1][-1] for epoch_iter in epochsData]
            mse = [epoch_iter.data['mse'] for epoch_iter in epochsData]

        loss_train = [epoch_iter.input_output['all_batches_training_history_loss'][-1][-1] for epoch_iter in epochsData]
        loss = [epoch_iter.data['loss'] for epoch_iter in epochsData]


        #ta einja mse va acc check shodan
        #mehrdad
        plot = False
        if(plot):
            plt.plot(acc_train)
            plt.plot(acc)
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')

            plt.tight_layout()
            plt.savefig(directory + '/Accuracy' + str(e + 1) + '.pdf',bbox_inches='tight', pad_inches=0)
            #plt.show()
            plt.close()
    
            # summarize history for loss
            plt.plot(loss_train)
            plt.plot(loss)
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            #plt.show()
            plt.tight_layout()
            plt.savefig(directory + '/Loss' + str(e + 1) + '.pdf',bbox_inches='tight', pad_inches=0)
            plt.close()
    
        if(mortality):
            #Plot Brier
            if(not omit):
                fig = plt.figure(1, figsize=(10, 10))
                ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
                ax2 = plt.subplot2grid((3, 1), (2, 0))

                ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
                fraction_of_positives, mean_predicted_value = calibration_curve(y_test_all, prob_pos, n_bins=10)

                epoch.input_output['fraction_of_positives'] = fraction_of_positives
                epoch.input_output['mean_predicted_value'] = mean_predicted_value 


                ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                                label="%s (%1.3f)" % ('Model', clf_score))

                ax2.hist(prob_pos, range=(0, 1), bins=10, label='LSTM',
                                histtype="step", lw=2)

                ax1.set_ylabel("Fraction of positives")
                ax1.set_xlabel("Mean predicted value")

                ax1.set_ylim([-0.05, 1.05])
                ax1.legend(loc="lower right")
                ax1.set_title('Calibration plots  (reliability curve)')

                ax2.set_xlabel("Mean predicted value")
                ax2.set_ylabel("Count")
                ax2.legend(loc="upper center", ncol=2)

                plt.tight_layout()
                plt.savefig(directory + '/Calibration Epoch ' + str(e + 1) + '.pdf',bbox_inches='tight', pad_inches=0)
                plt.close()

                #plt.show()

                #Plot Brier for all

                    #Plot Brier


                fig = plt.figure(1, figsize=(10, 10))
                ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
                ax2 = plt.subplot2grid((3, 1), (2, 0))

                ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
                fraction_of_positives, mean_predicted_value = calibration_curve(y_test_all, prob_pos, n_bins=10)

                epoch.input_output['fraction_of_positives'] = fraction_of_positives
                epoch.input_output['mean_predicted_value'] = mean_predicted_value 


                for x in epochsData:
                    ax1.plot(x.input_output['mean_predicted_value'], x.input_output['fraction_of_positives'], "s-",
                            label="%s (%1.3f)" % ('Epoch{0}'.format(str(x.number + 1)), x.data['brier']))

                #ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                    #            label="%s (%1.3f)" % ('LSTM', clf_score))

                ax2.hist(prob_pos, range=(0, 1), bins=10, label='LSTM',
                                histtype="step", lw=2)

                ax1.set_ylabel("Fraction of positives")
                ax1.set_xlabel("Mean predicted value")

                ax1.set_ylim([-0.05, 1.05])
                ax1.legend(loc="lower right")
                ax1.set_title('Calibration plots  (reliability curve)')

                ax2.set_xlabel("Mean predicted value")
                ax2.set_ylabel("Count")
                ax2.legend(loc="upper center", ncol=2)

                plt.tight_layout()
                plt.savefig(directory + '/Calibration Epoch All ' + str(e + 1) + '.pdf',bbox_inches='tight', pad_inches=0)
                plt.close()


                #Area Under the Curve
                fpr, tpr, _ = epoch.input_output['roc_curve']

                #mehrdad
                '''
                plt.figure()
                lw = 2
                #plt.plot(fpr, tpr, color='darkorange',
                #     lw=lw, label='ROC curve (area = %0.2f)' % epoch.data['roc_auc'])
                plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic')
                plt.legend(loc="lower right")
                plt.savefig(directory+'/AUC Epoch'+str(e+1)+'.pdf',bbox_inches='tight', pad_inches=0)
                plt.close()
                '''
                #mehrdad
                '''
                plt.figure()
                lw = 2
                for x in epochsData:        
                    fpr, tpr, _= x.input_output['roc_curve']
                    plt.plot(fpr, tpr,
                            lw=lw, label='ROC curve Epoch{0} (area = %0.2f)'.format(str(x.number+1)) % x.data['roc_auc'])
                plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic')
                plt.legend(loc="lower right")
                plt.savefig(directory+'/AUC Epochs All'+str(e+1)+'.pdf',bbox_inches='tight', pad_inches=0)
                plt.close()
                '''
        
        pickleFile = open(directory + '/' + 'data{0}.bin'.format(e + 1),'wb')
        jsonFile = open(directory + '/' + 'data{0}.txt'.format(e + 1),'wt')
        epochFile = open(directory + '/' + 'epoch{0}.txt'.format(e + 1),'wt')
    
        model.save(directory + '/' + 'model_keras{0}.h5'.format(e + 1))


        import json
        pickle.dump(epoch,pickleFile)
        pickleFile.flush()
        pickleFile.close()

        epochFile.write(str(epoch.number))
        epochFile.write(str(epoch.data))
        epochFile.write(str(epoch.input_output))
        epochFile.close()

        jsonFile.write(jsonpickle.encode(epoch))        
        #json.dump([epoch.number,epoch.data, epoch.input_output],jsonFile)
        jsonFile.flush()
        jsonFile.close()




    #if(reg_fall):
    print("Test Accuracy for regression: {0:.2f}".format(accForReg(y_test_all,result)))
    if(detection_mode == 2):
        print("Test TE for regression: ")
        print(TE(y_test_all,result))
    #print(numpy.array(y_test_all).reshape((len(y_test_all),1)))
    #print(numpy.array(result).reshape())
