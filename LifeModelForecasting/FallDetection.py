#region Imports
##################
## IMPORTS
##################
#for floating point division
from __future__ import division
script_version = '3.0'
date = 'May 12, 2018, Canada'

### CONFIG ### SHOULD BE BEFORE EVERYTHING FOR OS CONFIG###
from config import *

### KERAS ###
import keras.backend as Kernel
from keras import metrics
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, RepeatVector, LSTM, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.callbacks import TensorBoard
from keras.preprocessing import sequence
### SEQ 2 SEQ LSTM
import seq2seq
from seq2seq.models import SimpleSeq2Seq, Seq2Seq
### OTHER IMPORTS
import numpy
import sys
import math
import tensorflow
import threading
import queue
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, f1_score, classification_report, recall_score, brier_score_loss, precision_score, regression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

#Local files
from ReadBatchZipLM_Omit_Accel_Readdata_reg import *# K, F, numberOfSamples, Y, batch_read_thread, reset, max, Y_batch_read_thread
from metrics import *
from epoch import *
from visualization import *

#endregion
sequence_length = K
n_features = F

model = Sequential()

###################
### Model Definition
###################
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
    if(True):
        #mehrdad
        sequence_length = K
        n_features = F
        model = Sequential()
        model.add(LSTM(K, input_shape=(sequence_length, n_features)))
        model.add(RepeatVector(sequence_length))
        model.add(LSTM(K, return_sequences=True))
        model.add(TimeDistributed(Dense(n_features
                                       #,activation='sigmoid'
                                        )))
        # relu --> worse result
        metrics_seq2seq = ['mean_squared_error',MTE]#Metrics should be MSE and then MTE for accounting purposes

        if(lossFunction == Seq2SeqLossFunction.MeanSquaredErrorMSE):
            model.compile(loss='mean_squared_error', optimizer='adam', metrics=metrics_seq2seq)
        elif(lossFunction == Seq2SeqLossFunction.MeanToleranceErrorMTE):
            model.compile(loss=MTE, optimizer='adam', metrics=metrics_seq2seq)
        #model.compile(loss='mse', optimizer='adam',
        #metrics=['mean_squared_error'])
 
    X_test_all = []
    y_test_all = []
   
    logging.info("Script version: " + script_version)
    #logging.info((model.summary()))
    #endregion
    
    accuracy = []
    mean_square_error = []
    metricsReports = []

    q = queue.Queue(1)
    q2 = queue.Queue(1)
    all_epochs_Data = []
        
    for e in range(epochs):

        epoch = Epoch(e)
        epoch.Started()
        logging.info((model.summary()))                
    
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
        elif e > 0:#Should not be necessary anymore
            readingThread.join()
            readingThread2.join()
    
        batch_data = []

        #We store the train and test metrics/data for each minibatch here.
        #Usually use the last one as the epoch error.  Which may not be right
        epoch.input_output[metricsDic[Metric.loss.name].batch_train_data_index] = []
        epoch.input_output[metricsDic[Metric.accuracy.name].batch_train_data_index] = [] 
        epoch.input_output[metricsDic[Metric.mse.name].batch_train_data_index] = [] 
        epoch.input_output[metricsDic[Metric.mte.name].batch_train_data_index] = []         
                    
        for minibatch in range(no_mini_batches):
            #Read samples and prepare data
            minibatch_start_time = datetime.datetime.now()
            logging.info("*********Epoch {2} of {3}, Minibatch {0} of {1} @ {4}********".format(minibatch + 1,no_mini_batches,e + 1,epochs,str(minibatch_start_time.strftime(dateformat))))
            logging.info("Samples:{0}, {3}, MBatch: {1}, Test {2}, Total {4}, MB:{5}".format(totalSamples,minibatchSize,testPercentage,method.name, int(testPercentage * totalSamples),int(testPercentage * minibatchSize)))
      
            #Wait for I/O threads to finish
            readingThread.join()
            readingThread2.join()
            #Get the result of the I/O
            X = q.get()
            Y = q2.get()            

            #Calculate train and test percentage
            if(mortality == False and detection_mode == 2):
                te = list(numpy.random.randint(0, numberOfSamples,size=int((numberOfSamples * testPercentage))))#Rand 10 percent for testing
                tr = set(list(range(0, numberOfSamples))) - set(te)#The remaining 90 percent for training
            else:
                te = list(numpy.random.randint(0,len(X),size=int((len(X) * testPercentage))))#Rand 10 percent for testing
                tr = set(list(range(0,len(X)))) - set(te)#The remaining 90 percent for training

                #y_te = list(numpy.random.randint(0,len(Y),size=int((len(Y) *
                #testPercentage))))#Rand 10 percent for testing
                #y_tr = set(list(range(0,len(Y)))) - set(y_te)#The remaining 90
              
        
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

            ####Start reading X for the next batch (yes, already!)
            readingThread = threading.Thread(target=batch_read_thread,args=(q,minibatchSize))
            readingThread.start()
            ###Start reading Y for the next batch (yup!  before the current
            ###batch begins!)
            readingThread2 = threading.Thread(target=Y_batch_read_thread,args=(q2,minibatchSize))
            readingThread2.start()                      

            
            tf_callbacks = []
            if(minibatch == 0 or minibatch == no_mini_batches - 1):#Just write tensorboard for the first and last minibatch
                tf_callbacks = tensorboard_callbacks

            history = model.fit(X_train, 
                                y_train,                                
                                epochs=minibatch_epochs, batch_size=1
                                ,callbacks=tf_callbacks)

            logging.info("Batch History: ")
            logging.info(str(history.history.items()))
        
            scores = model.evaluate(numpy.array(X_test), numpy.array(y_test))#, verbose=0)
            logging.info("***Batch Score Scores: Test MSE={0}, Test MTE={1}***".format(scores[0], scores[1]))

            

            epoch.input_output[metricsDic[Metric.loss.name].batch_train_data_index].append((history.history['loss']))
            #epoch.input_output[metricsDic[Metric.accuracy.name].batch_train_data_index]=[]
            epoch.input_output[metricsDic[Metric.mse.name].batch_train_data_index].append((history.history['mean_squared_error']))
            epoch.input_output[metricsDic[Metric.mte.name].batch_train_data_index].append((history.history['MTE']))        
            
            
            logging.info("**END****Epoch {2} of {3}, Minibatch {0} of {1} in {4} ********".format(minibatch + 1,no_mini_batches,e + 1,epochs,str((datetime.datetime.now() - minibatch_start_time))))
            logging.info('')
            ##########################
            #### END OF MINIBATCH LOOP
            ##########################
        
        # Final evaluation of the model
        #####################################
        ###### 1 EPOCH TRAINING FINISHED
        #####################################
        
        
        #Too big to use for now!  Find a way to store the actual results later!
        #result = model.predict_on_batch(numpy.array(X_test_all))#

        #Remove these
        #train_result = model.predict_on_batch(numpy.array(X_train))#Final
        #metrics on the last train.  Better approach is to get the average of
        #the last epoch
        #scores_train_last_batch = model.evaluate(numpy.array(X_train),
        #numpy.array(y_train), verbose=0)
        #epoch.data['mse_train'] = scores_train_last_batch[0]
        #epoch.data['mte_train'] = scores_train_last_batch[1]

        scores_test_all = model.evaluate(numpy.array(X_test_all), numpy.array(y_test_all), verbose=0)        
        epoch.data[metricsDic[Metric.loss.name].epoch_test_data_index] = scores_test_all[0]
        epoch.data[metricsDic[Metric.mse.name].epoch_test_data_index] = scores_test_all[1]
        epoch.data[metricsDic[Metric.mte.name].epoch_test_data_index] = scores_test_all[2]                
              
        epoch.input_output["epoch_test_seq2seq"] = str(scores)
        #logging.info("***Test Score Seq2Seq***")
        logging.info("***Scores: Test Loss={0}, MSE={0}, Test MTE={1}***".format(scores_test_all[0],scores_test_all[1],scores_test_all[2]))
        logging.info(str(scores_test_all))
        logging.info("***End Test Score***")        
            
        epoch.Finished()        
        epoch.SaveToFile(directory, model)           
        all_epochs_Data.append(epoch)       
        
        Visualization.PlotMetric(all_epochs_Data,Metric.loss)
        Visualization.PlotMetric(all_epochs_Data,Metric.mte)
        Visualization.PlotMetric(all_epochs_Data,Metric.mse)
        #####################################
        ###### EPOCH LOOP END
        #####################################


        #print("Test Mortality_Seq2seq_Metric for seq2seq: ")
        #print(MTE(y_test_all,result))
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

    

    

    X_test_all = []
    y_test_all = []
    
    logging.info((model.summary()))

    accuracy = []
    mean_square_error = []
    metricsReports = []

    q = queue.Queue(1)
    #it was 1, for seq2seq has changed to 2
    all_epochs_Data = []

    for e in range(epochs):
        epoch = Epoch(e)
        epoch.Started()
    
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
        #These variables will be used to plot the data for all epochs
        epoch.input_output['all_batches_training_history_loss'] = []
        epoch.input_output['all_batches_training_history_accuracy'] = []
        epoch.input_output['all_batches_training_history_mse'] = []
        epoch.input_output['all_batches_training_history_mte'] = []
        epoch.input_output['all_batches_training_history_mte_test'] = []
        epoch.input_output['all_batches_training_history_mse_test'] = []
    
        for minibatch in range(no_mini_batches):
            #Read samples and prepare data
            minibatch_start_time = datetime.datetime.now()
            logging.info("*********Epoch {2} of {3}, Minibatch {0} of {1} @ {4}********".format(minibatch + 1,no_mini_batches,e + 1,epochs,str(minibatch_start_time.strftime(dateformat))))
            logging.info("Samples:{0}, {3}, MBatch: {1}, Test {2}, Total {4}, MB:{5}".format(totalSamples,minibatchSize,testPercentage,method.name, int(testPercentage * totalSamples),int(testPercentage * minibatchSize)))
      
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

        
        all_epochs_Data.append(epoch)
   
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
            logging.info(all_epochs_Data[ep].data)

            #print([str(met) for met in metricsAll[e]])
        logging.info("***Results***")
    
        if (problem_type == regressionP):
            for ep in range(e + 1):
                logging.info("Epoch {0} MSE:{1}, Loss:{2}".format(ep + 1,mean_square_error[ep][0],accuracy[ep][1]))
        else:
            for ep in range(e + 1):
                logging.info("Epoch {0} Accuracy:{1}, Loss:{2}".format(ep + 1,accuracy[ep][0],accuracy[ep][1]))
        
        logging.info("Parameters: Samples:{0}, Method={3}, Minibatch Size: {1}, Test: Ratio: {2}, Total {4}, MB:{5}".format(totalSamples,minibatchSize,testPercentage, method.name, int(testPercentage * totalSamples),int(testPercentage * minibatchSize)))
        if (problem_type == regressionP):
            logging.info("MSE: %.2f" % (scores[1]))
        else:
            logging.info("Accuracy: %.2f%%" % (scores[1] * 100))
    
        logging.info("***END OF EPOCH {0} of {1}** @ {2} in {3}*".format(e + 1, epochs,datetime.datetime.now().strftime(dateformat),str((datetime.datetime.now() - epoch_start_time))))
        logging.info("*****************************************")
        logging.info("")
    
        if(problem_type != regressionP):
            acc_train = [epoch_iter.input_output['all_batches_training_history_accuracy'][-1][-1] for epoch_iter in all_epochs_Data]
            acc = [epoch_iter.data['accuracy'] for epoch_iter in all_epochs_Data]
        else:
            mse_train = [epoch_iter.input_output['all_batches_training_history_mse'][-1][-1] for epoch_iter in all_epochs_Data]
            mse = [epoch_iter.data['mse'] for epoch_iter in all_epochs_Data]

        ##loss_train =
        ##[epoch_iter.input_output['all_batches_training_history_loss'][-1][-1]
        ##for epoch_iter in all_epochs_Data]
        ##loss = [epoch_iter.data['loss'] for epoch_iter in all_epochs_Data]


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
    
            #### summarize history for loss
            ###plt.plot(loss_train)
            ###plt.plot(loss)
            ###plt.title('model loss')
            ###plt.ylabel('loss')
            ###plt.xlabel('epoch')
            ###plt.legend(['train', 'test'], loc='upper left')
            ####plt.show()
            ###plt.tight_layout()
            ###plt.savefig(directory + '/Loss' + str(e + 1) +
            ###'.pdf',bbox_inches='tight', pad_inches=0)
            ###plt.close()
    
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


                for x in all_epochs_Data:
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
        
        epoch.SaveToFile(directory,model)        


    #if(reg_fall):
    print("Test Accuracy for regression: {0:.2f}".format(accForReg(y_test_all,result)))
    if(detection_mode == 2):
        print("Test TE for regression: ")
        print(TE(y_test_all,result))
    #print(numpy.array(y_test_all).reshape((len(y_test_all),1)))
    #print(numpy.array(result).reshape())
