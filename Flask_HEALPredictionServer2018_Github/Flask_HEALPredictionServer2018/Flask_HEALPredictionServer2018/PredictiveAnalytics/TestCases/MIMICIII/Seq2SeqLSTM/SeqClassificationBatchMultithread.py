# LSTM with dropout for sequence classification in the IMDB dataset
script_version='2.1'

from PredictiveAnalytics.TestCases.MIMICIII.Seq2SeqLSTM.ReadBatchZipLM import Y, batch_read, batch_read_thread, reset, method, filename
import numpy
import sys

#USE CPU ONLY
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"#'-1'

from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, f1_score, classification_report, recall_score, brier_score_loss, precision_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from keras import metrics
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.callbacks import TensorBoard
from keras.preprocessing import sequence
import tensorflow
import threading
import  queue
import pickle
import jsonpickle
import logging
import matplotlib.pyplot as plt


class Epoch(object):
    def __init__(self, e, **kwargs):
        self.number=e
        self.data=dict()
        self.input_output=dict()
#sys.stdout=open('log1000.txt','w')
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
#top_words = 5000
#(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
#tr=list(range(0,len(X),2))
'''
te=list(numpy.random.randint(0,len(X),size=int((len(X)*0.1))))#Rand 10 percent for testing
tr=set(list(range(0,len(X))))-set(te)#The remaining 90 percent for training
#te=list(range(1,len(X),2))
X_train=[X[i] for i in tr]
y_train=[Y[i] for i in tr]
X_test=[X[i] for i in te]
y_test=[Y[i] for i in te]
'''
#(X_train, y_train)=(X[for i in tr], Y[for i in trr])
#(X_test, y_test) = (X[te], Y[te])
# truncate and pad input sequences
#max_review_length = 500
#X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
#X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
#embedding_vecor_length = 32
model = Sequential()
#model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(32,8371), name="L1_LSTM"))
model.add(Dense(1, activation='sigmoid', name="Output_Dense"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print((model.summary()))


#totalSamples,minibatchSize,testPercentage, epochs, minibatch_epochs=(40,20,0.2,10,2)
totalSamples,minibatchSize,testPercentage, epochs, minibatch_epochs=(100,10,0.2,10,2)
#totalSamples,minibatchSize,testPercentage,epochs, minibatch_epochs=(1000,100,0.05,20,2)
#totalSamples,minibatchSize,testPercentage,epochs, minibatch_epochs=(10000,400,0.05,200,2)
#totalSamples,minibatchSize,testPercentage,epochs, minibatch_epochs=(34000,500,0.05,200,2)


no_mini_batches=int(totalSamples/minibatchSize)
X_test_all=[]
y_test_all=[]
#Setting up log files/folders
import datetime
dateformatFile='%Y-%m-%d  %H-%M-%S'

directory="Results "+str(datetime.datetime.now().strftime(dateformatFile))+' Samples-{0}_MB-{1}_TestPerc{2}_{3}'.format(totalSamples,minibatchSize, int(testPercentage*100),method)
print(directory)
if not os.path.exists(directory):
    os.makedirs(directory)

#tensorflowLogDir
#if not os.path.exists(directory+'/log/'+directory):
#    os.makedirs(directory)

#An empty file with the name of the input file!
open(directory+'/'+filename.format("Input"),'wb').close()
logFile=open(directory+'/'+'log.txt','wt')




#copyfile(filename.format("Input"), directory)

logging.basicConfig(filename=directory+'/'+'log.txt',format='%(message)s', level=logging.DEBUG)
#Adding log to console as well
consoleHandler=logging.StreamHandler()
consoleHandler.setFormatter(logging.Formatter('*\t%(message)s'))
logging.getLogger().addHandler(consoleHandler)
dateformat="%Y/%m/%d  %H:%M:%S"
logging.info("Script version: "+script_version)
logging.info("Log file created at"+datetime.datetime.now().strftime("%Y/%m/%d  %H:%M:%S"))
logging.info("Directory: {0}".format(directory))
logging.info("Filename Template: {0}".format(filename))
logging.info("Filename Input: {0}".format(filename.format("Input")))
logging.info("Filename Output: {0}".format(filename.format("Output")))

tensorboard = TensorBoard(log_dir=directory, write_graph=True)
#+"/logs/"+directory
#, histogram_freq=0

startTime=datetime.datetime.now()
logging.info('Start time: '+str(startTime))    
logging.info((model.summary()))
accuracy=[]

metricsReports=[]
q=queue.Queue(1)
epochsData=[]
for e in range(epochs):

    epoch_start_time=datetime.datetime.now()
    logging.info("\t##########################################################################")
    logging.info("\t############# Epoch {0} started @ {1} ####################".format(e+1,str(epoch_start_time.strftime(dateformat))))
    logging.info("\t##########################################################################")
    epoch=Epoch(e)
    X_test_all=[]
    y_test_all=[]
    if e==0:#No need for reseting the file if it is not the first minibatch
        #print("Reading Thread Starting (Epoch Thread)")
        reset()
        readingThread=threading.Thread(target=batch_read_thread,args=(q,minibatchSize))
        readingThread.start()

    X2=[]            
    if e>0:#Should not be necessary anymore
        readingThread.join()
            
    batch_data=[]
    epoch.input_output['all_batches_training_history_loss']=[]
    epoch.input_output['all_batches_training_history_accuracy']=[]
    for minibatch in range(no_mini_batches):
        #Read samples and prepare data
        minibatch_start_time=datetime.datetime.now()
        #print("Epoch {2} of {3}, Minibatch {0} of {1}".format(minibatch+1,no_mini_batches,e+1,epochs))
        logging.info("*********Epoch {2} of {3}, Minibatch {0} of {1} @ {4}********".format(minibatch+1,no_mini_batches,e+1,epochs,str(minibatch_start_time.strftime(dateformat))))
        #print("N:{0}, {3}, MBatch: {1}, Test {2}, Total {4}, MB:{5}".format(totalSamples,minibatchSize,testPercentage,method, int(testPercentage*totalSamples),int(testPercentage*minibatchSize)))
        logging.info("Samples:{0}, {3}, MBatch: {1}, Test {2}, Total {4}, MB:{5}".format(totalSamples,minibatchSize,testPercentage,method, int(testPercentage*totalSamples),int(testPercentage*minibatchSize)))
      
        #Starting the thread for reading
        #print('Len X2:')
        #print(len(X2))
        #readingThread=threading.Thread(target=batch_read_thread,args=(q,minibatchSize))
        #readingThread.start()        
        #print("Waiting for read thread to finish")
        readingThread.join()        
        #print("Finished!")                
        X2=q.get()
        #print('Len X2:')
        #print(len(X2))
        #len(X2)
        X=X2
        
        
        #X=batch_read(minibatchSize)
        
        te=list(numpy.random.randint(0,len(X),size=int((len(X)*testPercentage))))#Rand 10 percent for testing
        tr=set(list(range(0,len(X))))-set(te)#The remaining 90 percent for training
        #te=list(range(1,len(X),2))
        offset=minibatchSize*minibatch
        X_train=[X[i] for i in tr]
        y_train=[Y[offset+i] for i in tr]
        X_test=[X[i] for i in te]
        y_test=[Y[offset+i] for i in te]
        X_test_all.extend(X_test)
        y_test_all.extend(y_test)
        #print("X Shape is:")
        #print(numpy.array(X_train).shape)
        if(minibatch==(no_mini_batches-1)):#If last minibatch, reset to read the first batch for next epoch
           reset()
        #if(minibatch<(no_mini_batches-1)):#Read the next batch if not the last minibatch
        #print("Parallel Reading Thread Starting")
        readingThread=threading.Thread(target=batch_read_thread,args=(q,minibatchSize))
        readingThread.start()        
        #validation_data=[X_test, y_test],
        history=model.fit(X_train, y_train, epochs=minibatch_epochs, batch_size=1, callbacks=[tensorboard])
        logging.info("Batch History:")
        #for batchEp in history.epoch:            
         #   logging.info("Training Ep{0}-Accuracy:{1}-Loss:{2}"+str(str(history.history.items()), history.history['loss'][batchEp],history.history['accu'][batchEp]))
        logging.info(str(history.history.items()))
        scores = model.evaluate(X_test, y_test)#, verbose=0)
        epoch.input_output['all_batches_training_history_loss'].append((history.history['loss']))
        epoch.input_output['all_batches_training_history_accuracy'].append(history.history['acc'])        
        
        #metrics.binary_accuracy()
    
        #print("Accuracy: %.2f%%" % (scores[1]*100))
        logging.info("Test Accuracy: %.3f%%" % (scores[1]*100))
        #"Accuracy: %.2f%%" % (scores[1]*100)
        #print(scores)
        #print("Predictions:")
        #x=numpy.array(X_test)
        #x = np.expand_dims(x, axis=0)
        #print(model.predict(X_train,minibatchSize))
        #print("Predictions on batch:")
        #print(model.predict_on_batch(X_test))
        #print("Binary Accuracy:")
        ##print(metrics.binary_accuracy(y_test,round(model.predict(X_test))))
        #import pprint
        #pprint.pprint()
        logging.info("Test Accuracy: %.3f%%" % (scores[1]*100))
        logging.info("**END****Epoch {2} of {3}, Minibatch {0} of {1} in {4} ********".format(minibatch+1,no_mini_batches,e+1,epochs,str((datetime.datetime.now()-minibatch_start_time))))
        logging.info('')
    

    

    # Final evaluation of the model    
    logging.info("For {0} test samples".format(len(y_test_all)))
    logging.info("True Values")
    logging.info(y_test_all)
    logging.info("Predicted Values")
    result=model.predict_on_batch(X_test_all)
    logging.info(model.predict_on_batch(X_test_all))
    logging.info("Comparison on all:")
    result_both=[(float(result[i]),round(float(result[i])), y_test_all[i]) for i in range(len(result))]
    y_score=[round(float(y)) for y in result]
    #Calculate Precision    

    scores = model.evaluate(X_test_all, y_test_all, verbose=0)
    accuracy.append((scores[1]*100, scores[0]))

    #class epoch: e=e
    #import sklearn
    #sklearn.metrics.metrics
    epoch.data['accuracy']=scores[1]
    epoch.data['loss']=scores[0]
    epoch.data['average_precision']=average_precision_score(y_test_all, y_score)
    epoch.data['roc_auc']=roc_auc_score(y_test_all, y_score)
    epoch.input_output['roc_curve']=roc_curve(y_test_all, y_score)
    epoch.data['f1']=f1_score(y_test_all, y_score)
    epoch.data['recall']=recall_score(y_test_all, y_score)
    epoch.data['precision']=precision_score(y_test_all, y_score)
    
   
    
    
    

    #Calculate Brier Score
    if hasattr(model, "predict_proba"):
            prob_pos = model.predict_proba(X_test_all)#[:, 1]
    else:  # use decision function
            prob_pos = model.decision_function(X_test_all)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

    


    epoch.input_output['prob_pos']=prob_pos
    #epoch.input_output['X_test_all']=X_test_all

    epoch.input_output['Y_test_all']=y_test_all
    epoch.input_output['y_predictions']=y_score
    
    clf_score = brier_score_loss(y_test_all, prob_pos, pos_label=1)
    brier=clf_score 

    epoch.data['brier']=clf_score
    #epoch.brier=clf_score

    logging.info('Brier: {0:0.2f}'.format(clf_score))

    

    #print(':')
    epoch.input_output['classification_report']=classification_report(y_test_all, y_score)
    #logging.info(epoch.input_output['classification_report'])#classification_report(y_test_all, y_score))    

    '''
    metricsAll.append((str('Average precision-recall: {0:0.2f}'.format(epoch.average_precision)),str('roc_auc: {0:0.2f}'.format(roc_auc_score(y_test_all, y_score))),
                       str('F1: {0:0.2f}'.format(f1_score(y_test_all, y_score))),
                      str('recall: {0:0.2f}'.format(recall_score(y_test_all, y_score))),
                      print('Brier: {0:0.2f}'.format(clf_score)),
                      #str('roc_curve count: {0}'.format(roc_curve(y_test_all, y_score)))
                      ))#,'classification report :',classification_report(y_test_all, y_score)))
                      '''
    metricsReports.append(epoch.input_output['classification_report'])

    epochsData.append(epoch)
    


    logging.info(result_both)
    logging.info("Predictions on all:")
    
    logging.info("")
    logging.info("*****************************************")
    logging.info("***Epoch Summary***")
    logging.info("Parameters: Samples:{0}, Minibatch Size: {1}, Test ratio: {2} ".format(totalSamples,minibatchSize,testPercentage))
    logging.info(str(epoch.data))
    logging.info('Average precision-recall: {0:0.2f}'.format(epoch.data['average_precision']))
    logging.info('AU-ROC: {0:0.2f}'.format(epoch.data['roc_auc']))
   
    logging.info('F1: {0:0.2f}'.format(epoch.data['f1']))    
    logging.info('Recall: {0:0.2f}'.format(epoch.data['recall']))
    logging.info('Precision: {0:0.2f}'.format(epoch.data['precision']))
    logging.info("Accuracy: %.2f%%" % (scores[1]*100))
    logging.info("Loss: %.2f%%" % (scores[0]))
    logging.info("Brier: %.2f%%" % (brier))
    
    logging.info("*****************************************")
    logging.info("***Classification Reports***")
    for ep in range(e+1):
        logging.info("Epoch {0} accuracy:{1}, loss:{2}".format(ep+1,accuracy[ep][0],accuracy[ep][1]))
        logging.info(metricsReports[ep])        

    #print("***Metrics***")
    logging.info("***Metrics***")
    for ep in range(e+1):
        logging.info("Epoch {0} accuracy:{1}, loss:{2}".format(ep+1,accuracy[ep][0],accuracy[ep][1]))
        
        #print("Epoch {0} metrics".format(ep+1))        
        logging.info(epochsData[ep].data)




        #print([str(met) for met in metricsAll[e]])
    logging.info("***Results***")
    for ep in range(e+1):
        logging.info("Epoch {0} Accuracy:{1}, Loss:{2}".format(ep+1,accuracy[ep][0],accuracy[ep][1]))
    logging.info("Parameters: Samples:{0}, Method={3}, Minibatch Size: {1}, Test: Ratio: {2}, Total {4}, MB:{5}".format(totalSamples,minibatchSize,testPercentage,
                                                                                               method, int(testPercentage*totalSamples),int(testPercentage*minibatchSize)))
    logging.info("Accuracy: %.2f%%" % (scores[1]*100))
    logging.info("***END OF EPOCH {0} of {1}** @ {2} in {3}*".format(e+1, epochs,datetime.datetime.now().strftime(dateformat),str((datetime.datetime.now()-epoch_start_time))))
    logging.info("*****************************************")
    logging.info("")


            
    acc_train=[epoch_iter.input_output['all_batches_training_history_accuracy'][-1][-1] for epoch_iter in epochsData]
    loss_train=[epoch_iter.input_output['all_batches_training_history_loss'][-1][-1] for epoch_iter in epochsData]

    acc=[epoch_iter.data['accuracy'] for epoch_iter in epochsData]
    loss=[epoch_iter.data['loss'] for epoch_iter in epochsData]

    plt.plot(acc_train)
    plt.plot(acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.tight_layout()
    plt.savefig(directory+'/Accuracy'+str(e+1)+'.pdf',bbox_inches='tight', pad_inches=0)
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
    plt.savefig(directory+'/Loss'+str(e+1)+'.pdf',bbox_inches='tight', pad_inches=0)
    plt.close()


    

    #Plot Brier

    fig = plt.figure(1, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test_all, prob_pos, n_bins=10)

    epoch.input_output['fraction_of_positives']=fraction_of_positives
    epoch.input_output['mean_predicted_value']=mean_predicted_value 
    

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
    plt.savefig(directory+'/Calibration Epoch '+str(e+1)+'.pdf',bbox_inches='tight', pad_inches=0)
    plt.close()

    #plt.show()
    
    #Plot Brier for all

        #Plot Brier
    

    fig = plt.figure(1, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test_all, prob_pos, n_bins=10)

    epoch.input_output['fraction_of_positives']=fraction_of_positives
    epoch.input_output['mean_predicted_value']=mean_predicted_value 
    

    for x in epochsData:
        ax1.plot(x.input_output['mean_predicted_value'], x.input_output['fraction_of_positives'], "s-",
                label="%s (%1.3f)" % ('Epoch{0}'.format(str(x.number+1)), x.data['brier']))

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
    plt.savefig(directory+'/Calibration Epoch All '+str(e+1)+'.pdf',bbox_inches='tight', pad_inches=0)
    plt.close()


    #Area Under the Curve
    fpr, tpr, _= epoch.input_output['roc_curve']


    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % epoch.data['roc_auc'])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(directory+'/AUC Epoch'+str(e+1)+'.pdf',bbox_inches='tight', pad_inches=0)
    plt.close()

    
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




    pickleFile=open(directory+'/'+'data{0}.bin'.format(e+1),'wb')
    jsonFile=open(directory+'/'+'data{0}.txt'.format(e+1),'wt')
    epochFile=open(directory+'/'+'epoch{0}.txt'.format(e+1),'wt')
    
    model.save(directory+'/'+'model_keras{0}.h5'.format(e+1))

    



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

    

#print("***All Summary***")
#print("Parameters: Samples:{0}, Minibatch Size: {1}, Test ratio: {2} ".format(totalSamples,minibatchSize,testPercentage))

#for e in range(epochs):
 #   print("Epoch {0} accuracy:{1}, loss:{2}".format(e,accuracy[e][0],accuracy[e][1]))