# LSTM with dropout for sequence classification in the IMDB dataset
from PredictiveAnalytics.TestCases.MIMICIII.Seq2SeqLSTM.ReadBatchZipLM import Y, batch_read, reset
import numpy
import sys
from keras import metrics
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

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
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(32,8371)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

totalSamples,minibatchSize,testPercentage=(40,20,0.2)
#totalSamples,minibatchSize,testPercentage=(1000,100,0.05)
#totalSamples,minibatchSize,testPercentage=(10000,400,0.05)


no_mini_batches=int(totalSamples/minibatchSize)
X_test_all=[]
y_test_all=[]

epochs=10;
minibatch_epochs=2
accuracy=[]
for e in range(epochs):
    X_test_all=[]
    y_test_all=[]
    reset()
    for minibatch in range(no_mini_batches):
        #Read samples and prepare data
        print("Epoch {2} of {3}, Minibatch {0} of {1}".format(minibatch+1,no_mini_batches,e+1,epochs))
        X=batch_read(minibatchSize)
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
        model.fit(X_train, y_train, epochs=minibatch_epochs, batch_size=1)
        scores = model.evaluate(X_test, y_test)#, verbose=0)
        #metrics.binary_accuracy()
    
        print("Accuracy: %.2f%%" % (scores[1]*100))
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





    

    # Final evaluation of the model    
    print("For {0} test samples".format(len(y_test_all)))
    print("True Values")
    print(y_test_all)
    print("Predicted Values")
    result=model.predict_on_batch(X_test_all)
    print(model.predict_on_batch(X_test_all))
    print("Comparison on all:")
    result_both=[(float(result[i]),round(float(result[i])), y_test_all[i]) for i in range(len(result))]
    y_score=[round(float(y)) for y in result]
    #Calculate Precision
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import roc_auc_score, roc_curve, f1_score, classification_report, recall_score
    average_precision = average_precision_score(y_test_all, y_score)

    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    print('roc_auc_score: {0:0.2f}'.format(roc_auc_score(y_test_all, y_score)))
    print('roc_curve count: {0}'.format(roc_curve(y_test_all, y_score)))
    print('F1 score: {0:0.2f}'.format(f1_score(y_test_all, y_score)))    
    print('recall score: {0:0.2f}'.format(recall_score(y_test_all, y_score)))
    print('classification report :')
    print(classification_report(y_test_all, y_score))

    #Calculate Recall


    #Calculate F1 Score



    print(result_both)
    print("Predictions on all:")
    scores = model.evaluate(X_test_all, y_test_all, verbose=0)
    
    print("")
    print("*****************************************")
    print("***Epoch Summary***")
    print("Parameters: Samples:{0}, Minibatch Size: {1}, Test ratio: {2} ".format(totalSamples,minibatchSize,testPercentage))
    print("Accuracy: %.2f%%" % (scores[1]*100))
    accuracy.append((scores[1]*100, scores[0]))
    for ep in range(e+1):
        print("Epoch {0} accuracy:{1}, loss:{2}".format(ep+1,accuracy[ep][0],accuracy[ep][1]))
    print("***END OF EPOCH {0} of {1}***".format(e+1, epochs))
    print("*****************************************")
    print("")
    

print("***All Summary***")
print("Parameters: Samples:{0}, Minibatch Size: {1}, Test ratio: {2} ".format(totalSamples,minibatchSize,testPercentage))

for e in range(epochs):
    print("Epoch {0} accuracy:{1}, loss:{2}".format(e,accuracy[e][0],accuracy[e][1]))