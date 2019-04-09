# LSTM with dropout for sequence classification in the IMDB dataset
from PredictiveAnalytics.TestCases.MIMICIII.Seq2SeqLSTM.ReadLM import X, Y
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
#top_words = 5000
#(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
#tr=list(range(0,len(X),2))
te=list(numpy.random.randint(0,len(X),size=int((len(X)*0.1))))#Rand 10 percent for testing
tr=set(list(range(0,len(X))))-set(te)#The remaining 90 percent for training
#te=list(range(1,len(X),2))
X_train=[X[i] for i in tr]
y_train=[Y[i] for i in tr]
X_test=[X[i] for i in te]
y_test=[Y[i] for i in te]
#(X_train, y_train)=(X[for i in tr], Y[for i in trr])
#(X_test, y_test) = (X[te], Y[te])
# truncate and pad input sequences
max_review_length = 500
#X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
#X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
#model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(32,8372)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
print("X Shape is:")
print(numpy.array(X_train).shape)
model.fit(X_train, y_train, epochs=100, batch_size=1)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))