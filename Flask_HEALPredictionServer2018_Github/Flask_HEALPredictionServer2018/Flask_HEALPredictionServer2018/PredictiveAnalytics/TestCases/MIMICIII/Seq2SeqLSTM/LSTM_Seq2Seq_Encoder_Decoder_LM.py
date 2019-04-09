from PredictiveAnalytics.TestCases.MIMICIII.Seq2SeqLSTM.ReadLM import X, Y
from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy

# generate a sequence of random integers
def generate_sequence(length, n_unique):
	return [randint(1, n_unique-1) for _ in range(length)]

# prepare data for the LSTM
def get_dataset(n_in, n_out, cardinality, n_samples):
 X1, X2, y = list(), list(), list()
 
 for i in range(n_samples):
    # generate source sequence
    #source = generate_sequence(n_in, cardinality)
    source = X[i]
    # define padded target sequence
    target = Y[i] #source[:n_out] #Or make it a list
    #target.reverse()
    # create padded input target sequence
    target_in = Y[i]#[0] + target[:-1]
    # encode
    #src_encoded = to_categorical([source], num_classes=cardinality)
    tar_encoded = to_categorical([target], num_classes=2)#cardinality)
    tar2_encoded = to_categorical([target_in], num_classes=2)#cardinality)
    # store
    X1.append(source) #src_encoded)

    #X1ArraySource=array(source)
    #print("source Array shape (each sample) is: {s}".format(s=X1ArraySource.shape))
    
    #print("X1 size (each sample) is: {s}".format(s=len(X[i][0])))
    
    #X1Array=array(X1)
    #print("X1 Array shape (each sample) is: {s}".format(s=X1Array.shape))
    

    #X2.append(target_in) #tar2_encoded)
    #y.append(target)#tar_encoded)
    X2.append(tar2_encoded)
    y.append(tar_encoded)
 return array(X1), array(X2), array(y)

# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, n_units):
	# define training encoder
	encoder_inputs = Input(shape=(None, n_input))
	encoder = LSTM(n_units, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	encoder_states = [state_h, state_c]
	# define training decoder
	decoder_inputs = Input(shape=(None, n_output))
	decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	decoder_dense = Dense(n_output, activation='softmax')
	decoder_outputs = decoder_dense(decoder_outputs)
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	# define inference encoder
	encoder_model = Model(encoder_inputs, encoder_states)
	# define inference decoder
	decoder_state_input_h = Input(shape=(n_units,))
	decoder_state_input_c = Input(shape=(n_units,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
	# return all models
	return model, encoder_model, decoder_model

# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
	# encode
	state = infenc.predict(source)
	# start of sequence input
	target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
	# collect predictions
	output = list()
	for t in range(n_steps):
		# predict next char
		yhat, h, c = infdec.predict([target_seq] + state)
		# store prediction
		output.append(yhat[0,0,:])
		# update state
		state = [h, c]
		# update target sequence
		target_seq = yhat
	return array(output)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

# configure problem
n_features = 8372 + 1
n_steps_in = 32
n_steps_out = 1
# define model
train, infenc, infdec = define_models(n_features, n_features, 100)
train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# generate training dataset
X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 100)
print(X1.shape,X2.shape,y.shape)
# train model

#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
tr=range(0,100,2)
te=range(1,100,2)
hist=train.fit([X1[tr], X2[tr]], y[tr], epochs=100,validation_split=0.20)
print("Train:")
score=train.evaluate([X1[tr], X2[tr]], y[tr])
print(score)
print("Test:")
score=train.evaluate([X1[te], X2[te]], y[te])
print(hist.history)

#import csv
#with open('epochs.csv', 'wt') as csvfile:
 #       w=csv.DictWriter(csvfile,hist.history.keys())
  #      w.writeheader()
   #     w.writerow(hist.history)


# list all data in history
print(hist.history.keys())
# summarize history for accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#print(score)
#train.fit(X, Y, epochs=1)

'''
# evaluate LSTM
total, correct = 100, 0
for _ in range(total):
	X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
	target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
	if array_equal(one_hot_decode(y[0]), one_hot_decode(target)):
		correct += 1
print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))
'''
print(score)
# spot check some examples
#for _ in range(10):
#	X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
#	target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
#	print('X=%s y=%s, yhat=%s' % (one_hot_decode(X1[0]), one_hot_decode(y[0]), one_hot_decode(target)))