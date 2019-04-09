import keras

from Flask_HEALPredictionServer2018.PredictiveAnalytics.TestCases.MIMICIII.Seq2SeqLSTM.ReadBatchZipLM import batch_read
import numpy

model=keras.models.load_model('model_keras.h5')

def predict_mortality(X_string='empty'):
    if(X_string=='empty'):
        X=batch_read(1)
    else:
        lines=list(map(lambda x: float(x),X_string[:-1]))    
        X=[]
        F=8371
        K=32
        X.append([lines[i*F:(i +1)*F] for i in range(K)])
    result=model.predict(numpy.array(X))
    return result


