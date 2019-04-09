from flask import Flask, jsonify
from PredictiveAnalytics.TestCases.MIMICIII.Seq2SeqLSTM.SeqClassificationPrediction import predict_mortality

app=Flask(__name__)

@app.route('/')
def index():
    return "Index API"

@app.route('/api/v0.1/predict/<id>')
def predict(id):
    return "Index API"+id

@app.route('/api/v0.1/predict_mortality')
def mortality():
    return str(predict_mortality())

if(__name__=="__main__"):    
    app.run(host='0.0.0.0', debug=False)
    
