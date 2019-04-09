"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template, request
from Flask_HEALPredictionServer2018 import app

from flask import Flask, jsonify
from Flask_HEALPredictionServer2018.PredictiveAnalytics.TestCases.MIMICIII.Seq2SeqLSTM.SeqClassificationPrediction import predict_mortality

@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )

@app.route('/api/v0.1/predict_mortality', methods=['GET', 'POST'])
def mortality():
    if(request.method=='POST'):
        print(len(str(request.data)))
        
        return (str(predict_mortality(request.data)))#['ack']
    else:#'GET'
        return str(predict_mortality())
