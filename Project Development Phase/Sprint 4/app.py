from flask import Flask, render_template, flash, request, session,send_file
from flask import render_template, redirect, url_for, request

import sys

import pickle


import numpy as np

app = Flask(__name__)
app.config['DEBUG']
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

@app.route("/")
def homepage():
    return render_template('home.html')



@app.route("/result", methods=['GET', 'POST'])
def result():
    if request.method == 'POST':

        t1 = request.form['t1']
        t2 = request.form['t2']
        t3 = request.form['t3']

        t4 = request.form['t4']
        t5 = request.form['t5']
        t6 = request.form['t6']
        t7 = request.form['t7']

        t1 = float(t1)
        t2 = float(t2)
        t3 = float(t3)
        t4 = float(t4)
        t5 = float(t5)
        t6 = float(t6)
        t7 = float(t7)



        filename = 'Model/prediction-rfc-model.pkl'
        classifier = pickle.load(open(filename, 'rb'))

        data = np.array([[t1,t2,t3,t4,t5,t6,t7]])

        my_prediction = classifier.predict(data)

        print(my_prediction)



        print(my_prediction[0])

        out = float(my_prediction[0])

        if( out > 0.50):
            res = "you are eligible "
        else:
            res = "you are Not eligible "
        return render_template('home.html',res=res)













if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)