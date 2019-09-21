from flask import render_template
from flask import request
from flaskapp import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
import sys, os
import subprocess
import pickle

#sys.path.append('..')
import lr_predict

# Python code to connect to Postgres
# You may need to modify this based on your OS,
# as detailed in the postgres dev setup materials.
user = 'tristanmiller' #add your Postgres username here
host = 'localhost'
dbname = 'birth_db'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user, host = host, password = 'mypassword') #add your Postgres password here

@app.route('/')
@app.route('/index')
def index():
    return render_template("input.html")

@app.route('/output')
def chord_output():
    url = request.args.get('youtube_url')
    video_code = url[32:]

    lr_predict.predict(url, 'default_10', f"{video_code}", 0.5, 21.35, 7, 1,
                        'Models/', 'Results/')

    results = pickle.load(open(f"Results/{video_code}.pkl", 'rb'))

    predictions = []
    last_chord = '~'
    for i,result in enumerate(results):
        if result != last_chord:
            last_chord = result
        else:
            result = ''
        if i % 2 == 0:
            predictions.append(dict(time=f"{int(i/2/60):.0f}:{(i/2 % 60):02.0f}", chord=result))
        else:
            predictions.append(dict(time="", chord=result))
        #if i >= 5:
            #break
    return render_template("output.html", predictions = predictions, video = video_code)
