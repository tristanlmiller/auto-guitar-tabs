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
    url = request.args.get('youtube_url')
    return render_template("input.html", preset_url=url)

@app.route('/output')
def chord_output():
    url = request.args.get('youtube_url')
    video_code = url[32:]

    lr_predict.predict(url, 'default_10', f"{video_code}", 0.5, 21.35, 7, 1,
                        'Models/', 'Results/')

    results = pickle.load(open(f"Results/{video_code}.pkl", 'rb'))
    results['time'] = ['' if i % 2 == 1 else
        f"{int(i/2/60):.0f}:{(i/2 % 60):02.0f}" for i in range(results.shape[0])]

    nochange = results['full'] == results['full'].shift(1)
    results['root'].loc[nochange] = ''
    results['quality'].loc[nochange] = ''
    results['interval'].loc[nochange] = ''
    results['inv'].loc[nochange] = ''

    return render_template("output.html", predictions = results, video = video_code)
