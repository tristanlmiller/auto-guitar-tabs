from flask import render_template
from flask import request
from flaskapp import app
#from sqlalchemy import create_engine
#from sqlalchemy_utils import database_exists, create_database
import pandas as pd
#import psycopg2
import sys, os
#import subprocess
import pickle

#sys.path.append('..')
import lr_predict

# Python code to connect to Postgres
# You may need to modify this based on your OS,
# as detailed in the postgres dev setup materials.
#user = 'tristanmiller' #add your Postgres username here
#host = 'localhost'
#dbname = 'birth_db'
#db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
#con = None
#con = psycopg2.connect(database = dbname, user = user, host = host, password = 'mypassword') #add your Postgres password here

@app.route('/')
@app.route('/index')
def index():
    url = request.args.get('youtube_url')
    return render_template("input.html", preset_url=url)

@app.route('/output')
def chord_output():
    url = request.args.get('youtube_url')
    video_code = url[32:]

    model_path = 'xgb_multifinal'
    lr_predict.predict(url, model_path, f"{video_code}", 0.5, 24.5, 7, 1,
                        'Models/', 'Results/', 3)

    results = pickle.load(open(f"Results/{video_code}.pkl", 'rb'))
    results['time'] = ['' if i % 2 == 1 else
        f"{int(i/2/60):.0f}:{(i/2 % 60):02.0f}" for i in range(results.shape[0])]

    nochange = results['full'] == results['full'].shift(1)
    results['root'].loc[nochange] = ''
    results['quality'].loc[nochange] = ''
    results['interval'].loc[nochange] = ''
    results['inv'].loc[nochange] = ''

    return render_template("output.html", predictions = results, video = video_code)

@app.route('/demo')
def chord_demo():
    results = pd.DataFrame()
    #results['time'] = ['']*8

    results['root'] = ['A','B','C','C','C','C','C','C']
    results['quality'] = ['maj','maj','maj','min','dim','maj','maj','maj']
    results['interval'] = ['','','','','','maj7','','']
    results['inv'] = ['','','','','','','/3','/5']
    results['notes'] = ['A3 C#4 E4 A4','B3 D#4 F#4 B4','C3 E4 G4 C4','C3 Eb4 G4 C4',
        'C3 Eb4 Gb4 C4','C3 E4 G4 C4 B5','E3 E4 G4 C4','G3 E4 G4 C4']

    return render_template("demo.html", predictions = results)
