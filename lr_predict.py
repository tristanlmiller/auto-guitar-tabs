#!/usr/bin/env python
'''
Predicts chords for an mp3 at the given link, converts to readable format.
Although this is labeled lr_predict, it can in fact be used for any model in the sklearn format.

usage: link model destination [--block block_length --min minfreq --oct num_octaves --bin bins_per_note]

Options:
link - youtube link to video
model - prefix of model filepath (i.e. the output of lr_train)
destination - name of file where results are saved
--block blocklength - Divides chords/mp3s into blocks of time approximately equal to block_length in seconds.
    (Default = 0.5)
--min minfreq - minimum frequency in the Constant-Q Transform (Default = 21.35 Hz, or F0)
--oct num-octaves - number of octaves to include as featuress (Default = 7)
--bin bins_per_note - number of CQT bins per note (of which there are 12 in an octave) (Default = 1)
--transpose - If True, then duplicates and transposes training data (Default = False)
--standard - If True, then saves extra copy of features transposed so root is C.
    Also saves extra copy of train labels so that dimensions match (Default = False)
'''
    
import chord_loader
import pandas as pd
import numpy as np
import sys, os
import sklearn
from sklearn.linear_model import LogisticRegression
import pickle

import requests
import youtube_dl
import traceback

def main():
    
    args = sys.argv[1:]
    args.append('')
    
    if not args:
        print('usage: link model destination [--block block_length --min minfreq --oct num_octaves --bin bins_per_note]')
        sys.exit(1)
    
    #default options:
    block_length = 0.5
    minfreq = 21.35
    num_octaves = 7
    bins_per_note = 1
    model_dir = 'Models/'
    target_dir = 'Results/'
    
    #parse user options
    link = args[0]
    del args[0]
    model_filename = args[0]
    del args[0]
    destination = args[0]
    del args[0]
    if args[0] == '--block':
        block_length = float(args[1])
        del args[0:2]
    if args[0] == '--min':
        minfreq = float(args[1])
        del args[0:2]
    if args[0] == '--oct':
        num_octaves = int(args[1])
        del args[0:2]
    if args[0] == '--bin':
        bins_per_note = int(args[1])
        del args[0:2]
    if args[0] == '--frac':
        fraction = float(args[1])
        del args[0:2]
    predict(link, model_filename, destination, block_length, minfreq, num_octaves, bins_per_note, model_dir, target_dir)

def predict(link, model_filename, destination, block_length, minfreq, num_octaves, bins_per_note, model_dir, target_dir):
    if os.path.exists(f'{target_dir}{destination}.pkl'):
        return None
    #load music
    download_mp3(link,'Data/temp.mp3')
    song_features = chord_loader.get_features("Data/temp.mp3",block_length,minfreq,num_octaves,bins_per_note)
    os.remove('Data/temp.mp3')
        
    #load models from pickles
    with open(f"{model_dir}{model_filename}.pkl", 'rb') as f:
        root_model = pickle.load(f)
        quality_model = pickle.load(f)
        add_model = pickle.load(f)
        inv_model = pickle.load(f)
    
    #predict the root
    root_labels = root_model.predict(song_features)
    song_labels = np.zeros((root_labels.shape[0],4))
    song_labels[:,1:] = np.nan
    song_labels[:,0] = root_labels
    
    #standardize the root
    standard_song_features, standard_song_labels = chord_loader.standardize_root(
        song_features,song_labels,bins_per_note)
    
    #predict the rest of the chord
    quality_labels = quality_model.predict(standard_song_features)
    add_labels = add_model.predict(standard_song_features)
    inv_labels = inv_model.predict(standard_song_features)

    song_labels[~np.equal(song_labels[:,0],-1),1] = quality_labels
    song_labels[~np.equal(song_labels[:,0],-1),2] = add_labels
    song_labels[~np.equal(song_labels[:,0],-1),3] = inv_labels

    chord_symbols = chord_loader.read_chords(song_labels[1:,:])
    
    #save file
    pickle.dump(chord_symbols, open(f'{target_dir}{destination}.pkl', 'wb'))
        
def download_mp3(link,destination):
    options = {
        'format': 'bestaudio/best', # choice of quality
        'extractaudio' : True,      # only keep the audio
        'audioformat' : "mp3",      # convert to mp3 
        'outtmpl': '%(id)s',        # name the file the ID of the video
        'noplaylist' : True,}       # only download single song, not playlist
    ydl = youtube_dl.YoutubeDL(options)

    with ydl:
        # download video
        try:
            result = ydl.extract_info(link, download=True)
            os.rename(result['id'], destination)

        except Exception as e:
            print("Can't download audio! %s\n" % traceback.format_exc())

if __name__ == '__main__':
    main()