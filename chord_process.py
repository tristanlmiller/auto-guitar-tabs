#!/usr/bin/env python
'''
Processes all chords and mp3s and saves them as npy files.

usage: destination [--block block_length --min minfreq --oct num_octaves --bin bins_per_note --transpose --standard --frac fraction']

Options:
destination - prefix for the names of destination files
Files will be saved with the following suffixes:
    _ftrain.npy - training set features
    _ltrain.npy - training set labels
    _fvalid.npy - validation set features
    _lvalid.npy - validation set labels
    _ftest.npy - test set features
    _ltest.npy - test set labels
If standard option is used, also saves files with these suffixes:
    _fstrain.npy - standardized training set features
    _lstrain.npy - standardized training set labels
    _fsvalid.npy - standardized validation set features
    _lsvalid.npy - standardized validation set labels
    _fstest.npy - standardized test set features
    _lstest.npy - standardized test set labels
--block blocklength - Divides chords/mp3s into blocks of time approximately equal to block_length in seconds.
    (Default = 0.5)
--min minfreq - minimum frequency in the Constant-Q Transform (Default = 21.35 Hz, or F0)
--oct num-octaves - number of octaves to include as featuress (Default = 7)
--bin bins_per_note - number of CQT bins per note (of which there are 12 in an octave) (Default = 1)
--transpose - If True, then duplicates and transposes training data (Default = False)
--standard - If True, then saves extra copy of features transposed so root is C.
    Also saves extra copy of train labels so that dimensions match (Default = False)
--fraction - Use only on fraction of data (Default = 1)
'''
    
import chord_loader
import pandas as pd
import numpy as np
import sys, os
import sklearn
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def main():
    
    args = sys.argv[1:]
    args.append('')
    
    if not args:
        print('usage: destination [--block block_length --min minfreq --oct num_octaves --bin bins_per_note --transpose --standard] --frac fraction')
        sys.exit(1)
    
    #default options:
    block_length = 0.5
    minfreq = 21.35
    num_octaves = 7
    bins_per_note = 1
    transpose = False
    standard = False
    fraction = 1
    target_dir = 'Data/processed/'
    
    #parse user options
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
    if args[0] == '--transpose':
        transpose = True
        del args[0]
    if args[0] == '--standard':
        standard = True
        del args[0]
    if args[0] == '--frac':
        fraction = float(args[1])
        del args[0:2]
    
    #load directory and split into train, validation, and test sets
    print('Performing test/training split')
    song_directory = pd.read_csv('song_directory.csv')
    valid_directory = song_directory.loc[~pd.isna(song_directory['mp3_filepath'])]
    np.random.seed(35402374)
    if fraction < 1:
        valid_directory, temp_set = train_test_split(valid_directory,shuffle=True,train_size = fraction)
    temp_set, test_set = train_test_split(valid_directory,shuffle=True,train_size = 0.8)
    train_set, valid_set = train_test_split(temp_set,shuffle=True,train_size = 0.75)
    
    #process data
    print('Processing test data')
    features_test, labels_test = chord_loader.get_features_labels(test_set,block_length,minfreq,
                                                                    num_octaves,bins_per_note,False)
    np.save(f'{target_dir}{destination}_ftest.npy',features_test)
    np.save(f'{target_dir}{destination}_ltest.npy',labels_test)
    
    if standard:
        standard_features_test, standard_labels_test = chord_loader.standardize_root(features_test,labels_test,
                                                                                       bins_per_note)
        np.save(f'{target_dir}{destination}_fstest.npy',standard_features_test)
        np.save(f'{target_dir}{destination}_lstest.npy',standard_labels_test)
    
    print('Processing validation data')
    features_valid, labels_valid = chord_loader.get_features_labels(valid_set,block_length,minfreq,
                                                                    num_octaves,bins_per_note,False)
    np.save(f'{target_dir}{destination}_fvalid.npy',features_valid)
    np.save(f'{target_dir}{destination}_lvalid.npy',labels_valid)
    if standard:
        standard_features_valid, standard_labels_valid = chord_loader.standardize_root(features_valid,labels_valid,
                                                                                       bins_per_note)
        np.save(f'{target_dir}{destination}_fsvalid.npy',standard_features_valid)
        np.save(f'{target_dir}{destination}_lsvalid.npy',standard_labels_valid)
    
    print('Processing training data')
    features_train, labels_train = chord_loader.get_features_labels(train_set,block_length,minfreq,
                                                                    num_octaves,bins_per_note,transpose)
    np.save(f'{target_dir}{destination}_ftrain.npy',features_train)
    np.save(f'{target_dir}{destination}_ltrain.npy',labels_train)
    if standard:
        standard_features_train, standard_labels_train = chord_loader.standardize_root(features_train,labels_train,
                                                                                           bins_per_note,transposed=transpose)
        np.save(f'{target_dir}{destination}_fstrain.npy',standard_features_train)
        np.save(f'{target_dir}{destination}_lstrain.npy',standard_labels_train)
    
    #generate directory file if it doesn't already exist.
    print('Recording settings')
    if not os.path.exists(target_dir + 'directory.csv'):
        header = 'filepath,block_length,minfreq,num_octaves,bins_per_note,transpose,standard,fraction'
        with open(target_dir + 'directory.csv','a') as f:
            f.write(header)
    
    #record settings in file
    newrow = f"\n{destination},{block_length},{minfreq},{num_octaves},{bins_per_note},{transpose},{standard},{fraction}"
    with open(target_dir + 'directory.csv','a') as f:
        f.write(newrow)

if __name__ == '__main__':
    main()