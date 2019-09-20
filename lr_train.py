#!/usr/bin/env python
'''
Trains a logistic regression model on processed data, saves model as pickle.

usage: source destination [--weighted --frac fraction]

Options:
source - prefix of data files used for this model (same as what was provided to chord_process.py)
destination - prefix of destination file names
Files will be saved with the following suffixes:
    _root.pkl - model of root
    _quality.pkl - model of quality
    _add.pkl - model of add
    _inv.pkl - model of inversion
--weighted - if option is selected, then logistic regression is performed with weighting
--frac - fraction of training data to use, e.g. if you're making learning curves (Default = 1)
'''
    
import chord_loader
import pandas as pd
import numpy as np
import sys, os
import sklearn
from sklearn.linear_model import LogisticRegression
import pickle

def main():
    
    args = sys.argv[1:]
    args.append('')
    
    if not args:
        print('usage: source destination [--weighted --frac fraction]')
        sys.exit(1)
    
    #default options:
    weight = None
    fraction = 1
    source_dir = 'Data/processed/'
    target_dir = 'Models/'
    
    #parse user options
    source = args[0]
    del args[0]
    destination = args[0]
    del args[0]
    if args[0] == '--weighted':
        weight = 'balanced'
        del args[0:1]
    if args[0] == '--frac':
        fraction = float(args[1])
        if fraction < 0 or fraction > 1:
            print('fraction must be between 0 and 1')
            sys.exit(1)
        del args[0:2]
        
    #get information from processed data directory
    data_info = pd.read_csv(source_dir + 'directory.csv')
    curr_data_info = data_info.loc[data_info['filepath']==source,:]
    if curr_data_info.shape[0] < 1:
        print('Source not found in directory')
        sys.exit(1)
    curr_data_info = curr_data_info.iloc[-1,:]
    
    #create logistic regression models
    root_model = LogisticRegression(class_weight=weight,multi_class='ovr',
                                                solver='lbfgs', max_iter=200)
    quality_model = LogisticRegression(class_weight=weight,multi_class='ovr',
                                                solver='lbfgs', max_iter=200)
    add_model = LogisticRegression(class_weight=weight,multi_class='ovr',
                                                solver='lbfgs', max_iter=200)
    inv_model = LogisticRegression(class_weight=weight,multi_class='ovr',
                                                solver='lbfgs', max_iter=200)
    
    #load training data
    features_train = np.load(f'Data/processed/{source}_ftrain.npy')
    labels_train = np.load(f'Data/processed/{source}_ltrain.npy')
    if curr_data_info['standard']:
        standard_features_train = np.load(f'Data/processed/{source}_fstrain.npy')
        standard_labels_train = np.load(f'Data/processed/{source}_lstrain.npy')
        
    #select fraction of training songs
    if fraction < 1:
        if curr_data_info['transpose']:
            kept_rows = np.tile(np.arange(labels_train.shape[0]) <= labels_train.shape[0]*fraction,12)
        else:
            kept_rows = np.arange(labels_train.shape[0]) <= labels_train.shape[0]*fraction
        features_train = features_train[kept_rows,:]
        labels_train = labels_train[kept_rows,:]
        if curr_data_info['standard']:
            standard_features_train = standard_features_train[kept_rows,:]
            standard_labels_train = standard_labels_train[kept_rows,:]
    
    #Train models
    root_model.fit(features_train, labels_train[:,0])
    if curr_data_info['standard']:
        quality_model.fit(standard_features_train, standard_labels_train[:,1])
        add_model.fit(standard_features_train, standard_labels_train[:,2])
        inv_model.fit(standard_features_train, standard_labels_train[:,3])
    else:
        quality_model.fit(features_train, labels_train[:,1])
        add_model.fit(features_train, labels_train[:,2])
        inv_model.fit(features_train, labels_train[:,3])
    
    #save files
    pickle.dump(root_model, open(f'{target_dir}{destination}_root.pkl', 'wb'))
    pickle.dump(quality_model, open(f'{target_dir}{destination}_quality.pkl', 'wb'))
    pickle.dump(add_model, open(f'{target_dir}{destination}_add.pkl', 'wb'))
    pickle.dump(inv_model, open(f'{target_dir}{destination}_inv.pkl', 'wb'))
    
    #generate directory file if it doesn't already exist.
    if not os.path.exists(target_dir + 'lr_directory.csv'):
        header = 'sourcepath,filepath,weighted,fraction'
        with open(target_dir + 'directory.csv','a') as f:
            f.write(header)
    
    #record settings in file
    newrow = f"\n{source},{destination},{weight},{fraction}"
    with open(target_dir + 'lr_directory.csv','a') as f:
        f.write(newrow)

if __name__ == '__main__':
    main()