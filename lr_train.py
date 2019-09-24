#!/usr/bin/env python
'''
Trains a logistic regression model on processed data, saves model as pickle.

usage: source destination [--weighted --C L2weight --frac fraction]

Options:
source - prefix of data files used for this model (same as what was provided to chord_process.py)
destination - prefix of destination file names
Files will be saved with the following suffixes:
    _root.pkl - model of root
    _quality.pkl - model of quality
    _add.pkl - model of add
    _inv.pkl - model of inversion
--weighted - if option is selected, then logistic regression is performed with balanced weighting
--C L2weight - regularization strength input into logistic regression model
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
    L2weight = 1.0
    
    #parse user options
    source = args[0]
    del args[0]
    destination = args[0]
    del args[0]
    if args[0] == '--weighted':
        weight = 'balanced'
        del args[0:1]
    if args[0] == '--C':
        L2weight = float(args[1])
        del args[0:2]
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
    root_model = LogisticRegression(class_weight=weight,multi_class='ovr',C=L2_weight,
                                                solver='lbfgs', max_iter=1000)
    quality_model = LogisticRegression(class_weight=weight,multi_class='ovr',C=L2_weight,
                                                solver='lbfgs', max_iter=1000)
    add_model = LogisticRegression(class_weight=weight,multi_class='ovr',C=L2_weight,
                                                solver='lbfgs', max_iter=1000)
    inv_model = LogisticRegression(class_weight=weight,multi_class='ovr',C=L2_weight,
                                                solver='lbfgs', max_iter=1000)
    
    #load data
    features_train = np.load(f'{source_dir}{source}_ftrain.npy')
    labels_train = np.load(f'{source_dir}{source}_ltrain.npy')
    features_valid = np.load(f'{source_dir}{source}_fvalid.npy')
    labels_valid = np.load(f'{source_dir}{source}_lvalid.npy')
    features_test = np.load(f'{source_dir}{source}_ftest.npy')
    labels_test = np.load(f'{source_dir}{source}_ltest.npy')
    if curr_data_info['standard']:
        standard_features_train = np.load(f'{source_dir}{source}_fstrain.npy')
        standard_labels_train = np.load(f'{source_dir}{source}_lstrain.npy')
        standard_features_valid = np.load(f'{source_dir}{source}_fsvalid.npy')
        standard_labels_valid = np.load(f'{source_dir}{source}_lsvalid.npy')
        standard_features_test = np.load(f'{source_dir}{source}_fstest.npy')
        standard_labels_test = np.load(f'{source_dir}{source}_lstest.npy')
        
    #select fraction of training songs
    if fraction < 1:
        if curr_data_info['transpose']:
            kept_rows = np.tile(np.arange(labels_train.shape[0]/12) <= labels_train.shape[0]*fraction/12,12)
        else:
            kept_rows = np.arange(labels_train.shape[0]) <= labels_train.shape[0]*fraction
        features_train = features_train[kept_rows,:]
        labels_train = labels_train[kept_rows,:]
        if curr_data_info['standard']:
            kept_rows = np.arange(standard_labels_train.shape[0]) <= standard_labels_train.shape[0]*fraction
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
    
    #save models
    with open(f'{target_dir}{destination}.pkl', 'wb') as f:
        pickle.dump(root_model, f)
        pickle.dump(quality_model, f)
        pickle.dump(add_model, f)
        pickle.dump(inv_model, f)
    
    #generate directory file if it doesn't already exist.
    if not os.path.exists(target_dir + 'lr_directory.csv'):
        header = 'sourcepath,filepath,weighted,fraction,L2weight'
        with open(target_dir + 'lr_directory.csv','a') as f:
            f.write(header)
    
    #record settings in file
    newrow = f"\n{source},{destination},{weight},{fraction},{L2weight}"
    with open(target_dir + 'lr_directory.csv','a') as f:
        f.write(newrow)
        
    #Next, get metrics

if __name__ == '__main__':
    main()