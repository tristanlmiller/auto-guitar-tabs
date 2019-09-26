#!/usr/bin/env python
'''
Trains a model on processed data, saves model as pickle.

usage: source destination model [--frac fraction --metrics --params param1 param2 etc. ]

Options:
source - prefix of data files used for this model (same as what was provided to chord_process.py)
destination - prefix of destination file names
--frac fraction - fraction of training data to use, e.g. if you're making learning curves (Default = 1)
--metrics - calculates metrics only using pre-existing model
--params param1 param2 etc. - set of parameters that depends on the model type.  See model type for more info.
'''
    
import chord_loader
import pandas as pd
import numpy as np
import sys, os
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle

def main():
    
    args = sys.argv[1:]
    args.append('')
    
    if not args:
        print('usage: source destination model [--frac fraction --metrics --params param1 param2 etc. ]')
        sys.exit(1)
    
    #default options:
    weight = None
    fraction = 1
    source_dir = 'Data/processed/'
    target_dir = 'Models/'
    metrics_only = False
    params = []
    
    #parse user options
    source = args[0]
    del args[0]
    destination = args[0]
    del args[0]
    model = args[0]
    del args[0]
    if args[0] == '--frac':
        fraction = float(args[1])
        if fraction < 0 or fraction > 1:
            print('fraction must be between 0 and 1')
            sys.exit(1)
        del args[0:2]
    if args[0] == '--metrics':
        metrics_only = True
        del args[0:1]
    if args[0] == '--params':
        for i in range(1,len(args)-1):
            params.append(args[i])
    prepare_train(model, source, destination, source_dir, target_dir, fraction, metrics_only, params)
    
def prepare_train(model, source, destination, source_dir, target_dir, fraction, metrics_only, params):
    '''Prepare data for training, and then train it'''

    #load data
    features_train = np.load(f'{source_dir}{source}_ftrain.npy')
    labels_train = np.load(f'{source_dir}{source}_ltrain.npy')
    features_valid = np.load(f'{source_dir}{source}_fvalid.npy')
    labels_valid = np.load(f'{source_dir}{source}_lvalid.npy')
    features_test = np.load(f'{source_dir}{source}_ftest.npy')
    labels_test = np.load(f'{source_dir}{source}_ltest.npy')
    standard_features_train = np.load(f'{source_dir}{source}_fstrain.npy')
    standard_labels_train = np.load(f'{source_dir}{source}_lstrain.npy')
    standard_features_valid = np.load(f'{source_dir}{source}_fsvalid.npy')
    standard_labels_valid = np.load(f'{source_dir}{source}_lsvalid.npy')
    standard_features_test = np.load(f'{source_dir}{source}_fstest.npy')
    standard_labels_test = np.load(f'{source_dir}{source}_lstest.npy')
    
    #select fraction of training songs
    if fraction < 1:
        kept_rows = np.tile(np.arange(labels_train.shape[0]/12) <= labels_train.shape[0]*fraction/12,12)
        features_train = features_train[kept_rows,:]
        labels_train = labels_train[kept_rows,:]
        kept_rows = np.arange(standard_labels_train.shape[0]) <= standard_labels_train.shape[0]*fraction
        standard_features_train = standard_features_train[kept_rows,:]
        standard_labels_train = standard_labels_train[kept_rows,:]

    if not metrics_only:
        #create and train models
        root_model, quality_model, add_model, inv_model = train(model, params,
            features_train,
            labels_train,
            features_valid,
            labels_valid,
            features_test,
            labels_test,
            standard_features_train,
            standard_labels_train,
            standard_features_valid,
            standard_labels_valid,
            standard_features_test,
            standard_labels_test)
        
        #save models
        with open(f'{target_dir}{destination}.pkl', 'wb') as f:
            pickle.dump(root_model, f)
            pickle.dump(quality_model, f)
            pickle.dump(add_model, f)
            pickle.dump(inv_model, f)
        
        #generate info file
        info = pd.DataFrame(columns=['sourcepath','filepath','model','params'])
        info['sourcepath'] = [source]
        info['filepath'] = [destination]
        info['model'] = [model]
        info['params'] = [params]
        info.to_csv(f'{target_dir}{destination}_info.csv',index=False)
        
    else:
        with open(f'{target_dir}{destination}.pkl', 'wb') as f:
            root_model = pickle.load(f)
            quality_model = pickle.load(f)
            add_model = pickle.load(f)
            inv_model = pickle.load(f)
        
    #Get F1, accuracy, and confusion_matrix metrics
    metrics = {}
    root_predict_train = root_model.predict(features_train)
    root_predict_valid = root_model.predict(features_valid)
    root_predict_test = root_model.predict(features_test)
    metrics['root_acc_train'] = sklearn.metrics.accuracy_score(labels_train[:,0],root_predict_train)
    metrics['root_acc_valid'] = sklearn.metrics.accuracy_score(labels_valid[:,0],root_predict_valid)
    metrics['root_acc_test'] = sklearn.metrics.accuracy_score(labels_test[:,0],root_predict_test)
    metrics['root_cmat_train'] = sklearn.metrics.confusion_matrix(labels_train[:,0],root_predict_train,labels=range(-1,12))
    metrics['root_cmat_valid'] = sklearn.metrics.confusion_matrix(labels_valid[:,0],root_predict_valid,labels=range(-1,12))
    metrics['root_cmat_test'] = sklearn.metrics.confusion_matrix(labels_test[:,0],root_predict_test,labels=range(-1,12))
    metrics['root_F1_train'] = sklearn.metrics.f1_score(labels_train[:,0],root_predict_train,average='weighted')
    metrics['root_F1_valid'] = sklearn.metrics.f1_score(labels_valid[:,0],root_predict_valid,average='weighted')
    metrics['root_F1_test'] = sklearn.metrics.f1_score(labels_test[:,0],root_predict_test,average='weighted')
    
    quality_predict_train = quality_model.predict(standard_features_train)
    quality_predict_valid = quality_model.predict(standard_features_valid)
    quality_predict_test = quality_model.predict(standard_features_test)
    metrics['quality_acc_train'] = sklearn.metrics.accuracy_score(standard_labels_train[:,1],quality_predict_train)
    metrics['quality_acc_valid'] = sklearn.metrics.accuracy_score(standard_labels_valid[:,1],quality_predict_valid)
    metrics['quality_acc_test'] = sklearn.metrics.accuracy_score(standard_labels_test[:,1],quality_predict_test)
    metrics['quality_cmat_train'] = sklearn.metrics.confusion_matrix(standard_labels_train[:,1],quality_predict_train,labels=range(8))
    metrics['quality_cmat_valid'] = sklearn.metrics.confusion_matrix(standard_labels_valid[:,1],quality_predict_valid,labels=range(8))
    metrics['quality_cmat_test'] = sklearn.metrics.confusion_matrix(standard_labels_test[:,1],quality_predict_test,labels=range(8))
    metrics['quality_F1_train'] = sklearn.metrics.f1_score(standard_labels_train[:,1],quality_predict_train,average='weighted')
    metrics['quality_F1_valid'] = sklearn.metrics.f1_score(standard_labels_valid[:,1],quality_predict_valid,average='weighted')
    metrics['quality_F1_test'] = sklearn.metrics.f1_score(standard_labels_test[:,1],quality_predict_test,average='weighted')
    
    add_predict_train = add_model.predict(standard_features_train)
    add_predict_valid = add_model.predict(standard_features_valid)
    add_predict_test = add_model.predict(standard_features_test)
    metrics['add_acc_train'] = sklearn.metrics.accuracy_score(standard_labels_train[:,2],add_predict_train)
    metrics['add_acc_valid'] = sklearn.metrics.accuracy_score(standard_labels_valid[:,2],add_predict_valid)
    metrics['add_acc_test'] = sklearn.metrics.accuracy_score(standard_labels_test[:,2],add_predict_test)
    metrics['add_cmat_train'] = sklearn.metrics.confusion_matrix(standard_labels_train[:,2],add_predict_train,labels=range(9))
    metrics['add_cmat_valid'] = sklearn.metrics.confusion_matrix(standard_labels_valid[:,2],add_predict_valid,labels=range(9))
    metrics['add_cmat_test'] = sklearn.metrics.confusion_matrix(standard_labels_test[:,2],add_predict_test,labels=range(9))
    metrics['add_F1_train'] = sklearn.metrics.f1_score(standard_labels_train[:,2],add_predict_train,average='weighted')
    metrics['add_F1_valid'] = sklearn.metrics.f1_score(standard_labels_valid[:,2],add_predict_valid,average='weighted')
    metrics['add_F1_test'] = sklearn.metrics.f1_score(standard_labels_test[:,2],add_predict_test,average='weighted')
    
    inv_predict_train = inv_model.predict(standard_features_train)
    inv_predict_valid = inv_model.predict(standard_features_valid)
    inv_predict_test = inv_model.predict(standard_features_test)
    metrics['inv_acc_train'] = sklearn.metrics.accuracy_score(standard_labels_train[:,3],inv_predict_train)
    metrics['inv_acc_valid'] = sklearn.metrics.accuracy_score(standard_labels_valid[:,3],inv_predict_valid)
    metrics['inv_acc_test'] = sklearn.metrics.accuracy_score(standard_labels_test[:,3],inv_predict_test)
    metrics['inv_cmat_train'] = sklearn.metrics.confusion_matrix(standard_labels_train[:,3],inv_predict_train,labels=range(3))
    metrics['inv_cmat_valid'] = sklearn.metrics.confusion_matrix(standard_labels_valid[:,3],inv_predict_valid,labels=range(3))
    metrics['inv_cmat_test'] = sklearn.metrics.confusion_matrix(standard_labels_test[:,3],inv_predict_test,labels=range(3))
    metrics['inv_F1_train'] = sklearn.metrics.f1_score(standard_labels_train[:,3],inv_predict_train,average='weighted')
    metrics['inv_F1_valid'] = sklearn.metrics.f1_score(standard_labels_valid[:,3],inv_predict_valid,average='weighted')
    metrics['inv_F1_test'] = sklearn.metrics.f1_score(standard_labels_test[:,3],inv_predict_test,average='weighted')
    
    #compute total accuracy
    all_predict_train = np.zeros((int(root_predict_train.shape[0]/12),4))
    all_predict_train[:,0] = root_predict_train[0:all_predict_train.shape[0]]
    notnan_train = ~np.isnan(labels_train[:,0])
    standard_predict_train = (labels_train[notnan_train,0] >= 0)[0:all_predict_train.shape[0]]
    all_predict_train[standard_predict_train,1] = quality_predict_train
    all_predict_train[standard_predict_train,2] = add_predict_train
    all_predict_train[standard_predict_train,3] = inv_predict_train
    total_acc_train = np.all(all_predict_train == (labels_train[notnan_train,:])[0:all_predict_train.shape[0]],axis=1)
    metrics['total_acc_train'] = sum(total_acc_train)/len(total_acc_train)
    
    all_predict_valid = np.zeros((root_predict_valid.shape[0],4))
    all_predict_valid[:,0] = root_predict_valid
    notnan_valid = ~np.isnan(labels_valid[:,0])
    standard_predict_valid = labels_valid[notnan_valid,0] >= 0
    all_predict_valid[standard_predict_valid,1] = quality_predict_valid
    all_predict_valid[standard_predict_valid,2] = add_predict_valid
    all_predict_valid[standard_predict_valid,3] = inv_predict_valid
    total_acc_valid = np.all(all_predict_valid == labels_valid[notnan_valid,:],axis=1)
    metrics['total_acc_valid'] = sum(total_acc_valid)/len(total_acc_valid)
    
    all_predict_test = np.zeros((root_predict_test.shape[0],4))
    all_predict_test[:,0] = root_predict_test
    notnan_test = ~np.isnan(labels_test[:,0])
    standard_predict_test = labels_test[notnan_test,0] >= 0
    all_predict_test[standard_predict_test,1] = quality_predict_test
    all_predict_test[standard_predict_test,2] = add_predict_test
    all_predict_test[standard_predict_test,3] = inv_predict_test
    total_acc_test = np.all(all_predict_test == labels_test[notnan_test,:],axis=1)
    metrics['total_acc_test'] = sum(total_acc_test)/len(total_acc_test)

    #save metrics
    with open(f'{target_dir}{destination}_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
        
def train(model, params, features_train,labels_train,features_valid,labels_valid,features_test,labels_test,standard_features_train,standard_labels_train,standard_features_valid,standard_labels_valid,standard_features_test,standard_labels_test):
    '''Creates and trains model of given type and parameters'''
    if model == 'lr':
        return train_lr(model, params, features_train,labels_train,features_valid,labels_valid,features_test,labels_test,standard_features_train,standard_labels_train,standard_features_valid,standard_labels_valid,standard_features_test,standard_labels_test)
    elif model == 'rf':
        return train_rf(model, params, features_train,labels_train,features_valid,labels_valid,features_test,labels_test,standard_features_train,standard_labels_train,standard_features_valid,standard_labels_valid,standard_features_test,standard_labels_test)
    elif model == 'xgb':
        return train_xgb(model, params, features_train,labels_train,features_valid,labels_valid,features_test,labels_test,standard_features_train,standard_labels_train,standard_features_valid,standard_labels_valid,standard_features_test,standard_labels_test)
    
def train_lr(model, params, features_train,labels_train,features_valid,labels_valid,features_test,labels_test,standard_features_train,standard_labels_train,standard_features_valid,standard_labels_valid,standard_features_test,standard_labels_test):
    '''Logistic regression model
    params:
    sample weight ('T' for balanced or 'F' for none)
    L2weight (float, regularization parameter)
    '''
    weight = 'balanced' if params[0] == 'T' else None
    L2weight = float(params[1])
    
    options = {'class_weight':weight,
               'multi_class':'ovr',
               'C':L2weight,
               'solver':'lbfgs',
               'max_iter':1000}
    root_model = LogisticRegression(**options)
    quality_model = LogisticRegression(**options)
    add_model = LogisticRegression(**options)
    inv_model = LogisticRegression(**options)
    
    #Train models
    root_model.fit(features_train, labels_train[:,0])
    quality_model.fit(standard_features_train, standard_labels_train[:,1])
    add_model.fit(standard_features_train, standard_labels_train[:,2])
    inv_model.fit(standard_features_train, standard_labels_train[:,3])
    
    return root_model, quality_model, add_model, inv_model

def train_rf(model, params, features_train,labels_train,features_valid,labels_valid,features_test,labels_test,standard_features_train,standard_labels_train,standard_features_valid,standard_labels_valid,standard_features_test,standard_labels_test):
    '''Random Forest model
    params:
    sample weight ('T' for balanced or 'F' for none)
    n_estimators (int, number of estimators)
    max_depth (int, max depth of each tree)
    '''
    weight = 'balanced' if params[0] == 'T' else None
    num_estimators = int(params[1])
    max_depth = int(params[2])
    
    model_options = {'class_weight':weight,
               'n_estimators':num_estimators,
                     'max_features':'sqrt',
                    'max_depth':max_depth}
    root_model = RandomForestClassifier(**model_options)
    quality_model = RandomForestClassifier(**model_options)
    add_model = RandomForestClassifier(**model_options)
    inv_model = RandomForestClassifier(**model_options)

    #Train models
    root_model.fit(features_train, labels_train[:,0])
    quality_model.fit(standard_features_train, standard_labels_train[:,1])
    add_model.fit(standard_features_train, standard_labels_train[:,2])
    inv_model.fit(standard_features_train, standard_labels_train[:,3])
    
    return root_model, quality_model, add_model, inv_model
    
def train_xgb(model, params, features_train,labels_train,features_valid,labels_valid,features_test,labels_test,standard_features_train,standard_labels_train,standard_features_valid,standard_labels_valid,standard_features_test,standard_labels_test):
    '''Extreme Gradient Boosting model
    params:
    sample weight ('T' for balanced or 'F' for none)
    n_estimators (int, number of estimators)
    max_depth (int, max depth of each tree)
    learning_rate (float, learning rate of boosting)
    '''
    weight = 'balanced' if params[0] == 'T' else None
    num_estimators = int(params[1])
    max_depth = int(params[2])
    learning_rate = float(params[3])
    
    model_options = {'max_depth': max_depth,
                     'learning_rate': learning_rate,
                     'n_estimators':num_estimators,
                     'n_jobs':4
                    }

    #Train models
    if weight == 'balanced':
        root_model.fit(features_train, labels_train[:,0], sample_weight = get_weights(labels_train[:,0]))
        quality_model.fit(standard_features_train, standard_labels_train[:,1],
                          sample_weight = get_weights(standard_labels_train[:,1]))
        add_model.fit(standard_features_train, standard_labels_train[:,2],
                      sample_weight = get_weights(standard_labels_train[:,2]))
        inv_model.fit(standard_features_train, standard_labels_train[:,3],
                      sample_weight = get_weights(standard_labels_train[:,3]))
    else:
        root_model.fit(features_train, labels_train[:,0])
        quality_model.fit(standard_features_train, standard_labels_train[:,1])
        add_model.fit(standard_features_train, standard_labels_train[:,2])
        inv_model.fit(standard_features_train, standard_labels_train[:,3])
            
    return root_model, quality_model, add_model, inv_model

def get_weights(labels):
    '''Given a set of categorical labels, produces a weight for each label which is inverse to its frequency'''
    values,counts = np.unique(labels,return_counts=True)
    class_weight = {}
    for value,count in zip(values,counts):
        class_weight[value] = 1/(count+1) #add one to prevent divide-by-zero errors
    weights = np.vectorize(class_weight.get)(labels)
    mult = labels.shape[0]/sum(weights)
    return weights*mult
                
if __name__ == '__main__':
    main()