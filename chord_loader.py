#!/usr/bin/env python
import pandas as pd
import numpy as np
import re
import librosa

############################################################
#Define some dicts
############################################################

#Root integer -> standardized note
num_to_root = {0:'C',1:'Db',2:'D',3:'Eb',4:'E',5:'F',6:'F#',7:'G',8:'Ab',9:'A',10:'Bb',11:'B',-1:'~'}

#Root note -> integer
root_to_num = {'Cb':11,'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'E#':5,'Fb':4,'F':5,'F#':6,'Gb':6,
              'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11,'B#':0,'N':-1,'X':np.nan}

#shorthand -> chord quality
shorthand_to_quality = {'1':'unison',
                        '5':'power',
                        'maj':'maj','min':'min','dim':'dim','aug':'aug','sus4':'sus',
                        'sus2':'sus', #and change the root to the perfect 5th
                        '7':'maj','min7':'min','maj7':'maj','hdim7':'dim','dim7':'dim','minmaj7':'min',
                        'maj6':'maj','min6':'min',
                        'min9':'min','maj9':'maj','9':'maj',
                        '11':'maj','min11':'min','13':'maj','maj13':'maj','min13':'min'}

#chord quality -> number class
quality_to_num = {'unison':0,'power':1,'maj':2,'min':3,'sus':4,'dim':5,'aug':6}
#number class -> chord quality
num_to_quality = {0:'unison',1:'power',2:'maj',3:'min',4:'sus',5:'dim',6:'aug'}

#shorthand -> added interval, standardized
shorthand_to_add = {'1':'',
                    '5':'',
                    'maj':'','min':'','dim':'','aug':'','sus4':'','sus2':'',
                    '7':'7','min7':'7','maj7':'maj7','hdim7':'7','dim7':'6','minmaj7':'maj7',
                    'maj6':'6','min6':'6',
                    'min9':'9','maj9':'9','9':'9',
                    '11':'4','min11':'4','13':'6','maj13':'6','min13':'6'}

#interval list -> added interval, standardized
interval_to_add = {'7':'maj7',
                   'b7':'7','#6':'7',
                   '9':'9','2':'9',
                   'b9':'min9','b2':'min9',
                    '11':'4','4':'4',
                  '#11':'tt','#4':'tt','b5':'tt',
                  '13':'6','6':'6',
                   'b13':'min6','b6':'min6','#5':'min6'}

#added interval -> number class
add_to_num = {'':0,'maj7':1,'7':2,'9':3,'min9':4,'4':5,'tt':6,'6':7,'min6':8}
#number class -> added interval, standardized
num_to_add = {0:'',1:'maj7',2:'min7',3:'9',4:'min9',5:'4',6:'tt',7:'6',8:'min6'}

#inversion -> number class
inv_to_num = {'':0,'5':1,'3':2}
#number class -> inversion, standardized
num_to_inv = {0:'',1:'/5',2:'/3'}

############################################################
#Functions used in simplifying chord properties
############################################################

#returns root in int format
def get_root(row):
    if row[1] == 'sus2':
        return (root_to_num[row[0]] + 7) % 12
    elif row[0] in root_to_num:
        return root_to_num[row[0]]
    else:
        return np.nan

#returns quality in int format
def get_quality(row):
    if row[1] in shorthand_to_quality:
        return quality_to_num[shorthand_to_quality[row[1]]]
    elif (row[0] != 'N') and (row[0] != 'X'):
        #some chords just aren't labeled properly, but I think they're major chords.
        if row[2] == '1,5':
            return quality_to_num['power']
        elif row[2] == '1':
            return quality_to_num['unison']
        else:
            return quality_to_num['maj']
    else:
        return np.nan

#return inversion type in int format.  Only counts first and second inversions
def simplify_inversion(row):
    if row[1] == 'sus2':
        return inv_to_num['']
    if row[3] in ['5','b5','#5','#4']:
        return inv_to_num['5']
    if row[3] in ['3','b3','b4']:
        return inv_to_num['3']
    else:
        return inv_to_num['']

#return added interval in int format
def get_add(row):
    if (row[0] != 'N') and (row[0] != 'X'):
        #valid chord
        if row[1] == 'sus2':
            #ignore sus2 chords, since it's too much of a pain to compute adds
            return add_to_num['']
        if row[1] in shorthand_to_add:
            out = shorthand_to_add[row[1]]
            if out:
                #if the shorthand implies an add, use that one
                return add_to_num[out]
        #otherwise look for add in add column
        if not row[2]:
            #if no add is noted, there is no add
            return add_to_num['']
        #if adds are noted, check last one
        interval = str(row[2]).split(',')[-1]
        if interval not in interval_to_add:
            #if the given add isn't one of the ones I care about (e.g. 5)
            return add_to_num['']
        else:
            #if the given add is one I care about
            return add_to_num[interval_to_add[interval]]
    else:
        #no valid chord means no add
        return add_to_num['']

############################################################
#Functions for loading and processing files
############################################################    

def get_features_labels(song_list,block_length,minfreq,num_octaves,bins_per_note,transpose):
    """Loads and processes the chord and mp3 files, deletes rows with missing data
    
    Arguments:
    song_directory - a dataframe with song information (subset of song_directory, assumes all rows are valid)
    block_length - target time interval between feature rows
    minfreq - minimum frequency analyzed
    num_octaves - number octaves analyzed
    bins_per_note - number of bins per note (12 notes per octave)
    transpose - True if you want to duplicate and transpose data"""
    block_length = get_true_block(block_length,minfreq,num_octaves,bins_per_note)
    
    all_features = np.ndarray((0,(12*num_octaves+11)*bins_per_note+1))
    all_labels = np.ndarray((0,4))
    for i, row in song_list.iterrows():
        if row.dataset == 'isophonics/The Beatles':
            sep = ' '
        else:
            sep = '\t'
        song_labels = get_labels(row.chord_filepath,sep,block_length)
        song_features = get_features(row.mp3_filepath,block_length,minfreq,num_octaves,bins_per_note)
        #note: both of these are ndarrays
        
        #remove rows that don't appear in both arrays
        if song_labels.shape[0] > song_features.shape[0]:
            song_labels = song_labels[:(song_features.shape[0]),:]
        elif song_labels.shape[0] < song_features.shape[0]:
            song_features = song_features[:(song_labels.shape[0]),:]
        good_rows = ~np.isnan(song_labels[:,0])
        song_labels = song_labels[good_rows,:]
        song_features = song_features[good_rows,:]
        
        all_features = np.concatenate((all_features,song_features),axis=0)
        all_labels = np.concatenate((all_labels,song_labels),axis=0)
    
    if transpose:
        og_size = all_features.shape[0]
        num_features = all_features.shape[1]
        labels_copy = all_labels.copy()
        #duplicate and transpose data
        for i in range(-5,0):
            duplicate_features = np.zeros((og_size,num_features))
            duplicate_features[:,1:(i*bins_per_note)] = all_features[:og_size,(1-i*bins_per_note):]
            duplicate_features[:,0] = all_features[:og_size,0]
            duplicate_labels = labels_copy.copy()
            duplicate_labels[:,0] = transpose_root(duplicate_labels[:,0],i)
            
            all_features = np.concatenate((all_features,duplicate_features),axis=0)
            all_labels = np.concatenate((all_labels,duplicate_labels),axis=0)
            
        for i in range(1,7):
            duplicate_features = np.zeros((og_size,num_features))
            duplicate_features[:,(1+i*bins_per_note):] = all_features[:og_size,1:(-i*bins_per_note)]
            duplicate_features[:,0] = all_features[:og_size,0]
            duplicate_labels = labels_copy.copy()
            duplicate_labels[:,0] = transpose_root(duplicate_labels[:,0],i)
            
            all_features = np.concatenate((all_features,duplicate_features),axis=0)
            all_labels = np.concatenate((all_labels,duplicate_labels),axis=0)
        
    return all_features,all_labels

def standardize_root(features,labels,bins_per_note,dropna=True,transposed=False):
    """Given a set of features and labels, transposes each time block so that root is 0
    if dropna is true, then rows with N or X chords are dropped
    if transposed is true, then only takes first 1/12th of the data (the rest being transposed)"""
    if transposed:
        og_size = int(features.shape[0]/12)
    else:
        og_size = features.shape[0]
    num_features = features.shape[1]
    
    duplicate_features = np.zeros((og_size,num_features))
    duplicate_features[:,0] = features[:og_size,0]
    for j,i in enumerate(labels[:og_size,0]):
        i = int(i)
        if i > 0 and i <= 6:
            duplicate_features[j,1:(-i*bins_per_note)] = features[j,(1+i*bins_per_note):]
        elif i > 6:
            i -= 12
            duplicate_features[j,(1-i*bins_per_note):] = features[j,1:(i*bins_per_note)]
        elif i == 0:
            duplicate_features[j,:] = features[j,:]
    if dropna:
        valid_rows = np.logical_and(~np.equal(labels[:og_size,0], -1), ~np.isnan(labels[:og_size,0]))
        duplicate_features = duplicate_features[valid_rows,:]
        duplicate_labels = (labels[:og_size,:])[valid_rows,:]
    return duplicate_features, duplicate_labels
                                                  
                                                  
def transpose_root(roots,transposition):
    """transposes an array of roots"""
    roots[roots >= 0] = (roots[roots >= 0] + transposition) % 12
    return roots

def get_labels(filepath,sep,block_length):
    """loads and processes chord data from target file"""
    df = pd.read_csv(filepath,sep=sep,header=None,names=["start_time","end_time","chord"])
    blocked_df = blockify(df,block_length)
    return chord_simplify(blocked_df)

def get_true_block(block_length,minfreq,num_octaves,bins_per_note):
    "Returns the true block_length, given desired block_length"""
    maxfreq = minfreq*(2**num_octaves)
    sr = 4*maxfreq
    hop_restriction = 2**(num_octaves-1)
    hop_length = round(block_length*sr/hop_restriction)*hop_restriction
    true_block = hop_length/sr
    return true_block
    
def get_features(filepath,block_length,minfreq,num_octaves,bins_per_note):
    """Loads mp3 file and computes features (normalized dB spectra).
    The resulting block_length will be returned, as it won't be exactly what was asked for"""
    #calculate constants
    maxfreq = minfreq*(2**num_octaves)
    sr = 4*maxfreq
    hop_restriction = 2**(num_octaves-1)
    hop_length = round(block_length*sr/hop_restriction)*hop_restriction
    #load wave
    wav,sr_ = librosa.load(filepath,sr=sr)
    #apply cqt
    cqt_options = {'sr':sr,
               'hop_length':hop_length,
               'fmin':minfreq,
               'n_bins':num_octaves*bins_per_note*12,
               'bins_per_octave':bins_per_note*12}
    spec = librosa.cqt(wav, **cqt_options)
    #convert to features
    db = librosa.amplitude_to_db(np.abs(spec)).T
    db_mean = np.mean(db)
    db_std = np.std(db)
    #features is padded out by one row, one column, plus column-space for transposition
    features = np.zeros((db.shape[0]+1,11*bins_per_note+db.shape[1]+1))
    features[1:,(5*bins_per_note+1):-(6*bins_per_note)] = (db - db_mean ) / db_std
    #very first column/row is a flag for start of song
    features[0,0] = 1
    return features

def blockify(df,block_length):
    """Given a dataframe read from a chord file, converts to Series with chords in each block.
    Uses whatever chord is played at the start of the block."""
    
    #pad size with one N chord at beginning
    size = int(df['end_time'].iloc[-1]/block_length)+1
    output = pd.Series(np.zeros((size,)))
    output[:] = ''
    
    for i, row in df.iterrows():
        #first find the blocks where the chord starts and ends
        start_block = int(row.start_time / block_length)+1
        end_block = int(row.end_time / block_length)+1
        output[start_block:end_block] = row.chord
    output[0] = 'N'
    return output

def chord_simplify(chords):
    '''
    Interprets and simplifies the chords, given a pd.Series of chords.
    creates a ndarray with four colums: root, quality, add, and inversion.
    '''
    #extract string matches from chord column
    matchstr = r'^([^:(\/]+):?([^(\/]+)?(?:\(([^)]+)\))?(?:\/(\S*))?$'
    m = chords.str.extract(matchstr)
    #m[0] is the root
    #m[1] is the shorthand (e.g. min7, maj)
    #m[2] is added intervals
    #m[3] is bass
    
    out = np.zeros((m.shape[0],4))
    out[:,0] = m.apply(lambda x: get_root(x),axis=1)
    out[:,1] = m.apply(lambda x: get_quality(x),axis=1)
    out[:,2] = m.apply(lambda x: get_add(x),axis=1)
    out[:,3] = m.apply(lambda x: simplify_inversion(x),axis=1)
    
    return out

def read_chords(labels):
    '''Given a set of model predictions, produces a DataFrame with additional columns for display by the app.
    '''
    chord_info = pd.DataFrame({'rootnum':labels[:,0],'qualitynum':labels[:,1],
                               'addnum':labels[:,2],'invnum':labels[:,3]})
    
    chord_info = chord_info.fillna(value=0)
    
    chord_info['root'] = [num_to_root[x] for x in chord_info['rootnum']]
    chord_info['quality'] = [num_to_quality[x] for x in chord_info['qualitynum']]
    chord_info['interval'] = [num_to_add[x] for x in chord_info['addnum']]
    chord_info['inv'] = [num_to_inv[x] for x in chord_info['invnum']]
    chord_info['full'] = [row.root+row.quality+row.interval+row.inv for i,row in chord_info.iterrows()]
    
    #hard part: producing the list of notes
    chord_info['notes'] = ''
    bass_octave = '4'
    triad_octave = '5'
    add_octave = '6'
    for i,row in chord_info.iterrows():
        if row.root == '~':
            continue
        notes = ''
        #get bass note
        if row.inv == '':
            notes += num_to_root[row.rootnum] + bass_octave
        elif row.inv == '/5':
            if row.quality  == 'aug':
                notes += num_to_root[(row.rootnum+8) % 12] + bass_octave
            elif row.quality == 'dim':
                notes += num_to_root[(row.rootnum+6) % 12] + bass_octave
            else:
                notes += num_to_root[(row.rootnum+7) % 12] + bass_octave
        elif row.inv == '/3':
            if row.quality in ['min','dim']:
                notes += num_to_root[(row.rootnum+3) % 12] + bass_octave
            else:
                notes += num_to_root[(row.rootnum+4) % 12] + bass_octave
        
        #get basic triad
        if row.quality == 'maj':
            notes += ' ' + num_to_root[(row.rootnum+4) % 12] + triad_octave
            notes += ' ' + num_to_root[(row.rootnum+7) % 12] + triad_octave
            notes += ' ' + num_to_root[row.rootnum] + triad_octave
        elif row.quality == 'min':
            notes += ' ' + num_to_root[(row.rootnum+3) % 12] + triad_octave
            notes += ' ' + num_to_root[(row.rootnum+7) % 12] + triad_octave
            notes += ' ' + num_to_root[row.rootnum] + triad_octave
        elif row.quality == 'power':
            notes += ' ' + num_to_root[(row.rootnum+7) % 12] + triad_octave
            notes += ' ' + num_to_root[row.rootnum] + triad_octave
        elif row.quality == 'dim':
            notes += ' ' + num_to_root[(row.rootnum+3) % 12] + triad_octave
            notes += ' ' + num_to_root[(row.rootnum+6) % 12] + triad_octave
            notes += ' ' + num_to_root[row.rootnum] + triad_octave
        elif row.quality == 'aug':
            notes += ' ' + num_to_root[(row.rootnum+4) % 12] + triad_octave
            notes += ' ' + num_to_root[(row.rootnum+8) % 12] + triad_octave
            notes += ' ' + num_to_root[row.rootnum] + triad_octave
        else: #unison case
            notes += ' ' + num_to_root[row.rootnum] + triad_octave
            
        #get add
        if row.interval == 'maj7':
            notes += ' ' + num_to_root[(row.rootnum+11) % 12] + add_octave
        elif row.interval == 'min7':
            notes += ' ' + num_to_root[(row.rootnum+10) % 12] + add_octave
        elif row.interval == '6':
            notes += ' ' + num_to_root[(row.rootnum+9) % 12] + add_octave
        elif row.interval == 'min6':
            notes += ' ' + num_to_root[(row.rootnum+8) % 12] + add_octave
        elif row.interval == '9':
            notes += ' ' + num_to_root[(row.rootnum+2) % 12] + add_octave
        elif row.interval == 'min9':
            notes += ' ' + num_to_root[(row.rootnum+1) % 12] + add_octave
        elif row.interval == '4':
            notes += ' ' + num_to_root[(row.rootnum+5) % 12] + add_octave
        elif row.interval == 'tt':
            notes += ' ' + num_to_root[(row.rootnum+6) % 12] + add_octave
        chord_info['notes'].iloc[i] = notes
        
    return chord_info
