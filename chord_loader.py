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

#shorthand -> added interval, standardized
shorthand_to_add = {'1':'',
                    '5':'',
                    'maj':'','min':'','dim':'','aug':'','sus4':'',#'sus2':'',
                    '7':'min7','min7':'min7','maj7':'maj7','hdim7':'min7','dim7':'6','minmaj7':'maj7',
                    'maj6':'6','min6':'6',
                    'min9':'9','maj9':'9','9':'9',
                    '11':'4','min11':'4','13':'6','maj13':'6','min13':'6'}

#interval list -> added interval, standardized
interval_to_add = {'7':'maj7',
                   'b7':'min7','#6':'min7',
                   '9':'9','2':'9',
                   'b9':'min9','b2':'min9',
                    '11':'4','4':'4',
                  '#11':'tt','#4':'tt','b5':'tt',
                  '13':'6','6':'6',
                   'b13':'min6','b6':'min6','#5':'min6'}

############################################################
#Functions used in simplifying chord properties
############################################################

#returns root in int format
def get_root(row):
    if row[1] == 'sus2':
        return (root_to_num[row[0]] + 7) % 12
    else:
        return root_to_num[row[0]]

def get_quality(row):
    if row[1] in shorthand_to_quality:
        return shorthand_to_quality[row[1]]
    elif (row[0] != 'N') and (row[0] != 'X'):
        #some chords just aren't labeled properly, but I think they're major chords.
        if row[2] == '1,5':
            return 'power'
        elif row[2] == '1':
            return 'unison'
        else:
            return 'maj'
    else:
        return np.nan

#return inversion type, ignoring any past 5/3
def simplify_inversion(row):
    if row[1] == 'sus2':
        return np.nan
    if row[3] in ['5','b5','#5','#4']:
        return '5'
    if row[3] in ['3','b3','b4']:
        return '3'
    else:
        return np.nan

#return added interval
def get_add(row):
    if row[1] not in shorthand_to_add:
        return np.nan
    out = shorthand_to_add[row[1]]
    if out:
        return out
    if not row[2]:
        return np.nan
    interval = str(row[2]).split(',')[-1]
    if interval not in interval_to_add:
        return np.nan
    else:
        return interval_to_add[interval]

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
    transposition - True if you want to duplicate and transpose data"""
    block_length = get_true_block(block_length,minfreq,num_octaves,bins_per_note)
    
    all_features = np.ndarray((0,(12*num_octaves+11)*bins_per_note+2))
    all_labels = pd.DataFrame()
    for i, row in song_list.iterrows():
        if row.dataset == 'isophonics/The Beatles':
            sep = ' '
        else:
            sep = '\t'
        song_labels = get_labels(row.chord_filepath,sep,block_length)
        song_features = get_features(row.mp3_filepath,block_length,minfreq,num_octaves,bins_per_note)
        #note: song_labels is a DataFrame, song_features is an ndarray
        
        #remove rows that don't appear in both arrays
        if song_labels.shape[0] > song_features.shape[0]:
            song_labels = song_labels.iloc[:(song_features.shape[0]),:]
        elif song_labels.shape[0] < song_features.shape[0]:
            song_features = song_features[:(song_labels.shape[0]),:]
        good_rows = ~np.isnan(song_labels['root'])
        song_labels = song_labels.loc[good_rows,:]
        song_features = song_features[good_rows,:]
        
        all_features = np.concatenate((all_features,song_features),axis=0)
        all_labels = all_labels.append(song_labels,ignore_index=True)
    
    if transpose:
        og_size = all_features.shape[0]
        num_features = all_features.shape[1]
        num_labels = all_labels.shape[1]
        labels_copy = all_labels.copy()
        transposition = np.zeros((og_size*12,))
        t_counter = 0
        #duplicate and transpose data
        for i in range(-5,0):
            duplicate_features = np.zeros((og_size,num_features))
            duplicate_features[:,:i] = all_features[:og_size,(-i):]
            duplicate_labels = labels_copy.copy()
            duplicate_labels['root'] = duplicate_labels['root'].apply(lambda x: transpose_root(x,i))
            
            all_features = np.concatenate((all_features,duplicate_features),axis=0)
            all_labels = all_labels.append(duplicate_labels,ignore_index=True)
            t_counter += 1
            transposition[(og_size*t_counter):(og_size*(t_counter+1))] = i
            
        for i in range(1,7):
            duplicate_features = np.zeros((og_size,num_features))
            duplicate_features[:,i:] = all_features[:og_size,:(-i)]
            duplicate_labels = labels_copy.copy()
            duplicate_labels['root'] = duplicate_labels['root'].apply(lambda x: transpose_root(x,i))
            
            all_features = np.concatenate((all_features,duplicate_features),axis=0)
            all_labels = all_labels.append(duplicate_labels,ignore_index=True)
            t_counter += 1
            transposition[(og_size*t_counter):(og_size*(t_counter+1))] = i
        
        return all_features,all_labels,transposition
    else:
        return all_features,all_labels
    
def get_labels(filepath,sep,block_length):
    """loads and processes chord data from target file"""
    df = pd.read_csv(filepath,sep=sep,header=None,names=["start_time","end_time","chord"])
    blocked_df = blockify(df,block_length)
    return chord_simplify(blocked_df)

def get_true_block(block_length,minfreq,num_octaves,bins_per_note):
    "Returns the true block_length, given desired block_length"""
    maxfreq = minfreq*(2**num_octaves)
    sr = 4*maxfreq
    hop_length = round(block_length*sr/64)*64
    true_block = hop_length/sr
    return true_block
    
def get_features(filepath,block_length,minfreq,num_octaves,bins_per_note):
    """Loads mp3 file and computes features (normalized dB spectra).
    The resulting block_length will be returned, as it won't be exactly what was asked for"""
    #calculate constants
    maxfreq = minfreq*(2**num_octaves)
    sr = 4*maxfreq
    hop_length = round(block_length*sr/64)*64
    #load wave
    wav,sr_ = librosa.load(filepath,sr=sr)
    #apply cqt
    cqt_options = {'sr':sr,
               'hop_length':hop_length,
               'fmin':minfreq,
               'n_bins':num_octaves*bins_per_note*12+1,
               'bins_per_octave':bins_per_note*12}
    spec = librosa.cqt(wav, **cqt_options)
    #convert to features
    db = librosa.amplitude_to_db(np.abs(spec)).T
    db_mean = np.mean(db)
    db_var = np.var(db)
    #features is padded out by one row, one column, plus column-space for transposition
    features = np.zeros((db.shape[0]+1,11*bins_per_note+db.shape[1]+1))
    features[1:,(5*bins_per_note+1):-(6*bins_per_note)] = (db - db_mean ) / db_var
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

def transpose_root(rootnum,transposition):
    """transposes the root (in number format, e.g. 0=C, 1=Db)"""
    if rootnum < 0:
        return rootnum
    else:
        return (rootnum + transposition) % 12

def chord_simplify(chords):
    '''
    Interprets and simplifies the chords, given a pd.Series of chords.
    creates a Dataframe with four colums: root, quality, add, and inversion.
    '''
    #extract string matches from chord column
    matchstr = r'^([^:(\/]+):?([^(\/]+)?(?:\(([^)]+)\))?(?:\/(\S*))?$'
    m = chords.str.extract(matchstr)
    #m[0] is the root
    #m[1] is the shorthand (e.g. min7, maj
    #m[2] is added intervals
    #m[3] is bass
    
    df = pd.DataFrame()
    df['root'] = m.apply(lambda x: get_root(x),axis=1)
    df['quality'] = m.apply(lambda x: get_quality(x),axis=1)
    df['add'] = m.apply(lambda x: get_add(x),axis=1)
    df['inversion'] = m.apply(lambda x: simplify_inversion(x),axis=1)
    
    return df


