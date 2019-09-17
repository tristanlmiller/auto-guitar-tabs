import pandas as pd
import numpy as np
import re

def chord_simplify(df):
    '''
    Interprets and simplifies the chords, given a dataframe with one column labeled 'chord'.
    Adds to the dataframe 4 new columns called 'root','quality','add','inversion'.
    '''
    
    ####################
    #extract string matches from chord column
    matchstr = r'^([^:(\/]+):?([^(\/]+)?(?:\(([^)]+)\))?(?:\/(\S*))?$'
    m = df['chord'].str.extract(matchstr)
    #m[0] is the root
    #m[1] is the shorthand (e.g. min7, maj
    #m[2] is added intervals
    #m[3] is bass
    
    ####################
    #Define some dicts

    #Root note -> integer
    root_to_num = {'Cb':11,'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'E#':5,'Fb':4,'F':5,'F#':6,'Gb':6,
                  'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11,'B#':0,'N':-1,'X':np.nan}

    #Root integer -> standardized note
    num_to_root = {0:'C',1:'Db',2:'D',3:'Eb',4:'E',5:'F',6:'F#',7:'G',8:'Ab',9:'A',10:'Bb',11:'B',-1:'~'}

    #shorthand -> chord quality
    shorthand_to_quality = {'1':'unison',
                            '5':'power',
                            'maj':'maj','min':'min','dim':'dim','aug':'aug','sus4':'sus',
                            'sus2':'sus', #and change the root to the perfect 5th
                            '7':'maj','min7':'min','maj7':'maj','hdim7':'dim','dim7':'dim','minmaj7':'min',
                            'maj6':'maj','min6':'min',
                            'min9':'min','maj9':'maj','9':'maj',
                            '11':'maj','min11':'min','13':'maj','maj13':'maj','min13':'min',np.nan:np.nan}

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
    
    ########################
    #Define some functions to extract/simplify chord properties
    
    #returns root in int format
    def get_root(row):
        if row[1] == 'sus2':
            return (root_to_num[row[0]] + 7) % 12
        else:
            return root_to_num[row[0]]
    
    def get_quality(row):
    if row[1] in shorthand_to_quality:
        return shorthand_to_quality[row[1]]
    elif row[0] != 'N' and row[0] != 'X':
        #some chords just aren't labeled properly, but I think they're major chords.
        if row[2] == '1,5':
            return 'power'
        elif row[2] = '1':
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
    
    df['root'] = m.apply(lambda x: get_root(x),axis=1)
    df['quality'] = m.apply(lambda x: get_quality(x),axis=1)
    df['add'] = m.apply(lambda x: get_add(x),axis=1)
    df['inversion'] = m.apply(lambda x: simplify_inversion(x),axis=1)


