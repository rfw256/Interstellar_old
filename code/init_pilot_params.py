import PyQt5
from psychopy import core, visual, gui, data, event, monitors
from psychopy.tools.filetools import fromFile, toFile
from psychopy.hardware.emulator import launchScan
import numpy as np
import random
import glob
import pickle
import warnings
import os.path as op
import os
import pandas as pd

def init_pilot_params(expParams):
    # Set filenames and paths to be used
    tsv_filename_trial = '/sub-%03d_designTrial.tsv' % expParams['Subject']
    tsv_filename_contrast = '/sub-%03d_designContrast.tsv' % expParams['Subject']
    pickle_filename = '/sub-%03d_expParams.pickle' % expParams['Subject']
    subj_dir = op.join('../design/', "sub-%03d" % expParams['Subject'])

    # If subject directory does not exist, make one
    if not op.exists(subj_dir):
        os.makedirs(subj_dir)

    # If previous params file doesn't already exist in aforementioned directory, 
    # initialize one with new positions
    if pickle_filename not in os.listdir(subj_dir):
        # If number of trials is not even, return a value error
        if expParams['nPositions'] % len(expParams['iti_list']):
            raise ValueError("Number of Positions not a multiple of number of ITIs.")

        # Generate isoeccentric positions for stimuli, by randomly sampling radians 
        # w/in nPositions bins of equal size
        xstart = np.arange(0, 2*np.pi, 2*np.pi / expParams['nPositions'])
        xstop = xstart + 2*np.pi/expParams['nPositions']
        x = np.random.uniform(xstart, xstop)
        positions = expParams['eccentricity'] * np.array([np.sin(x), np.cos(x)]).T 

        expParams['Positions'] = positions
        expParams['AnglesRadians'] = x
        write_mode = 'w'
    
    # If params file already exists, load previous position/angle parameters
    else:
        file = open(pickle_filename, "rb")
        prevParams = pickle.load(file)
        file.close()
        
        expParams['Positions'] = prevParams['Positions']
        expParams['AnglesRadians'] = prevParams['AnglesRadians']
        write_mode = 'a'

    # Generate randomized trials & itis
    trialnums = list(range(expParams['nPositions']))
    random.shuffle(trialnums)
    itis = expParams['iti_list'] * int(expParams['nPositions'] / len(expParams['iti_list']))
    random.shuffle(itis)

    # Initialize dictionaries
    trialParams = {}
    saccade_data = {}
    trial_design = pd.DataFrame(columns =         
        ['trialNum', 'ITIDur', 'gratingPos', 'gratingOri', 'gratingAng', 'saccadeType',
        'saccadeDuration', 'saccadeInput'])
    contrast_design = pd.DataFrame(columns = ['trialNum', 'decrementStart', 'decrementStop', 
        'responseStop', 'contrast'])

    # Loop through each trial and generate trial-specific parameters
    for i, trial in enumerate(trialnums):
        pos = expParams['Positions'][trial]
        ori = np.degrees(expParams['AnglesRadians'][trial]) + 90
        
        if ori >= 360: ori -= 360

        # Generate 1-4 contrasts per trial, at random intervals
        decrements, response_periods, contrasts = generate_decrements(
            expParams['trialDuration'], 
            expParams['max_decrements'], 
            expParams['decrementDuration'], 
            expParams['responseDuration'], 
            constantContrast = expParams['constantContrast'])

        # Save generated trial parameters
        trialParams[str(i)] = {
            'trialNum': str(i),
            'ITIDur': itis[trial],
            'gratingPos': pos,
            'gratingOri': ori,
            'gratingAng': np.degrees(expParams['AnglesRadians'][(expParams["Positions"] == pos)[:, 0]])[0],
            'decrements': decrements,
            'response_periods': response_periods,
            'contrasts': contrasts,
            'saccadeType': expParams['saccadeType'],
            'saccadeDuration': expParams['saccadeDuration'],
            'saccadeInput': expParams['saccadeInput']
        }
        
        # Preload saccade data for each trial w/ trial parameters
        saccade_data[str(i)] = {
            'trialNum': i,
            'ITIDur': itis[trial],
            'gratingPosX': pos[0],
            'gratingPosY': pos[1],
            'gratingOri': ori,
            'gratingAng': np.degrees(expParams['AnglesRadians'][(expParams["Positions"] == pos)[:, 0]])[0],
            'nDecrements': decrements.shape[0],
            'nDetected': 0,
            'nMissed': 0,
            'hits': 0,
            'falseAlarms': 0,
            'meanAccuracy': 0,
            'saccadeType': expParams['saccadeType'],
            'saccadePosX': None,
            'saccadePosY': None,
            'saccadeAng': None,
            'saccadeEcc': None,
            'saccadeRT': None,
            'saccadeErrorDist': None,
            'saccadeErrorAng': None,
            'saccadeErrorEcc': None
        }

        # Append design info to design DataFrames
        trial_design = trial_design.append({
            'trialNum': str(i),
            'ITIDur': itis[trial],
            'gratingPos': pos,
            'gratingOri': ori,
            'gratingAng': np.degrees(expParams['AnglesRadians'][(expParams["Positions"] == pos)[:, 0]])[0],
            'saccadeType': expParams['saccadeType'],
            'saccadeDuration': expParams['saccadeDuration'],
            'saccadeInput': expParams['saccadeInput']
            }, ignore_index = True)

        cdf = pd.DataFrame(np.asarray([i * np.ones(len(decrements)), decrements[:, 0], decrements[:, 1], response_periods[:, 1], contrasts]).T,
            columns = ['trialNum', 'decrementStart', 'decrementStop', 'responseStop', 'contrast']
        )
        contrast_design = contrast_design.append(cdf, ignore_index=True)

    # Write design dataframes to tsv
    trial_design.to_csv(op.join(subj_dir, tsv_filename_trial), sep = '\t', mode=write_mode)
    contrast_design.to_csv(op.join(subj_dir, tsv_filename_contrast), sep = '\t', mode=write_mode)

    # Return run trial parameters
    return trialParams, saccade_data






    


