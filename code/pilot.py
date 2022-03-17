from psychopy import core, visual, gui, data, event, monitors
from psychopy.tools.filetools import fromFile, toFile
from psychopy.hardware.emulator import launchScan
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
import stat
import numpy as np
import random
import glob
import pickle
import warnings
import os
import os.path as op
import pandas as pd
from datetime import date
from math import fabs, hypot

try:
    import pylink
except ImportError:
    warnings.warn("Unable to find pylink, will not be able to collect eye-tracking data")


# EXPERIMENT PARAMETERS
expParams = {
    'Subject': 0,
    'Run': 1,
    'saccadeType': ['Saccade', 'No Saccade'],
    'saccadeInput': ['EyeLink', "Mouse"],
    'expMode': ['Test', 'Scan'],
    'Output Directory': "/Users/robwoodry/Documents/Research/Interstellar/data",
    #'Output Directory': "/Users/winawerlab/Experiments/Interstellar/data",
    
    # Parameters below this line will be fixed and uneditable from the dialog
    'Screen Distance': 68,
    'Screen Width': 32,
    'Screen Resolution': [1920, 1080],
    'TR': 1,
    'volumes': 270,
    'skipSync': 3,
    'sync': '5',
    'iti_list': [2.5, 3.5, 4.5, 5.5],
    'nPositions': 4,
    'max_decrements': 1,
    'eccentricity': 7,
    'trialDuration': 4,
    'saccadeDuration': 1,
    'decrementDuration': 0.5,
    'responseDuration': 1,
    'constantContrast': 0.65,
}

'''
TODO:
- Add feedback to end of trial
- double check data collection
- write up tsv variable descriptions and how they are computed
X Download Eyelink package from SR tools and get pylink to import properly
- Figure out issue with while loop and timing/refresh rate
X Add code to create subject data folder if not created
X Add code to save a session tsv file setting params for trial and contrasts
X add code that checks to see if file already exists
- swap mouse input for eyetracker mouse input from psychopy
- Edit saccade computations accordingly ^
- Add run column to design files
'''

'''HELPER FUNCTIONS'''
def _setup_eyelink(win_size, win):
    """set up the eyelink eye-tracking
    """

    # Connect to eyelink
    eyetracker = pylink.EyeLink('100.1.1.1')
    genv = EyeLinkCoreGraphicsPsychoPy(eyetracker, win)
    pylink.openGraphicsEx(genv)

    # Set content of edf file
    eyetracker.sendCommand('link_sample_data=LEFT,RIGHT,GAZE,AREA')
    eyetracker.sendCommand('file_sample_data=LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS')
    eyetracker.sendCommand('file_event_filter=LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON')

    # Set coords
    eyetracker.sendCommand('screen_pixel_coords=0 0 {} {}'.format(*win_size))
    eyetracker.sendMessage('DISPLAY_COORDS 0 0 {} {}'.format(*win_size))

    # Calibrate
    eyetracker.setCalibrationType('HV5')
    eyetracker.doTrackerSetup(win_size)
    pylink.closeGraphics()

    return eyetracker


def receiveEDF(eyetracker, data_path, subject, run):
    edf_file = "Is%02dr%02d.edf" % (subject, run)
    local_edf = os.path.join(data_path, "sub-%03d" % subject, edf_file)
    print(edf_file, local_edf)

    eyetracker.receiveDataFile(edf_file, local_edf)

def generate_decrements(trialDuration, max_decrements, decrementDuration, responsePeriod, constantContrast = 0):
    # Generate N = max_decrements bins of indices in ms for decrements to possibly start
    bins = np.array_split(np.arange(0, trialDuration*1000 + 1), max_decrements)

    # Get low and high range of vals for bins, pad by half of response period so no overlap between bins
    bins_start = [np.min(a) + responsePeriod*1000 for a in bins]
    bins_stop = [np.max(a) - responsePeriod*1000 for a in bins]

    # Rondomly select decrements' start times, add stop time = decrementDuration.
    decs = np.random.randint(bins_start, bins_stop)
    decrements = np.asarray([decs, decs + decrementDuration*1000]).T

    # Randomly select n decrements where n is a value from 1 to max_decrements
    n = np.random.randint(max_decrements) + 1
    decrements = decrements[np.sort(np.random.choice(decrements.shape[0], n, replace=False))]

    # Generate valid response periods, from start of decrement to stop of decrement + responsePeriod
    valid_response_periods = np.copy(decrements)
    valid_response_periods[:, 1] += responsePeriod*1000

    contrasts = np.round(np.random.rand(decrements.shape[0]), 2)

    if constantContrast: contrasts = np.zeros(decrements.shape[0]) + constantContrast

    return decrements, valid_response_periods, contrasts


def get_keypress(printkey=False, sync = expParams['sync']):
    keys = event.getKeys()
    if keys and keys[0] == 'q':
        win.close()
        core.quit()
    elif keys and keys[0] == sync:
        if printkey: print("SCANNER PULSE")
    elif keys and keys[0] != sync:
        if printkey: print(keys[0])
        return keys[0]
    else:
        return None


def saccade_response(parameters, eyetracker, event, fixation, win, globalClock):
    scn_width, scn_height = win.size
    
    if expParams['saccadeType'] == 'Saccade':
        fixation.color = 'green'
    else:
        fixation.color = 'red'
    
    saccadeClockStart = globalClock.getTime()
        
    # 0- left, 1-right, 2-binocular
    eye_used = eyetracker.eyeAvailable()
    if eye_used == 1:
        eyetracker.sendMessage("EYE_USED 1 RIGHT")
    elif eye_used == 0 or eye_used == 2:
        eyetracker.sendMessage("EYE_USED 0 LEFT")
        eye_used = 0
    else:
        print("Error in getting the eye information!")
        return pylink.TRIAL_ERROR

    got_sac = False 
    sac_start_time = -1
    SRT = -1
    land_err = -1
    sac = None

    event.clearEvents()
    fixation.draw()
    win.flip()
        
    while globalClock.getTime() - saccadeClockStart <= parameters['saccadeDuration']:
        if not got_sac:
            # Grab events in buffer
            eye = eyetracker.getNextData()
            if (eye is not None) and (eye == pylink.ENDSACC):
                eye_dat = eyetracker.getFloatData()
                if eye_dat.getEye() == eye_used:
                    sac_amp = eye_dat.getAmplitude()
                    sac_start_time = eye_dat.getStartTime()
                    sac_end_time = eye_dat.getEndTime()
                    sac_start_pos = eye_dat.getStartGaze()
                    sac_end_pos = eye_dat.getEndGaze()
                        
                    if sac_start_time <= saccadeClockStart:
                        sac_start_time = -1
                        pass
                    elif hypot(sac_amp[0], sac_amp[1]) > 1.5:
                        offset = int(eyetracker.trackerTime() - sac_start_time)
                        sac_response_msg = '{} saccade_resp'.format(offset)
                        eyetracker.sendMessage(sac_response_msg)
                        SRT = sac_start_time - saccadeClockStart
                        sac = np.array([sac_x + scn_width/2.0, sac_y + scn_height/2.0])
                        land_err = np.linalg.norm(sac - parameters['GratingPos'])
                        got_sac = True
        else:
            pass
    
    return SRT, sac, land_err       


def init_pilot_params(expParams):
    # Set filenames and paths to be used
    tsv_filename_trial = '/sub-%03d_designTrial.tsv' % expParams['Subject']
    tsv_filename_contrast = '/sub-%03d_designContrast.tsv' % expParams['Subject']
    pickle_filename = '/sub-%03d_expParams.pickle' % expParams['Subject']
    subj_dir = op.join('../design/', "sub-%03d" % expParams['Subject'])

    # If subject directory does not exist, make one
    os.umask(0)
    if not op.exists(subj_dir):
        os.makedirs(subj_dir, mode = 0o777)
    else:
        os.chmod(subj_dir, mode = 0o777)
        

    # If previous params file doesn't already exist in aforementioned directory, 
    # initialize one with new positions
    if pickle_filename.strip('/') not in os.listdir(subj_dir):

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
        header = True
    
    # If params file already exists, load previous position/angle parameters
    else:
        file = open(subj_dir + pickle_filename, "rb")
        prevParams = pickle.load(file)
        file.close()
        
        expParams['Positions'] = prevParams['Positions']
        expParams['AnglesRadians'] = prevParams['AnglesRadians']
        write_mode = 'a'
        header = False

    # Generate randomized trials & itis
    trialnums = list(range(expParams['nPositions']))
    random.shuffle(trialnums)
    itis = expParams['iti_list'] * int(expParams['nPositions'] / len(expParams['iti_list']))
    random.shuffle(itis)

    # Initialize dictionaries
    trialParams = {}
    saccade_data = {}
    trial_design = pd.DataFrame(columns =         
        ['run', 'trialNum', 'ITIDur', 'gratingPosX', 'gratingPosY', 'gratingOri', 'gratingAng', 'saccadeType',
        'saccadeDuration', 'saccadeInput'])
    contrast_design = pd.DataFrame(columns = ['run', 'trialNum', 'decrementStart', 'decrementStop', 
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
            'run': expParams['Run'],
            'trialNum': str(i),
            'ITIDur': itis[trial],
            'gratingPosX': pos[0],
            'gratingPosY': pos[1],
            'gratingOri': ori,
            'gratingAng': np.degrees(expParams['AnglesRadians'][(expParams["Positions"] == pos)[:, 0]])[0],
            'saccadeType': expParams['saccadeType'],
            'saccadeDuration': expParams['saccadeDuration'],
            'saccadeInput': expParams['saccadeInput']
            }, ignore_index = True, )

        cdf = pd.DataFrame(np.asarray([expParams['Run'] * np.ones(len(decrements)),
            i * np.ones(len(decrements)), decrements[:, 0], decrements[:, 1], 
            response_periods[:, 1], contrasts]).T,
            columns = ['run', 'trialNum', 'decrementStart', 'decrementStop', 
                'responseStop', 'contrast']
        )
        contrast_design = contrast_design.append(cdf, ignore_index=True)

    # Write design dataframes to tsv
    trial_design.to_csv(subj_dir + tsv_filename_trial, sep = '\t', mode=write_mode, header=header)
    contrast_design.to_csv(subj_dir + tsv_filename_contrast, sep = '\t', mode=write_mode, header=header)

    # Return run trial parameters
    return trialParams, saccade_data


# Add current time
expParams['dateStr'] = data.getDateStr()

# Present parameter dialog
dlg = gui.DlgFromDict(expParams, title = 'Perception Pilot', fixed = [
    'Screen Distance', 'Screen Width', 'Screen Resolution',
    'dateStr', 'TR', 'volumes', 'skipSync', 'sync', 'iti_list', 
    'nPositions', 'max_decrements', 'eccentricity', 'trialDuration',
    'saccadeDuration', 'decrementDuration', 'responseDuration', 
    'constantContrast'],
    order = list(expParams.keys()))

# INITIALIZE EXPERIMENT
subject = expParams['Subject']
subdir = '../data/sub-%03d/' % subject
dateStr = expParams['dateStr']
run = expParams['Run']

trialParams, saccade_data = init_pilot_params(expParams)
contrast_data = {}
nPressed = -1

# Save experiment params to file
if dlg.OK:
    params_filename = '../design/sub-%03d/sub-%03d_expParams.pickle' % (subject, subject)
    toFile(params_filename, expParams)
else:
    core.quit()

# Create window & stimuli
monitor = monitors.Monitor('testMonitor', distance = expParams['Screen Distance'], width = expParams['Screen Width'])
win = visual.Window(
    expParams['Screen Resolution'], allowGUI=True, monitor=monitor, units='deg',
    fullscr = False)

grating = visual.GratingStim(
    win, sf=1, size=3, mask='gauss', maskParams = {'sd': 5},
    pos=[-4,0], ori=0, units = 'deg')
fixation = visual.GratingStim(
    win, color=-1, colorSpace='rgb', tex=None, mask='cross', size=0.5)


# Eyetracker 
if expParams['saccadeInput'] == 'EyeLink':
    edf_name = "Is%02dr%02d.EDF" % (subject, run)
    print(edf_name)
    eyetracker = _setup_eyelink(expParams['Screen Resolution'], win)
    
    eyetracker.openDataFile(edf_name)
    pylink.flushGetkeyQueue()
    eyetracker.setOfflineMode()
    eyetracker.startRecording(1, 1, 1, 1)
    
elif expParams['saccadeInput'] == 'Mouse':
    eyetracker = event.Mouse(win=win)

# Display instructions
if expParams['saccadeType'] == 'Saccade':
    instructions = "[PARTICIPANT] Press 1 when you detect a change in contrast. At the end of each trial, make a saccade"
elif expParams['saccadeInput'] == 'No Saccade':
    instructions = "[PARTICIPANT] Press 1 when you detect a change in contrast. At the end of each trial, DO NOT make a saccade"
msg1 = visual.TextStim(win, pos=[0, +3], text='[OPERATOR] Hit 0 key when participant is ready')
msg2 = visual.TextStim(win, pos=[0, -3],
text="[PARTICIPANT] Press 1 when you detect a change in contrast. At the end of each trial, ")

msg1.draw()
msg2.draw()
fixation.draw()
win.flip()

# Wait for a response
event.waitKeys(keyList=['0'])

# Start up some clocks
globalClock = core.Clock()
trialClock = core.Clock()
itiClock = core.Clock()

# fMRI Sync Trigger
vol = launchScan(
    win,
    settings = {'TR': expParams['TR'], 'volumes': expParams['volumes'], 'skip': expParams['skipSync']},
    globalClock = globalClock,
    mode = expParams['expMode'],
    wait_msg = "Waiting for Sync Pulse"
)

# TRIAL LOOP
for trial in range(expParams['nPositions']):
    print("Trial " + str(trial) + " out of " + str(expParams['nPositions']))
    # Initialize Trial
    parameters = trialParams[str(trial)]

    print("INITIALIZING TRIAL")
    if parameters['saccadeInput'] == 'Mouse': eyetracker.setVisible(1)
    
    grating.pos = parameters['gratingPos']
    grating.ori = parameters['gratingOri']
    decrements = parameters['decrements']
    contrasts = parameters['contrasts']
    response_periods = parameters['response_periods']
    
    response_times = []
    response_acc = []
    response_contrast = []
    reaction_times = []
    detected = np.zeros(decrements.shape[0])

    # ITI        
    fixation.mask = 'cross'
    fixation.color = 'black'
    fixation.size = 0.5

    fixation.draw()
    win.flip()

    itiClock.reset()
    if expParams['saccadeInput'] == 'EyeLink': 
        while itiClock.getTime() < parameters['ITIDur']:
            get_keypress(printkey=True)
            event.clearEvents()
    
    # Stimulus Presentation - Initialize trial parameters
    lastContrastTimeSet = False
    lastContrastTime = 0
    lastContrast = 1
    nContrasts = 0
    hits = 0
    falseAlarms = 0
    fixation.mask = 'circle'
    fixation.size = 0.3
    
    fixation.draw()
    win.flip()

    trialClock.reset()
    
    # Stimulus Presentation - Trial
    if expParams['saccadeInput'] == 'EyeLink': 
        eyetracker.sendMessage('TRIALID %s' % parameters['trialNum'])
    while trialClock.getTime() < expParams['trialDuration']:
        t = trialClock.getTime() * 1000 

         # If trial time is in any decrement range, 
        if ((decrements[:, 0] <=  t) & (t < decrements[:, 1])).any():
            # Decrease contrast
            grating.setPhase(0.05, '+')
            grating.contrast = contrasts[(decrements[:, 0] <=  t) & (t < decrements[:, 1])][0]
            
            if not lastContrastTimeSet:
                lastContrastTime = t
                lastContrastTimeSet = True
                lastContrast = grating.contrast
                nContrasts += 1
                
            grating.draw()
            fixation.draw()
            win.flip()
        
        # Otherwise,
        else:
            # No contrast manipulation
            grating.setPhase(0.05, '+')
            grating.contrast = 1
            
            if lastContrastTimeSet:
                lastContrastTimeSet = False
                lastContrast = 1
                
            grating.draw()
            fixation.draw()
            win.flip()

        key = get_keypress(printkey=True)
        if key:
            response_times.append(globalClock.getTime())
            nPressed += 1
            response_acc = 0

            if ((response_periods[:, 0] <=  t) & (t < response_periods[:, 1])).any():
                response_acc = 1
                detected[nContrasts - 1] = 1
                hits += 1
            
            else:
                falseAlarms += 1
            
            contrast_data[str(nPressed)] = {
                'trialNum': trial,
                'nDecrements': decrements.shape[0],
                'contrast': lastContrast,
                'responseTimes': globalClock.getTime()*1000,
                'responseAcc': response_acc,
                'reactionTime': t - lastContrastTime
            }
            print(contrast_data[str(nPressed)].items())

        event.clearEvents()

    # Saccade Response 
    SRT, sac, land_err = saccade_response(
        parameters, eyetracker, event, fixation, win, globalClock)
            
    # Store trial data
    print("Storing Data")
    print("LE:", land_err)
    
    if sac:
        saccade_data[str(trial)]['nDetected'] = np.sum(detected)
        saccade_data[str(trial)]['nMissed'] = decrements.shape[0] - np.sum(detected)
        saccade_data[str(trial)]['hits'] = hits
        saccade_data[str(trial)]['falseAlarms'] = falseAlarms
        if hits + falseAlarms != 0: 
            saccade_data[str(trial)]['meanAccuracy'] = hits / (hits + falseAlarms)
        else:
            saccade_data[str(trial)]['meanAccuracy'] = 0
#        # Saccade data
#        saccade_data[str(trial)]['saccadePosX'] = mousePos[0]
#        saccade_data[str(trial)]['saccadePosY'] = mousePos[1]
#        saccade_data[str(trial)]['saccadeAng'] = mouseAng
#        saccade_data[str(trial)]['saccadeEcc'] = mouseEcc
#        saccade_data[str(trial)]['saccadeRT'] = mouseTime
#        saccade_data[str(trial)]['saccadeErrorDist'] = np.linalg.norm(mousePos - parameters['gratingPos'])
#        saccade_data[str(trial)]['saccadeErrorAng'] = angError
#        saccade_data[str(trial)]['saccadeErrorEcc'] = mouseEcc - expParams['eccentricity']

    else:
        saccade_data[str(trial)]['nDetected'] = np.sum(detected)
        saccade_data[str(trial)]['nMissed'] = decrements.shape[0] - np.sum(detected)
        saccade_data[str(trial)]['hits'] = hits
        saccade_data[str(trial)]['falseAlarms'] = falseAlarms
        if hits + falseAlarms != 0: 
            saccade_data[str(trial)]['meanAccuracy'] = hits / (hits + falseAlarms)
        else:
            saccade_data[str(trial)]['meanAccuracy'] = 0
    
# Post-trial delay for 6 s after last trial
delayStart = globalClock.getTime()
fixation.mask = 'cross'
fixation.color = 'black'
fixation.size = 0.5

fixation.draw()
win.flip()

# If data subdir doesn't exist, create one
subdir = os.path.join(expParams["Output Directory"], "sub-%03d" % subject)
os.umask(0)
if not op.exists(subdir):
    os.makedirs(subdir, mode = 0o777)
    
while globalClock.getTime() - delayStart < 6:
    pass


# Save data to files
saccades_filename = subdir + '/sub-%03d_saccades_run-%s.tsv' % (subject, str(run))
saccades_datafile = open(saccades_filename, 'w')
saccades_datafile.write("\t".join(map(str, list(saccade_data[str(0)].keys()))))
for trial in range(expParams['nPositions']):
    saccades_datafile.write("\n" + "\t".join(
        map(str, list(saccade_data[str(trial)].values()))))
        
saccades_datafile.close()

contrast_filename = subdir + '/sub-%03d_contrast_run-%s.tsv' % (subject, str(run))
contrast_datafile = open(contrast_filename, 'w')
contrast_datafile.write("\t".join(map(str, list(contrast_data[str(0)].keys()))))
for n in range(nPressed+1):
    contrast_datafile.write("\n" + "\t".join(
        map(str, list(contrast_data[str(n)].values()))))

contrast_datafile.close()

if expParams['saccadeInput'] == 'EyeLink':
    eyetracker = pylink.getEYELINK()
    eyetracker.stopRecording()
    eyetracker.closeDataFile()
    eyetracker.setOfflineMode()
    eyetracker.sendCommand('clear_screen 0')
    pylink.msecDelay(500)
    
    eyetracker.close()


print("EXP END")
win.close()

eyetracker = pylink.EyeLink('100.1.1.1')
eyetracker.open()
receiveEDF(eyetracker, expParams['Output Directory'], subject, run)

core.quit()



