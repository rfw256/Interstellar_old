from psychopy import core, visual, gui, data, event, monitors
from psychopy.tools.filetools import fromFile, toFile
from psychopy.hardware.emulator import launchScan
import numpy as np
import random
import glob
import pickle

# EXPERIMENT PARAMETERS
expParams = {
    'Subject': 0,
    'Session': 1,
    'saccadeType': ['Saccade', 'No Saccade'],
    'expMode': ['Test', 'Scan'],
    'Output Directory': "/Users/rfw256/Documents/Research/Interstellar/data",
    
    # Parameters below this line will be fixed and uneditable from the dialogbox
    'Screen Distance': 68,
    'Screen Width': 32,
    'Screen Resolution': [1024, 768],
    'TR': 1,
    'volumes': 5,
    'skipSync': 5,
    'sync': '5',
    'iti_list': [2.5, 3.5, 4.5, 5.5],
    'nPositions': 4,
    'max_decrements': 4,
    'eccentricity': 7,
    'trialDuration': 11.5,
    'saccadeDuration': 2,
    'decrementDuration': 0.5,
    'responseDuration': 1,
    'constantContrast': 0.65,
}

'''

TODO:
- Basic Trial Structure: 
    X ITI 2000-5000ms 
    X Stimulus Presentation 11.5 s
    X Contrast Decrement
    X Saccade / No-Saccade Response 1000ms
    X Function: Initialize experiment params
        X Generate trial params for trial in trials
        X Generate position bins
        X randomly select from pos bins
    X Trigger experiment w/ sync pulse
    X Add non-saccade condition (red dot)
    X Make dir of motion radial (always inward)
    X tinker w/ sd of gaussian
    X set up visual angle degree and monitor
    X keep contrast constant
    X separate saccade / no saccades into blocks
    X Fix dialog box issue
    X Transfer to Mac OS
    X For each run, go through each position once (random shuffle)
    X Static ITIS of 2.5-5.5 in steps of 1, so total trial time is multiple of TR
        X repeat N times where N = nPositions / 4
        X Randomly shuffle ITIs for each run
        X swap nTrials with nPositions
    X Tweak dialog box and params to have only whats needed
    X Change reaction time formula to time since last contrast
    X Investigate why contrast is multiplied by 1000
    X change contrast data formula to most recent contrast level
    X add missed column to data
    X Add 6 seconds to end of run
    X Add load previous params functionality and save current params
        X Add positions permanence via exp params
    - Edit most recent contrast decrement calculation
    - Add eyelink code
    - Figure out issue with while loop and timing/refresh rate 
    - (Check if pscyhopy is able to lock refresh rate?)

'''

'''HELPER FUNCTIONS'''
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


def generate_experiment(expParams):
    # Try to load previous subject parameters
    params_filenames = glob.glob(expParams["Output Directory"] + 
        "/sub-" + "{0:0=3d}_".format(expParams["Subject"]) + "*.pickle")
    
    # If Previous params exist, load positions and angles
    if params_filenames:
        file = open(params_filenames[-1], "rb")
        prevParams = pickle.load(file)
        file.close()
        
        expParams['Positions'] = prevParams['Positions']
        expParams['AnglesRadians'] = prevParams['AnglesRadians']
    
    # If previous params for sub does not exist, initialize positions and angles
    else:
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
    
    trialnums = list(range(expParams['nPositions']))
    random.shuffle(trialnums)
    itis = expParams['iti_list'] * int(expParams['nPositions'] / len(expParams['iti_list']))
    random.shuffle(itis)
    trialParams = {}
    print(expParams)
    
    for i, trial in enumerate(trialnums):
        pos = expParams['Positions'][trial]
        ori = np.degrees(expParams['AnglesRadians'][trial]) + 90

        decrements, response_periods, contrasts = generate_decrements(
            expParams['trialDuration'], 
            expParams['max_decrements'], 
            expParams['decrementDuration'], 
            expParams['responseDuration'], 
            constantContrast = expParams['constantContrast'])

        trialParams[str(i)] = {
            'ITIDur': itis[trial],
            'gratingPos': pos,
            'gratingOri': ori,
            'gratingAng': np.degrees(expParams['AnglesRadians'][(expParams["Positions"] == pos)[:, 0]])[0],
            'decrements': decrements,
            'response_periods': response_periods,
            'contrasts': contrasts,
            'saccadeType': expParams['saccadeType']
        }

    return trialParams

# Load parameters from prev run. If not, then use default set

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

# Make tsv files to save experiment data, and contrast responses
subdir = expParams['Output Directory']
subject = expParams['Subject']
date = expParams['dateStr']
session = expParams['Session']

saccades_filename = '/sub-' + "{0:0=3d}_".format(subject) + "saccades_run-" + str(session)
saccades_datafile = open(subdir + saccades_filename +'.tsv', 'w')
saccades_datafile.write(
    'trialNum\tITIDur\tgradientPosX\tgradientPosY\torientation\tnDecrements\tnDetected\tnMissed\thits\tfalseAlarms' +
    '\tmeanAccuracy/tsaccadeType\tsaccadePosX\tsaccadePosY\tsaccadeAng\tsaccadeEcc\tsaccadeRT\tsaccadeErrorDist' +
    '\tsaccadeErrorAng\tsaccadeErrorEcc')

contrast_filename = '/sub-' + "{0:0=3d}_".format(subject) + "contrast_run-" + str(session)
contrast_datafile = open(subdir + contrast_filename+'.tsv', 'w')
contrast_datafile.write('trialNum\tnDecrements\tcontrast\tresponseTimes\tresponseAcc\treactionTime')

# INITIALIZE EXPERIMENT
trialParams = generate_experiment(expParams)

# Save experiment params to file
if dlg.OK:
    params_filename = '/sub-' + "{0:0=3d}_".format(subject) + "expParams_run-" + str(session)
    toFile(subdir + params_filename + '.pickle', expParams)
    print("EXPPARAMS:")
    print(expParams)
else:
    core.quit()

# Create window & stimuli
monitor = monitors.Monitor('testMonitor', distance = expParams['Screen Distance'], width = expParams['Screen Width'])
win = visual.Window(
    expParams['Screen Resolution'], allowGUI=True, monitor=monitor, units='deg')

mouse = event.Mouse(win=win)

grating = visual.GratingStim(
    win, sf=1, size=3, mask='gauss', maskParams = {'sd': 5},
    pos=[-4,0], ori=0, units = 'deg')
fixation = visual.GratingStim(
    win, color=-1, colorSpace='rgb', tex=None, mask='cross', size=0.5)

globalClock = core.Clock()
trialClock = core.Clock()
itiClock = core.Clock()

# fMRI Trigger
vol = launchScan(
    win,
    settings = {'TR': expParams['TR'], 'volumes': expParams['volumes'], 'skip': expParams['skipSync']},
    globalClock = globalClock,
    mode = expParams['expMode'],
    wait_msg = "Waiting for Sync Pulse"
)

# Display instructions
msg1 = visual.TextStim(win, pos=[0, +3], text='Hit a key when ready')
msg2 = visual.TextStim(win, pos=[0, -3],
text="Press 1 when you detect a change in contrast.")

msg1.draw()
msg2.draw()
fixation.draw()
win.flip()

# Wait for a response
event.waitKeys(keyList=['1'])

# TRIAL LOOP
for trial in range(expParams['nPositions']):
    # Initialize Trial
    parameters = trialParams[str(trial)]

    print("INITIALIZING TRIAL")
    mouse.setVisible(1)
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
    print("ITI")
    fixation.mask = 'cross'
    fixation.color = 'black'
    fixation.size = 0.5

    fixation.draw()
    win.flip()

    itiClock.reset()
    while itiClock.getTime() < parameters['ITIDur']:
        get_keypress(printkey=True)
        event.clearEvents()
    
    # Stimulus Presentation
    print("STIMULUS")
    fixation.mask = 'circle'
    fixation.size = 0.3
    fixation.draw()
    win.flip()

    trialClock.reset()
    while trialClock.getTime() < expParams['trialDuration']:
        t = trialClock.getTime() * 1000 

         # If trial time is in any decrement range, 
        if ((decrements[:, 0] <=  t) & (t < decrements[:, 1])).any():
            # Decrease contrast
            grating.setPhase(0.05, '+')
            grating.contrast = contrasts[(decrements[:, 0] <=  t) & (t < decrements[:, 1])][0]
            grating.draw()
            fixation.draw()
            win.flip()
        
        # Otherwise,
        else:
            # No contrast manipulation
            grating.setPhase(0.05, '+')
            grating.contrast = 1
            grating.draw()
            fixation.draw()
            win.flip()

        key = get_keypress(printkey=True)
        if key:
            response_times.append(globalClock.getTime())

            if ((response_periods[:, 0] <=  t) & (t < response_periods[:, 1])).any():
                response_acc.append(1)
                valid_resp = (response_periods[:, 0] <=  t) & (t < response_periods[:, 1])
                response_contrast.append(contrasts[valid_resp][0])
                reaction_times.append(t - response_periods[valid_resp][0, 0])
                detected[valid_resp] = 1

            else:
                response_acc.append(0)
                response_contrast.append(1)
                reaction_time = np.min(np.abs(t - np.asarray(response_periods[:, 0])))
                reaction_times.append(reaction_time)

        event.clearEvents()

    # Saccade Response 
    print("RESPONSE")
    mouse.mouseClock.reset()
    clicked = False

    while mouse.mouseClock.getTime() < expParams['saccadeDuration']:

        if parameters['saccadeType'] == 'Saccade':
            fixation.color = 'green'
            fixation.draw()
            win.flip()

            get_keypress(printkey=True)

            if not clicked:
                buttons_pressed = mouse.getPressed()

                if sum(buttons_pressed):
                    mousePos = mouse.getPos()
                    mouseTime = mouse.mouseClock.getTime()
                    print(mousePos, mouseTime)
                    clicked = True

                    normMousePos = mousePos / np.linalg.norm(mousePos)
                    mouseAng = np.degrees(np.arccos(np.dot(
                        np.array([1, 0]), normMousePos)))
                    angError = np.degrees(np.arccos(np.dot(
                        parameters['gratingPos'] / np.linalg.norm(parameters['gratingPos']), 
                        normMousePos)))
                    mouseEcc = np.linalg.norm(mousePos - np.array([0, 0]))
            event.clearEvents()

        else:
            fixation.color = 'red'
            fixation.draw()
            win.flip()
        
    
    # Post-trial delay for 6 s after last trial
    if trial == expParams['nPositions'] - 1:
        delayStart = globalClock.getTime()
        fixation.mask = 'cross'
        fixation.color = 'black'
        fixation.size = 0.5

        fixation.draw()
        win.flip()
        
        while globalClock.getTime() - delayStart < 6:
            pass
    
    # Store trial data
    contrast_datafile = open(subdir + contrast_filename+'.tsv', 'a')

    [contrast_datafile.write("\n{}\t{}\t{:.2f}\t{:.0f}\t{}\t{:.0f}".format(
        trial,
        decrements.shape[0],
        response_contrast[i],
        response_times[i]*1000,
        response_acc[i],
        reaction_times[i]))
        for i in range(len(response_times))]

    saccades_datafile = open(subdir + saccades_filename+'.tsv', 'a')
    if clicked:
        saccade_data = {
            'trialNum': trial,
            'ITIDur': parameters['ITIDur'],
            'gratingPosX': parameters['gratingPos'][0],
            'gratingPosY': parameters['gratingPos'][1],
            'gratingOri': parameters['gratingOri'],
            'nDecrements': decrements.shape[0],
            'nDetected': np.sum(detected),
            'nMissed': decrements.shape[0] - np.sum(detected),
            'hits': np.sum(np.asarray(response_acc) == 1),
            'falseAlarms': np.sum(np.asarray(response_acc) == 0),
            'meanAccuracy': np.mean(np.asarray(response_acc)),
            'saccadeType': parameters['saccadeType'],
            'saccadePosX': mousePos[0],
            'saccadePosY': mousePos[1],
            'saccadeAng': mouseAng,
            'saccadeEcc':mouseEcc,
            'saccadeRT': mouseTime,
            'saccadeErrorDist': np.linalg.norm(mousePos - parameters['gratingPos']),
            'saccadeErrorAng': angError,
            'saccadeErrorEcc': mouseEcc - expParams['eccentricity']
        }

    else:
        saccade_data = {
            'trialNum': trial,
            'ITIDur': parameters['ITIDur'],
            'gratingPosX': parameters['gratingPos'][0],
            'gratingPosY': parameters['gratingPos'][1],
            'gratingOri': parameters['gratingOri'],
            'nDecrements': decrements.shape[0],
            'nDetected': np.sum(detected),
            'nMissed': decrements.shape[0] - np.sum(detected),
            'hits': np.sum(np.asarray(response_acc) == 1),
            'falseAlarms': np.sum(np.asarray(response_acc) == 0),
            'meanAccuracy': np.mean(np.asarray(response_acc)),
            'saccadeType': parameters['saccadeType'],
            'saccadePosX': None,
            'saccadePosY': None,
            'saccadeAng': None,
            'saccadeEcc': None,
            'saccadeRT': None,
            'saccadeErrorDist': None,
            'saccadeErrorAng': None,
            'saccadeErrorEcc': None
        }
    
    saccades_datafile.write("\n" + "\t".join(map(str, list(saccade_data.values()))))
