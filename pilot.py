from psychopy import core, visual, gui, data, event, monitors
from psychopy.tools.filetools import fromFile, toFile
from psychopy.hardware.emulator import launchScan
import numpy as np
import random

'''
TODO:
- Basic Trial Structure: 
    X ITI 2000-5000ms 
    - cHANGE itis TO ADD UP TO MULTIPLES OF tr
    X Stimulus Presentation 11.5 s
    X Contrast Decrement
    X Saccade / No-Saccade Response 1000ms
    X Function: Initialize experiment params
        X Generate trial params for trial in trials
        X Generate position bins
        X randomly select from pos bins
    X Trigger experiment w/ sync pulse
    X Add non-saccade condition (red dot)
    - Add eyelink code
    X Make dir of motion radial (always inward)
    X tinker w/ sd of gaussian
    X set up visual angle degree and monitor
    X keep contrast constant
    x separate saccade / no saccades into blocks

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


def get_keypress(printkey=False):
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
    # If number of trials is not even, return a value error
    if expParams['nTrials'] % 2:
        raise ValueError("Number of Trials not an even number.")

    trialParams = {}

    itis = range(expParams['iti_range'][0]*1000, expParams['iti_range'][1]*1000)

    # Generate isoeccentric positions for stimuli, by randomly sampling radians w/in nPositions bins of equal size
    xstart = np.arange(0, 2*np.pi, 2*np.pi / expParams['nPositions'])
    xstop = xstart + 2*np.pi/expParams['nPositions']
    x = np.random.uniform(xstart, xstop)
    positions = expParams['eccentricity'] * np.array([np.sin(x), np.cos(x)]).T 

    expParams['Positions'] = positions

    responseType = ['Saccade', 'No Saccade'] * int(expParams['nTrials']/2)
    random.shuffle(responseType)

    for trial in range(expParams['nTrials']):
        idx = np.random.randint(expParams['nPositions'])
        pos = positions[idx]
        ori = np.degrees(x[idx]) + 90


        decrements, response_periods, contrasts = generate_decrements(
            expParams['trialDuration'], 
            expParams['max_decrements'], 
            expParams['decrementDuration'], 
            expParams['responseDuration'], 
            constantContrast = expParams['constantContrast'])

        trialParams[str(trial)] = {
            'ITIDur': np.random.choice(itis) / 1000,
            'gratingPos': pos,
            'gratingOri': ori,
            'gratingAng': np.degrees(x[(positions == pos)[:, 0]])[0],
            'decrements': decrements,
            'response_periods': response_periods,
            'contrasts': contrasts,
            'saccadeType': expParams['saccadeType']
        }

    return trialParams


# EXPERIMENT PARAMETERS
nTrials = 10
iti_range = [2, 5]
nPositions = 4
eccentricity = 7
trialDuration = 11.5
subdir = "C:/Users/17868/NYU/FYP/data"
max_decrements = 4
responseDuration = 1
decrementDuration = 0.5
saccadeDuration = 2
expMode = 'Test'
TR = 0.720
volumes = 100
skipSync = 5
sync = '5'

expParams = {
    'nTrials': 10,
    'iti_range': [2, 5],
    'nPositions': 4,
    'eccentricity': 7,
    'trialDuration': 11.5,
    'max_decrements': 4,
    'decrementDuration': 0.5,
    'responseDuration': 1,
    'constantContrast': 0.5,
    'saccadeType': 'Saccade'
}

# INITIALIZE EXPERIMENT
trialParams = generate_experiment(expParams)

# Load parameters from prev run. If not, then use default set
try:
    expInfo = fromFile('lastParams.pickle')
except:
    expInfo = {
        'subject': 0,
        'version': 'pilot'
        }

# Add current time
expInfo['dateStr'] = data.getDateStr()

# Present parameter dialog
dlg = gui.DlgFromDict(expInfo, title = 'Perception Pilot', fixed = ['dateStr'])

if dlg.OK:
    toFile('lastParams.pickle', expInfo)
else:
    core.quit()

# Make tsv files to save experiment data, and contrast responses
saccades_filename = '\sub-' + "{0:0=3d}_".format(expInfo['subject']) + "saccades_run-" + expInfo['dateStr']
saccades_datafile = open(subdir + saccades_filename+'.tsv', 'w')
saccades_datafile.write(
    'trialNum\tITIDur\tgradientPosX\tgradientPosY\torientation\tnDecrements\tnDetected\thits\tfalseAlarms' +
    '\tmeanAccuracy\tsaccadePosX\tsaccadePosY\tsaccadeAng\tsaccadeEcc\tsaccadeRT\tsaccadeErrorDist' +
    '\tsaccadeErrorAng\tsaccadeErrorEcc')

contrast_filename = '\sub-' + "{0:0=3d}_".format(expInfo['subject']) + "contrast_run-" + expInfo['dateStr']
contrast_datafile = open(subdir + contrast_filename+'.tsv', 'w')
contrast_datafile.write('trialNum\tnDecrements\tcontrast\tresponseTimes\tresponseAcc\treactionTime')


# Create window & stimuli
monitor = monitors.Monitor('testMonitor', distance = 68, width = 32)
win = visual.Window(
    [1024, 768], allowGUI=True, monitor=monitor, units='deg')

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
    settings = {'TR': TR, 'volumes': volumes, 'skip': skipSync},
    globalClock = globalClock,
    mode = expMode,
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
for trial in range(nTrials):
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
                reaction_times.append(-1)

        event.clearEvents()

    # Saccade Response 
    print("RESPONSE")
    mouse.mouseClock.reset()
    clicked = False

    while mouse.mouseClock.getTime() < saccadeDuration:

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
        

    # Store trial data
    "SAVING TRIAL DATA"
    contrast_datafile = open(subdir + contrast_filename+'.tsv', 'a')

    [contrast_datafile.write("\n{}\t{}\t{:.2f}\t{:.0f}\t{}\t{:.0f}".format(
        trial,
        decrements.shape[0],
        response_contrast[i]*1000,
        response_times[i],
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
            'saccadeErrorEcc': mouseEcc - eccentricity
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
