from psychopy import core, visual, gui, data, event, monitors
from psychopy.tools.filetools import fromFile, toFile
from psychopy.hardware.emulator import launchScan
import numpy as np
import random
import glob
import pickle
import warnings

try:
    import pylink
except ImportError:
    warnings.warn("Unable to find pylink, will not be able to collect eye-tracking data")


# EXPERIMENT PARAMETERS
expParams = {
    'Subject': 0,
    'Run': 1,
    'saccadeType': ['Saccade', 'No Saccade'],
    'saccadeInput': ['Mouse', 'EyeLink'],
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
    X Edit most recent contrast decrement calculation
    X Have data saved at the end of the run, as opposed to at the end of every trial
    X add angle in degrees to saccade data
    - Add feedback to end of trial
    - double check data collection
        X Fix orientation so if above 360, resets to 0
        X saccadeErrorEcc looks off, so does saccadeEcc
            *** On iMac w/ retina, mouse pos is doubled. Effect should go away 
              when displaying to an external screen ***
        X Fix hits, nDetected, and nDecrements so they add up properly
        X Fix contrast data saving
    - write up tsv variable descriptions and how they are computed
    - Add eyelink code
    - Figure out issue with while loop and timing/refresh rate 
    - (Check if pscyhopy is able to lock refresh rate?)


'''

'''HELPER FUNCTIONS'''
def _setup_eyelink(win_size):
    """set up the eyelink eye-tracking
    """

    # Connect to eyelink
    eyetracker = pylink.EyeLink('192.168.1.5')
    pylink.openGraphics()

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
    saccade_data = {}

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
        
        if ori >= 360: ori -= 360

        decrements, response_periods, contrasts = generate_decrements(
            expParams['trialDuration'], 
            expParams['max_decrements'], 
            expParams['decrementDuration'], 
            expParams['responseDuration'], 
            constantContrast = expParams['constantContrast'])

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

    return trialParams, saccade_data


def saccade_response(parameters, saccadeInput, saccade, event, fixation, win):
    if expParams['saccadeType'] == 'Saccade':
        fixation.color = 'green'
    else:
        fixation.color = 'red'
        
    if parameters['saccadeInput'] == 'Mouse':
        saccade.mouseClock.reset()
        clicked = False
        
        while saccade.mouseClock.getTime() < parameters['saccadeDuration']:
            fixation.draw()
            win.flip()
            
            if parameters['saccadeType'] == 'Saccade':
                get_keypress(printkey=True)

                if not clicked:
                    buttons_pressed = saccade.getPressed()
                    if sum(buttons_pressed):
                        mousePos = saccade.getPos()
                        mouseTime = saccade.mouseClock.getTime()
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
                
    if parameters['saccadeInput'] == 'EyeLink':
        event.sendMessage("TRIALID %02d" % parameters['trialNum'])
    

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
run = expParams['Run']

# INITIALIZE EXPERIMENT
trialParams, saccade_data = generate_experiment(expParams)
contrast_data = {}
nPressed = -1

# Save experiment params to file
if dlg.OK:
    params_filename = '/sub-' + "{0:0=3d}_".format(subject) + "expParams"
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

# Eyetracker 
if expParams['saccadeInput'] == 'EyeLink':
    eyetracker = _setup_eyelink(expParams['Screen Resolution'])
    edf_path = '/sub-' + "{0:0=3d}_".format(subject) + "eyelink_run-" + str(run)
    
    assert edf_path is not None, "edf_path must be set so we can save the eyetracker output!"
    eyetracker.openDataFile('temp.EDF')
    pylink.flushGetkeyQueue()
    eyetracker.startRecording(1, 1, 1, 1)

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
    print("Trial " + str(trial) + " out of " + str(expParams['nPositions']))
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
    fixation.mask = 'cross'
    fixation.color = 'black'
    fixation.size = 0.5

    fixation.draw()
    win.flip()

    itiClock.reset()
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
    mouse.mouseClock.reset()
    clicked = False
    
    if parameters['saccadeType'] == 'Saccade':
        fixation.color = 'green'
        
        while mouse.mouseClock.getTime() < expParams['saccadeDuration']:
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
        
        while mouse.mouseClock.getTime() < expParams['saccadeDuration']:
            fixation.draw()
            win.flip()
            
    # Store trial data
    print("Storing Data")
    print(contrast_data.keys())
    
    if clicked:
        saccade_data[str(trial)]['nDetected'] = np.sum(detected)
        saccade_data[str(trial)]['nMissed'] = decrements.shape[0] - np.sum(detected)
        saccade_data[str(trial)]['hits'] = hits
        saccade_data[str(trial)]['falseAlarms'] = falseAlarms
        if hits + falseAlarms != 0: 
            saccade_data[str(trial)]['meanAccuracy'] = hits / (hits + falseAlarms)
        else:
            saccade_data[str(trial)]['meanAccuracy'] = 0
        saccade_data[str(trial)]['saccadePosX'] = mousePos[0]
        saccade_data[str(trial)]['saccadePosY'] = mousePos[1]
        saccade_data[str(trial)]['saccadeAng'] = mouseAng
        saccade_data[str(trial)]['saccadeEcc'] = mouseEcc
        saccade_data[str(trial)]['saccadeRT'] = mouseTime
        saccade_data[str(trial)]['saccadeErrorDist'] = np.linalg.norm(mousePos - parameters['gratingPos'])
        saccade_data[str(trial)]['saccadeErrorAng'] = angError
        saccade_data[str(trial)]['saccadeErrorEcc'] = mouseEcc - expParams['eccentricity']

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
    if trial == expParams['nPositions'] - 1:
        delayStart = globalClock.getTime()
        fixation.mask = 'cross'
        fixation.color = 'black'
        fixation.size = 0.5

        fixation.draw()
        win.flip()
        
        while globalClock.getTime() - delayStart < 6:
            pass
            
        # Save data to files
        saccades_filename = '/sub-' + "{0:0=3d}_".format(subject) + "saccades_run-" + str(run)
        saccades_datafile = open(subdir + saccades_filename +'.tsv', 'w')
        saccades_datafile.write("\t".join(map(str, list(saccade_data[str(0)].keys()))))
        for trial in range(expParams['nPositions']):
            saccades_datafile.write("\n" + "\t".join(
                map(str, list(saccade_data[str(trial)].values()))))
                
        saccades_datafile.close()


        contrast_filename = '/sub-' + "{0:0=3d}_".format(subject) + "contrast_run-" + str(run)
        contrast_datafile = open(subdir + contrast_filename+'.tsv', 'w')
        contrast_datafile.write("\t".join(map(str, list(contrast_data[str(0)].keys()))))
        for n in range(nPressed+1):
            contrast_datafile.write("\n" + "\t".join(
                map(str, list(contrast_data[str(n)].values()))))
        
        contrast_datafile.close()
