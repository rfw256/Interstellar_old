from __future__ import division
from __future__ import print_function

import pylink
import os
import platform
import random
import time
import sys
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
from psychopy import visual, core, event, monitors, gui, data
from datetime import date
from PIL import Image
from string import ascii_letters, digits
import os.path as op
import numpy as np
import pandas as pd
from psychopy.tools.filetools import fromFile, toFile
import pickle
from psychopy.hardware.emulator import launchScan

# EXPERIMENT PARAMATERS
expParams = {
    'Subject': 0,
    'Run': 1,
    'saccadeType': ['Saccade', 'No Saccade'],
    'saccadeInput': ['EyeLink', "Mouse"],
    'expMode': ['Test', 'Scan'],
    'use_retina': True,
    'Output Directory': "/Applications/EyeLink/SampleExperiments/Python/examples/Psychopy_examples/interstellar/data",
    #'Output Directory': "/Users/robwoodry/Documents/Research/Interstellar/data",
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
    'Contrast': 0.7,
    'eccentricity': 7,
    'trialDuration': 4,
    'saccadeDuration': 1,
    'decrementDuration': 0.5,
    'responseDuration': 1,
    'constantContrast': 0.65
    }

# HELPER FUNCTIONS
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


def init_pilot_params(expParams):
    # Switch to the script folder
    script_path = os.path.dirname(sys.argv[0])
    if len(script_path) != 0:
        os.chdir(script_path)

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
            'trialDuration': expParams['trialDuration'],
            'ITIDur': itis[trial],
            'gratingPos': pos,
            'gratingOri': ori,
            'gratingAng': np.degrees(expParams['AnglesRadians'][(expParams["Positions"] == pos)[:, 0]])[0],
            'decrements': decrements,
            'response_periods': response_periods,
            'contrast': expParams['Contrast'],
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


def connect_eyelink(expParams):
    edf_fname = "Is%02d_r%02d" % (expParams['Subject'], expParams['Run'])
    results_folder = '../results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    time_str = time.strftime("_%Y_%m_%d_%H_%M", time.localtime())
    session_identifier = edf_fname + time_str

    session_folder = os.path.join(results_folder, session_identifier)
    if not os.path.exists(session_folder):
        os.makedirs(session_folder)

    if expParams['saccadeInput'] == 'Mouse':
        el_tracker = pylink.EyeLink(None)
    elif expParams['saccadeInput'] == 'EyeLink':
        try:
            el_tracker = pylink.EyeLink("100.1.1.1")
        except RuntimeError as error:
            print('ERROR:', error)
            core.quit()
            sys.exit()
    
    return el_tracker, session_folder


def create_EDF(expParams, el_tracker):
    edf_fname = "Is%02d_r%02d" % (expParams['Subject'], expParams['Run'])
    edf_file = edf_fname + ".EDF"
    try:
        el_tracker.openDataFile(edf_file)
    except RuntimeError as err:
        print('ERROR:', err)
        # close the link if we have one open
        if el_tracker.isConnected():
            el_tracker.close()
        core.quit()
        sys.exit()
    
    preamble_text = 'RECORDED BY %s' % os.path.basename(__file__)
    el_tracker.sendCommand("add_file_preamble_text '%s'" % preamble_text)
    
    return edf_file


def configure_eyelink(expParams, el_tracker):
    el_tracker.setOfflineMode()
    eyelink_ver = 0  # set version to 0, in case running in Dummy mode
    if expParams['saccadeInput'] == 'EyeLink':
        vstr = el_tracker.getTrackerVersionString()
        eyelink_ver = int(vstr.split()[-1].split('.')[0])
        print('Running experiment on %s, version %d' % (vstr, eyelink_ver))

    file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
    link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'

    if eyelink_ver > 3:
        file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
        link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
    else:
        file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT'
        link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'
    el_tracker.sendCommand("file_event_filter = %s" % file_event_flags)
    el_tracker.sendCommand("file_sample_data = %s" % file_sample_flags)
    el_tracker.sendCommand("link_event_filter = %s" % link_event_flags)
    el_tracker.sendCommand("link_sample_data = %s" % link_sample_flags)

    el_tracker.sendCommand("calibration_type = HV9")
    el_tracker.sendCommand("button_function 5 'accept_target_fixation'")

    el_tracker.sendCommand('calibration_area_proportion 0.88 0.83')
    el_tracker.sendCommand('validation_area_proportion 0.88 0.83')
    
    return el_tracker


def setup_graphics(expParams, el_tracker):
#    mon = monitors.Monitor('myMonitor', distance = expParams['Screen Distance'], width = expParams['Screen Width'])
#    win = visual.Window(expParams['Screen Resolution'],
#                        fullscr=False,
#                        monitor=mon,
#                        allowGUI = True,
#                        units='deg')
    mon = monitors.Monitor('testMonitor', distance = expParams['Screen Distance'], width = expParams['Screen Width'])
    win = visual.Window(
        expParams['Screen Resolution'], allowGUI=True, monitor=mon, units='deg',
        fullscr = False)

    scn_width, scn_height = win.size

    if 'Darwin' in platform.system():
        if expParams['use_retina']:
            scn_width = int(scn_width/2.0)
            scn_height = int(scn_height/2.0)

    el_coords = "screen_pixel_coords = 0 0 %d %d" % (scn_width - 1, scn_height - 1)
    el_tracker.sendCommand(el_coords)

    dv_coords = "DISPLAY_COORDS  0 0 %d %d" % (scn_width - 1, scn_height - 1)
    el_tracker.sendMessage(dv_coords)

    genv = EyeLinkCoreGraphicsPsychoPy(el_tracker, win)
    print(genv) 

    foreground_color = (-1, -1, -1)
    background_color = win.color
    genv.setCalibrationColors(foreground_color, background_color)

    genv.setTargetType('picture')
    genv.setPictureTarget(os.path.join('images', 'fixTarget.bmp'))

    genv.setCalibrationSounds('', '', '')

    if expParams['use_retina']:
        genv.fixMacRetinaDisplay()

    pylink.openGraphicsEx(genv)
    
    return mon, win, genv


def clear_screen(win, genv):
    """ clear up the PsychoPy window""" 

    win.fillColor = genv.getBackgroundColor()
    win.flip()


def show_msg(win, text, genv, wait_for_keypress=True):
    """ Show task instructions on screen""" 
    scn_width, scn_height = win.size
    msg = visual.TextStim(win, text,
                          color=genv.getForegroundColor(),
                          wrapWidth=scn_width/2)
    clear_screen(win, genv)
    msg.draw()
    win.flip()

    # wait indefinitely, terminates upon any key press
    if wait_for_keypress:
        event.waitKeys()
        clear_screen(win)


def terminate_task(win, session_folder, edf_file, genv):
    """ Terminate the task gracefully and retrieve the EDF data file

    file_to_retrieve: The EDF on the Host that we would like to download
    win: the current window used by the experimental script
    """

    el_tracker = pylink.getEYELINK()

    if el_tracker.isConnected():
        # Terminate the current trial first if the task terminated prematurely
        error = el_tracker.isRecording()
        if error == pylink.TRIAL_OK:
            abort_trial()

        # Put tracker in Offline mode
        el_tracker.setOfflineMode()

        # Clear the Host PC screen and wait for 500 ms
        el_tracker.sendCommand('clear_screen 0')
        pylink.msecDelay(500)

        # Close the edf data file on the Host
        el_tracker.closeDataFile()

        # Show a file transfer message on the screen
        msg = 'EDF data is transferring from EyeLink Host PC...'
        show_msg(win, msg, genv, wait_for_keypress=False)

        # Download the EDF data file from the Host PC to a local data folder
        # parameters: source_file_on_the_host, destination_file_on_local_drive
        local_edf = os.path.join(session_folder, edf_file)
        try:
            el_tracker.receiveDataFile(edf_file, local_edf)
        except RuntimeError as error:
            print('ERROR:', error)

        # Close the link to the tracker.
        el_tracker.close()

    # close the PsychoPy window
    win.close()

    # quit PsychoPy
    core.quit()
    sys.exit()


def abort_trial():
    """Ends recording """

    el_tracker = pylink.getEYELINK()

    # Stop recording
    if el_tracker.isRecording():
        # add 100 ms to catch final trial events
        pylink.pumpDelay(100)
        el_tracker.stopRecording()

    # clear the screen
    clear_screen(win)
    # Send a message to clear the Data Viewer screen
    bgcolor_RGB = (116, 116, 116)
    el_tracker.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)

    # send a message to mark trial end
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_ERROR)

    return pylink.TRIAL_ERROR


def run_trial(trial_params, trial_index, scan_clock, contrast_data, win, fixation, grating):
    parameters = trial_params[str(trial_index)]

    # get a reference to the currently active EyeLink connection
    el_tracker = pylink.getEYELINK()

    # send a "TRIALID" message to mark the start of a trial
    el_tracker.sendMessage('TRIALID %d' % trial_index)

    # For illustration purpose,
    # send interest area messages to record in the EDF data file
    # here we draw a rectangular IA, for illustration purposes
    # format: !V IAREA RECTANGLE <id> <left> <top> <right> <bottom> [label]
    # for all supported interest area commands, see the Data Viewer Manual,
    # "Protocol for EyeLink Data to Viewer Integration"
    scn_width, scn_height = win.size
    left = int(scn_width/2.0) - 50
    top = int(scn_height/2.0) - 50
    right = int(scn_width/2.0) + 50
    bottom = int(scn_height/2.0) + 50
    ia_pars = (1, left, top, right, bottom, 'screen_center')
    el_tracker.sendMessage('!V IAREA RECTANGLE %d %d %d %d %d %s' % ia_pars)

    grating.pos = parameters['gratingPos']
    grating.ori = parameters['gratingOri']
    decrements = parameters['decrements']
    contrast = parameters['contrast']
    response_periods = parameters['response_periods']
    
    # ITI        
    fixation.mask = 'cross'
    fixation.color = 'black'
    fixation.size = 0.5

    fixation.draw()
    win.flip()

    iti_onsetTime = scan_clock.getTime()
    while scan_clock.getTime() - iti_onsetTime <= parameters["ITIDur"]:
        # check for keyboard events
        for keycode, modifier in event.getKeys(modifiers=True):
            # Terminate the task if Ctrl-c
            if keycode == 'c' and (modifier['ctrl'] is True):
                el_tracker.sendMessage('terminated_by_user')
                terminate_task()
                
    # Stimulus
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
    stim_onsetTime = scan_clock.getTime()
    while scan_clock.getTime() - stim_onsetTime <= parameters["trialDuration"]:
        t = (scan_clock.getTime() - stim_onsetTime) * 1000 

         # If trial time is in any decrement range, 
        if ((decrements[:, 0] <=  t) & (t < decrements[:, 1])).any():
            # Decrease contrast
            grating.setPhase(0.05, '+')
            
            if not lastContrastTimeSet:
                grating.contrast = parameters['contrast']
                lastContrastTime = t
                lastContrastTimeSet = True
                lastContrast = grating.contrast
                nContrasts += 1
                
        else:
            # Maintain contrast
            grating.setPhase(0.05, '+')
            
            if lastContrastTimeSet:
                grating.contrast = 1
                lastContrastTimeSet = False
                lastContrast = 1
            
        grating.draw()
        fixation.draw()
        win.flip()
        
        for keycode, modifier in event.getKeys(modifiers=True):
            # Terminate the task if Ctrl-c
            if keycode == 'c' and (modifier['ctrl'] is True):
                el_tracker.sendMessage('terminated_by_user')
                terminate_task()
            # If 1 was pressed, record response
            if keycode == '1':
                response_times.append(scan_clock.getTime())
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
    
    # Saccade Response
    got_sac = False
    sac_start_time = -1
    SRT = -1  # initialize a variable to store saccadic reaction time (SRT)
    land_err = -1  # landing error of the saccade
    acc = 0  # hit the correct region or not

    event.clearEvents()  # clear all cached events if there are any
    sacc_onsetTime = scan_clock.getTime()
    while scan_clock.getTime() - sacc_onsetTime <= parameters["saccadeDuration"]:
        # abort the current trial if the tracker is no longer recording
        error = el_tracker.isRecording()
        if error is not pylink.TRIAL_OK:
            el_tracker.sendMessage('tracker_disconnected')
            abort_trial()
            return error

        # check for keyboard events
        for keycode, modifier in event.getKeys(modifiers=True):
            # Terminate the task if Ctrl-c
            if keycode == 'c' and (modifier['ctrl'] is True):
                el_tracker.sendMessage('terminated_by_user')
                terminate_task()
                return pylink.ABORT_EXPT

        # grab the events in the buffer, for more details,
        # see the example script "link_event.py"
        eye_ev = el_tracker.getNextData()
        if (eye_ev is not None) and (eye_ev == pylink.ENDSACC):
            eye_dat = el_tracker.getFloatData()
            if eye_dat.getEye() == eye_used:
                sac_amp = eye_dat.getAmplitude()  # amplitude
                sac_start_time = eye_dat.getStartTime()  # onset time
                sac_end_time = eye_dat.getEndTime()  # offset time
                sac_start_pos = eye_dat.getStartGaze()  # start position
                sac_end_pos = eye_dat.getEndGaze()  # end position

                # a saccade was initiated
                if sac_start_time <= tar_onset_time:
                    sac_start_time = -1
                    pass  # ignore saccades occurred before target onset
                elif hypot(sac_amp[0], sac_amp[1]) > 1.5:
                    # log a message to mark the time at which a saccadic
                    # response occurred; note that, here we are detecting a
                    # saccade END event; the saccade actually occurred some
                    # msecs ago. The following message has an additional
                    # time offset, so Data Viewer knows when exactly the
                    # "saccade_resp" event actually happened
                    offset = int(el_tracker.trackerTime()-sac_start_time)
                    sac_response_msg = '{} saccade_resp'.format(offset)
                    el_tracker.sendMessage(sac_response_msg)
                    SRT = sac_start_time - tar_onset_time

    # send a 'TRIAL_RESULT' message to mark the end of trial, see Data
    # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_OK)
    
    return SRT, land_err, contrast_data


def run_experiment(expParams):
    # Add current time
    expParams['dateStr'] = data.getDateStr()

    # Present parameter dialog
    dlg = gui.DlgFromDict(expParams, title = 'Perception Pilot', fixed = [
        'Screen Distance', 'Screen Width', 'Screen Resolution',
        'dateStr', 'TR', 'volumes', 'skipSync', 'sync', 'iti_list', 
        'nPositions', 'max_decrements', 'Contrast', 'eccentricity', 'trialDuration',
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
    
    # Setup EyeLink & Window
    el_tracker, session_folder = connect_eyelink(expParams)
    edf_file = create_EDF(expParams, el_tracker)
    el_tracker = configure_eyelink(expParams, el_tracker)
    mon, win, genv = setup_graphics(expParams, el_tracker)
    
    # Create stimuli
    grating = visual.GratingStim(
        win, sf=1, size=3, mask='gauss', maskParams = {'sd': 5},
        pos=[-4,0], ori=0, units = 'deg')
    fixation = visual.GratingStim(
        win, color=-1, colorSpace='rgb', tex=None, mask='cross', size=0.5)
        
    
    # Display instructions
    if expParams['saccadeType'] == 'Saccade':
        instructions = "[PARTICIPANT] Press 1 when you detect a change in contrast. At the end of each trial, make a saccade"
    elif expParams['saccadeInput'] == 'No Saccade':
        instructions = "[PARTICIPANT] Press 1 when you detect a change in contrast. At the end of each trial, DO NOT make a saccade"
    msg1 = visual.TextStim(win, pos=[0, +5], text='[OPERATOR] Hit 0 key when participant is ready')
    msg2 = visual.TextStim(win, pos=[0, -5], text=instructions)

    msg1.draw()
    msg2.draw()
    fixation.draw()
    win.flip()

    # Wait for a response
    event.waitKeys(keyList=['0'])

    # Start timing
    scan_clock = core.Clock()
    globalClock = core.Clock()
    
    el_tracker.setOfflineMode()

    # Start recording, at the beginning of a new run
    # arguments: sample_to_file, events_to_file, sample_over_link,
    # event_over_link (1-yes, 0-no)
    try:
        el_tracker.startRecording(1, 1, 1, 1)
    except RuntimeError as error:
        print("ERROR:", error)
        terminate_task()

    # Allocate some time for the tracker to cache some samples
    pylink.pumpDelay(100)
    
    # fMRI Sync Trigger
    vol = launchScan(
        win,
        settings = {'TR': expParams['TR'], 'volumes': expParams['volumes'], 'skip': expParams['skipSync']},
        globalClock = globalClock,
        mode = expParams['expMode'],
        wait_msg = "Waiting for Sync Pulse"
    )
    
    # record a message to mark the start of scanning
    el_tracker.sendMessage('Scan_start_Run_%d' % (expParams['Run']))
    
    # reset the global clock to compare stimulus timing
    # to time 0 to make sure each trial is 6-sec long
    # this is known as "non-slip timing"
    scan_clock.reset()

    # Trial loop
    trial_index = 0
    for trial in range(expParams['nPositions']):
        SRT, land_err, contrast_data = run_trial(trialParams, trial_index, scan_clock, contrast_data, win, fixation, grating)
        trial_index += 1

    # send a message to mark the end of a run
    el_tracker.sendMessage('Scan_end_Run_%d' % (expParams['Run']))

    # clear the screen
    clear_screen(win, genv)

    # stop recording; add 100 msec to catch final events before stopping
    pylink.pumpDelay(100)
    el_tracker.stopRecording()
    
    # End
    terminate_task(win, session_folder, edf_file, genv)
    


run_experiment(expParams)