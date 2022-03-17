import os
import sys
import pylink


#os.umask(0)
#filepath = "/Users/robwoodry/Documents/Research/Interstellar/data/sub-003"
#os.makedirs(filepath, mode = 0o777)

subject = 1
run = 1
ip = '100.1.1.1'

data_path = "/Users/robwoodry/Documents/Research/Interstellar/data/"
et = pylink.EyeLink(ip)
et.open()

def receiveEDF(eyetracker, data_path, subject, run):
    edf_file = "Is%02dr%02d.edf" % (subject, run)
    local_edf = os.path.join(data_path, "sub-%03d" % subject, edf_file)
    print(edf_file, local_edf)

    eyetracker.receiveDataFile(edf_file, local_edf)
    
receiveEDF(et, data_path, subject, run)
et.close()

