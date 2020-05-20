import multiprocessing
import os
import csv
from sys import *
from math import *
import subprocess
import numpy as np
import pandas as pd
import timeit
import shutil

def os_sep():
    if os.name == 'posix':
        os_fold = '/'
    else:
        os_fold = "\\"
    return(os_fold)

def os_sep():
    if os.name == 'posix':
        os_fold = '/'
    else:
        os_fold = "\\"
    return(os_fold)

# Run cequal
def run_cequal(model_folder, fail_time):
    global wd
    os_fold = os_sep()
    if os.name == 'posix':
        subprocess.call(['../CE-QUAL-W2-Linux/w2_exe_linux', model_folder], timeout=fail_time)
    elif os.name == 'nt':
        subprocess.call(['w2_v4_64.exe', model_folder], timeout=fail_time)
    else:
        print("unknown OS")

# Write to control file
# **Cannot handle numbers larger than 8 orders of magnitude
def write_control(parameter, parameter_name, file_path, index):
    os_fold = os_sep()
    parameter = '%s' % float('%.5g' % parameter)
    parameter = parameter.rjust(7)
    control = open("{}w2_con.npt".format(file_path + os_fold), "r+")
    control_copy = control.readlines()
    match = np.array([line.find(parameter_name) for line in control_copy])
    char_len = len(parameter_name)
    location = [[row+1,position+char_len] for row,position in enumerate(match) if position > 0]
    old_str = control_copy[location[index][0]][location[index][1]-7:location[index][1]]
    new_str = control_copy[location[index][0]][:location[index][1]-7] + parameter + control_copy[location[index][0]][location[index][1]:]
    control_copy[location[index][0]] = new_str
    # write control file, overwrite existing
    new_control = open("{}w2_con.npt".format(file_path + os_fold), "w+")
    new_control.writelines(control_copy)
    new_control.close()

# Update wsc file from existing template
def write_wsc(wsc_vals, file_path, segs):
    os_fold = os_sep()
    wsc= pd.read_csv("{}{}".format(file_path, os_fold + "wsc.csv"), header=None)
    sim_days = wsc.loc[4,0]
    branch = [len(section) for section in segs]
    wsc.loc[2,1:] = list(np.concatenate(segs))
    index = 0
    wsc.loc[3] = [1] + ['0.625']*branch[0] + [wsc_vals[index]]*branch[1] + [wsc_vals[index+1]]*branch[2] +[wsc_vals[index+2]]*branch[3]
    wsc.loc[4] = [sim_days] + ['0.625']*branch[0] + [wsc_vals[index]]*branch[1] + [wsc_vals[index+1]]*branch[2] +[wsc_vals[index+2]]*branch[3]
    wsc.to_csv("{}{}".format(file_path, os_fold + "wsc.csv"), header=False, index=False)

wd = os.getcwd()
os_fold = os_sep()
year = '2018'
model_folder = wd + os_fold + year + os_fold + 'optimum'
data_path = wd + os_fold + "data" + os_fold + year
segs = [np.r_[1:10], np.r_[10:28], np.r_[46:51], np.r_[28:46, 51:63]]
# 2017
# params = [0.5131594238654958, 109.95578187000578, 0.9623762126381636, 0.6490257771192711, 0.7042075741476852]
# 2018
# params = [0.6357944, 110.3993, 0.5038029, 0.6571143, 0.7192591]
# 2009-2018
params = [0.341693421659008,110.59999668931,0.981545621908465,0.914952827642663,0.568234559457425]


write_control(params[0], "EXH2O", model_folder, 0)
write_control(params[1], "ESTR", model_folder, 0)
write_wsc(params[2:5], model_folder, segs)
# run_cequal(model_folder, 1800)
