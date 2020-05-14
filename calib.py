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
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.util.observer import ProgressBarObserver, BasicObserver
from jmetal.util.evaluator import MultiprocessEvaluator
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.util.comparator import DominanceComparator

"""
5 parameters
EXH2O, CI ESTR, WSC THOMAS, WSC SOUTH, WSC NORTH
4 objectives
Minimize Temp and SC RMSE at Basin North and Cosgrove Intake
*Ignore Conductivity at CI due to probe errors for calibration prior to 2016
"""

def os_sep():
    if os.name == 'posix':
        os_fold = '/'
    else:
        os_fold = "\\"
    return(os_fold)

wd = os.getcwd()
os_fold = os_sep()

# The bulk of the code is provided in the definition of the class 'cequal'
class cequal(FloatProblem):
    def __init__(self, number_of_variables: int = 5, number_of_objectives=4):
        """ param number_of_variables: number of decision variables of the problem.
        """
        super(cequal, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE] * number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(number_of_objectives)]
        self.lower_bound = [0.25, 104.3, 0.5, 0.5, 0.5]
        self.upper_bound = [0.65, 110.6, 1, 1, 1]

    def os_sep(self):
        if os.name == 'posix':
            os_fold = '/'
        else:
            os_fold = "\\"
        return(os_fold)

    # Run cequal with windows (nt) or linux (posix)
    def run_cequal(self, model_folder, fail_time):
        global wd
        os_fold = self.os_sep()
        if os.name == 'posix':
            subprocess.call(['../CE-QUAL-W2-Linux/w2_exe_linux', model_folder], timeout=fail_time)
        elif os.name == 'nt':
            subprocess.call(['w2_v4_64.exe', model_folder], timeout=fail_time)
        else:
            print("unknown OS")

    # Write/update parameters to control/input files
    # searches for parameter name in control (text) file then rewrites corresponding
    # parameter value beneath parameter name
    # cannot handle numbers larger than 8 orders of magnitude
    def write_control(self, parameter, parameter_name, file_path, index):
        os_fold = os_sep()
        parameter = '%s' % float('%.5g' % parameter) #5 sig figs
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

    # Update WSC from existing input files
    def write_wsc(self, wsc_vals, file_path, segs):
        os_fold = os_sep()
        wsc= pd.read_csv("{}{}".format(file_path, os_fold + "wsc.csv"), header=None)
        sim_days = wsc.loc[4,0]
        branch = [len(section) for section in segs]
        wsc.loc[2,1:] = list(np.concatenate(segs))
        index = 0
        wsc.loc[3] = [1] + ['0.625']*branch[0] + [wsc_vals[index]]*branch[1] + [wsc_vals[index+1]]*branch[2] +[wsc_vals[index+2]]*branch[3]
        wsc.loc[4] = [sim_days] + ['0.625']*branch[0] + [wsc_vals[index]]*branch[1] + [wsc_vals[index+1]]*branch[2] +[wsc_vals[index+2]]*branch[3]
        wsc.to_csv("{}{}".format(file_path, os_fold + "wsc.csv"), header=False, index=False)

    # Convert reservoir elevation to model layer
    def assign_layer(self, df):
        df['Layer'] = None
        df.loc[df['Elevation'] > 120.7, 'Layer']= 1
        df.loc[(df['Elevation'] <= 120.7) & (df['Elevation'] > 120.2), 'Layer']= 2
        df.loc[(df['Elevation'] <= 120.2) & (df['Elevation'] > 119.7), 'Layer']= 3
        df.loc[(df['Elevation'] <= 119.7) & (df['Elevation'] > 119.2), 'Layer']= 4
        df.loc[(df['Elevation'] <= 119.2) & (df['Elevation'] > 118.7), 'Layer']= 5
        df.loc[(df['Elevation'] <= 118.7) & (df['Elevation'] > 118.2), 'Layer']= 6
        df.loc[(df['Elevation'] <= 118.2) & (df['Elevation'] > 117.7), 'Layer']= 7
        df.loc[(df['Elevation'] <= 117.7) & (df['Elevation'] > 117.2), 'Layer']= 8
        df.loc[(df['Elevation'] <= 117.2) & (df['Elevation'] > 116.7), 'Layer']= 9
        df.loc[(df['Elevation'] <= 116.7) & (df['Elevation'] > 116.2), 'Layer']= 10
        df.loc[(df['Elevation'] <= 116.2) & (df['Elevation'] > 115.7), 'Layer']= 11
        df.loc[(df['Elevation'] <= 115.7) & (df['Elevation'] > 115.2), 'Layer']= 12
        df.loc[(df['Elevation'] <= 115.2) & (df['Elevation'] > 114.7), 'Layer']= 13
        df.loc[(df['Elevation'] <= 114.7) & (df['Elevation'] > 114.2), 'Layer']= 14
        df.loc[(df['Elevation'] <= 114.2) & (df['Elevation'] > 113.7), 'Layer']= 15
        df.loc[(df['Elevation'] <= 113.7) & (df['Elevation'] > 113.2), 'Layer']= 16
        df.loc[(df['Elevation'] <= 113.2) & (df['Elevation'] > 112.7), 'Layer']= 17
        df.loc[(df['Elevation'] <= 112.7) & (df['Elevation'] > 112.2), 'Layer']= 18
        df.loc[(df['Elevation'] <= 112.2) & (df['Elevation'] > 111.7), 'Layer']= 19
        df.loc[(df['Elevation'] <= 111.7) & (df['Elevation'] > 111.2), 'Layer']= 20
        df.loc[(df['Elevation'] <= 111.2) & (df['Elevation'] > 110.7), 'Layer']= 21
        df.loc[(df['Elevation'] <= 110.7) & (df['Elevation'] > 110.2), 'Layer']= 22
        df.loc[(df['Elevation'] <= 110.2) & (df['Elevation'] > 109.7), 'Layer']= 23
        df.loc[(df['Elevation'] <= 109.7) & (df['Elevation'] > 109.2), 'Layer']= 24
        df.loc[(df['Elevation'] <= 109.2) & (df['Elevation'] > 108.7), 'Layer']= 25
        df.loc[(df['Elevation'] <= 108.7) & (df['Elevation'] > 108.2), 'Layer']= 26
        df.loc[(df['Elevation'] <= 108.2) & (df['Elevation'] > 107.7), 'Layer']= 27
        df.loc[(df['Elevation'] <= 107.7) & (df['Elevation'] > 107.2), 'Layer']= 28
        df.loc[(df['Elevation'] <= 107.2) & (df['Elevation'] > 106.7), 'Layer']= 29
        df.loc[(df['Elevation'] <= 106.7) & (df['Elevation'] > 106.2), 'Layer']= 30
        df.loc[(df['Elevation'] <= 106.2) & (df['Elevation'] > 105.7), 'Layer']= 31
        df.loc[(df['Elevation'] <= 105.7) & (df['Elevation'] > 104.95), 'Layer']= 32
        df.loc[(df['Elevation'] <= 104.95) & (df['Elevation'] > 104.2), 'Layer']= 33
        df.loc[(df['Elevation'] <= 104.2) & (df['Elevation'] > 102.7), 'Layer']= 34
        df.loc[(df['Elevation'] <= 102.7) & (df['Elevation'] > 101.2), 'Layer']= 35
        df.loc[(df['Elevation'] <= 101.2) & (df['Elevation'] > 99.7 ), 'Layer']=  36
        df.loc[(df['Elevation'] <= 99.7 ) & (df['Elevation'] > 98.2 ), 'Layer']=  37
        df.loc[(df['Elevation'] <= 98.2 ) & (df['Elevation'] > 96.7 ), 'Layer']=  38
        df.loc[(df['Elevation'] <= 96.7 ) & (df['Elevation'] > 95.2 ), 'Layer']=  39
        df.loc[(df['Elevation'] <= 95.2 ) & (df['Elevation'] > 93.7 ), 'Layer']=  40
        df.loc[(df['Elevation'] <= 93.7 ) & (df['Elevation'] > 92.2 ), 'Layer']=  41
        df.loc[(df['Elevation'] <= 92.2 ) & (df['Elevation'] > 90.7 ), 'Layer']=  42
        df.loc[(df['Elevation'] <= 90.7 ) & (df['Elevation'] > 89.2 ), 'Layer']=  43
        df.loc[(df['Elevation'] <= 89.2 ) & (df['Elevation'] > 87.7 ), 'Layer']=  44
        df.loc[(df['Elevation'] <= 87.7 ) & (df['Elevation'] > 86.2 ), 'Layer']= 45
        df.loc[df['Elevation'] <= 86.2, 'Layer']= 9999
        return(df)

    # Error Metrics
    def RMSE(self, model, meas):
        return(((model - meas) ** 2).mean() ** .5)
    def AME(self, model, meas):
        return(abs(model-meas).mean())
    def ME(self, model, meas):
        return((model-meas).mean())

    # BASIN NORTH
    def Error_BN(self, model_folder, data_folder, year):
        os_fold = self.os_sep()
    # Measured Data
        MeasBN = pd.read_csv("{}{}".format(data_folder, os_fold + year + "BN_Measured.csv"), index_col=0)
        MeasBN = MeasBN.rename(columns = {'Julian_day': "Day"})
        MeasBN = self.assign_layer(MeasBN)
    # Modeled Results
        ModelBN = pd.read_csv("{}{}spr.opt".format(model_folder, os_fold))
        ModelBN = ModelBN[['Constituent', "Julian_day", 'Elevation', "Seg_42 "]]
        ModelBN.columns = ['Constituent', "Day", "Elevation", "Seg_42"]
        ModelBN.Day = ModelBN.Day.apply(np.floor).apply(int)
        ModelBN = self.assign_layer(ModelBN)
        ModelBN.Constituent= ModelBN.Constituent.str.strip()
        ModelBN.loc[ModelBN['Constituent'] == 'TDS', 'Seg_42'] = ModelBN.loc[ModelBN['Constituent'] == 'TDS', 'Seg_42'] / 0.6
        ModelBN.loc[ModelBN['Constituent'] == 'TDS', 'Constituent'] = 'Specific Conductivity'
    # Merged Set
        BN = pd.merge(ModelBN, MeasBN, on = ["Layer", "Day", "Constituent"])
        BN_temp = BN.loc[BN['Constituent'] == "Temperature"]
        BN_cond = BN.loc[BN['Constituent'] == "Specific Conductivity"]
    # Error calculations
        ErrorBN = {"TempRMSE" : self.RMSE(BN_temp['Seg_42'], BN_temp['Value']),
                   "TempAME" : self.AME(BN_temp['Seg_42'], BN_temp['Value']),
                   "TempME" : self.ME(BN_temp['Seg_42'], BN_temp['Value']),
                   "CondRMSE" : self.RMSE(BN_cond['Seg_42'], BN_cond['Value']),
                   "CondAME" : self.AME(BN_cond['Seg_42'], BN_cond['Value']),
                   "CondME" : self.ME(BN_cond['Seg_42'], BN_cond['Value'])}
        return(ErrorBN)

    # COSGROVE INTAKE
    def Error_CI(self, model_folder, data_folder, year):
        os_fold = self.os_sep()
        # Measured Data
        MeasCI = pd.read_csv("{}{}".format(data_folder, os_fold + year + "CI_Measured_Corrected.csv"), index_col=0)
        MeasCI = MeasCI.rename(columns = {"JDAY": "Day"})
        # Modeled Results
        RawCI = {
            'Temp' : pd.read_csv("{}{}".format(model_folder, os_fold + "two_str1_seg44.opt"), skiprows=3, names=['Day', "Temperature", "Temp_QSTR"],
                             index_col = False),
            'TDS' : pd.read_csv("{}{}".format(model_folder, os_fold + "cwo_str1_seg44.opt"), skiprows=3, names = ['Day', 'Specific Conductivity', 'NA'],
                                index_col = False)
        }

        ModelCI = pd.merge(RawCI['Temp'], RawCI['TDS'], on='Day')
        ModelCI = ModelCI[['Day', 'Temperature', 'Specific Conductivity']]
        ModelCI.Day = ModelCI.Day.apply(np.floor).apply(int)
        ModelCI['Specific Conductivity'] = ModelCI['Specific Conductivity'] / 0.6
        # Merged Set
        CI = pd.merge(ModelCI, MeasCI, on = 'Day')
        # Error calculations
        ErrorCI = {"TempRMSE" : self.RMSE(CI['Temperature_x'], CI['Temperature_y']),
                   "TempAME" : self.AME(CI['Temperature_x'], CI['Temperature_y']),
                   "TempME" : self.ME(CI['Temperature_x'], CI['Temperature_y']),
                   "CondRMSE" : self.RMSE(CI['Specific Conductivity_x'], CI['Specific Conductivity_y']),
                   "CondAME" : self.AME(CI['Specific Conductivity_x'], CI['Specific Conductivity_y']),
                   "CondME" : self.ME(CI['Specific Conductivity_x'], CI['Specific Conductivity_y'])}

        return(ErrorCI)

    # Write all iterations to file
    def write_full_outputs(self, vars, all_objs, trial_name):
        wd = os.getcwd()
        os_fold = self.os_sep()
        f = open(wd + os_fold + 'results' + os_fold + 'cequal' + trial_name + '_all_runs.txt', 'a')
        for item in vars:
            f.write("%s," % item)
        for item in all_objs:
            f.write("%s," % item)
        f.write("\n")
        f.close()

    # Borg CEQUAL function
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        global wd

        ###
        debug = 'off' # see below for description
        year = '2018' # model year
        trial_name = "2018_trial1" #output file name ID
        fail_time = 400 # maximum model run time (seconds)
        ###

        thread = multiprocessing.Process()
        process_id = str(thread.name)
        process_id = process_id.replace("-", "_")
        process_id = process_id.replace(":", "_")
        os_fold = self.os_sep()
        init = wd + os_fold + year + os_fold + 'init'
        model_folder = wd + os_fold + year + os_fold + str(process_id)
        data_path = wd + os_fold + 'data' + os_fold + year
        segs = [np.r_[1:10], np.r_[10:28], np.r_[46:51], np.r_[28:46, 51:63]]
        dec_vars = solution.variables[0:5]
        wsc_vals = dec_vars[2:5]

        files = os.listdir(init)
        if os.path.exists(model_folder):
            shutil.rmtree(model_folder)
        if os.path.exists(model_folder) == False:
            os.mkdir(model_folder)
        for file in files:
            shutil.copy(init + os_fold + file, model_folder)

        self.write_control(dec_vars[0], "EXH2O", model_folder, 0)
        self.write_control(dec_vars[1], "ESTR", model_folder, 0)
        self.write_wsc(wsc_vals, model_folder, segs)
        print([process_id, dec_vars])

        # debug on will raise error when model fails
        if debug == 'on':
            self.run_cequal(model_folder, fail_time)
            error_bn = self.Error_BN(model_folder, data_path, year)
            error_ci = self.Error_CI(model_folder, data_path, year)
            solution.objectives = ([error_bn['TempRMSE'],
                                    error_bn['CondRMSE'],
                                    error_ci['TempRMSE'],
                                    error_ci['CondRMSE']])
            all_objs = ([error_bn['TempRMSE'],
                        error_bn['CondRMSE'],
                        error_ci['TempRMSE'],
                        error_ci['CondRMSE']])
        if debug == 'off':
            try:
                self.run_cequal(model_folder, fail_time)
                error_bn = self.Error_BN(model_folder, data_path, year)
                error_ci = self.Error_CI(model_folder, data_path, year)
                solution.objectives = ([error_bn['TempRMSE'],
                                        error_bn['CondRMSE'],
                                        error_ci['TempRMSE'],
                                        error_ci['CondRMSE']])
                all_objs = ([error_bn['TempRMSE'],
                            error_bn['CondRMSE'],
                            error_ci['TempRMSE'],
                            error_ci['CondRMSE']])
            except:
                print("bad model")
                solution.objectives = [1e9] * self.number_of_objectives
                all_objs = [1e9] * 4

        self.write_full_outputs(solution.variables, all_objs, trial_name)
        print(solution.objectives)
        shutil.rmtree(model_folder)
        return solution

    def get_name(self):
        return 'cequal'

def run_optimization(max_eval, num_nodes, pop_size, offspring, trial_name):
    multiprocessing.freeze_support()
    problem = cequal()
    max_evaluations = max_eval

    algorithm = NSGAII(
        population_evaluator=MultiprocessEvaluator(num_nodes),
        problem=problem,
        population_size=pop_size,
        offspring_population_size=offspring,
        mutation=PolynomialMutation(probability= 0.2, distribution_index=20),
        crossover=SBXCrossover(probability=0.8, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        dominance_comparator=DominanceComparator()
    )

    algorithm.observable.register(ProgressBarObserver(max=max_evaluations))
    algorithm.observable.register(observer=BasicObserver())

    k = open(wd + os_fold + 'results' + os_fold + algorithm.get_name() + trial_name + '_all_runs.txt', 'w')
    k.close()

    f = open(wd + os_fold + 'results' + os_fold + trial_name + '_observer.txt', 'w')
    f.write("Evaluations\tBest_Fitness\n")
    f.close()

    algorithm.run()
    front = algorithm.get_result()

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time/3600))

    print_function_values_to_file(front, 'results' + os_fold + 'OBJ_' + algorithm.get_name() + "_" + problem.get_name() + trial_name + '.set')
    print_variables_to_file(front, 'results' + os_fold + 'VAR_' + algorithm.get_name() + "_" + problem.get_name() + trial_name + '.set')

if __name__ == '__main__':
    ## Make sure following model parameters are defined in 'evaluate' function
    # trial_name = "2018_trial1"
    # run_optimization(max_eval = 12, num_nodes = 4, pop_size = 12, offspring = 12, trial_name = trial_name)
    # run_optimization(max_eval = 2000, num_nodes = 4, pop_size = 20, offspring = 20, trial_name = trial_name)
    # run_optimization(max_eval = 6400, num_nodes = 32, pop_size = 32, offspring = 32, trial_name = trial_name)


# helpful code for debugging purposes
    # year = '2018'
    # model_folder = wd + os_fold + year + os_fold + 'optimum'
    # data_path = wd + os_fold + "data" + os_fold + year
    # fail_time = 300
    # problem = cequal()
    # segs = [np.r_[1:10], np.r_[10:28], np.r_[46:51], np.r_[28:46, 51:63]]
    # problem.write_control(0.25, "EXH2O", model_folder, 0)
    # problem.write_control(104.3, "ESTR", model_folder, 0)
    # problem.write_wsc([0.625, 0.625, 0.625], model_folder, segs)
    # problem.run_cequal(model_folder, fail_time)
    # error_bn = problem.Error_BN(model_folder, data_path, year)
    # error_ci = problem.Error_CI(model_folder, data_path, year)
    # print([error_bn, error_ci])
    # subprocess.call('/home/js17a/CE-QUAL-W2-Linux/w2_exe_linux /home/js17a/jmet/2017/init')
    # subprocess.call('/home/js17a/CE-QUAL-W2-Linux/w2_exe_linux /home/js17a/jmet/2017/init', 150)
    # subprocess.call(['./CE-QUAL-W2-Linux/w2_exe_linux', model_folder])
    # subprocess.call(['w2_v4_64.exe', model_folder])
