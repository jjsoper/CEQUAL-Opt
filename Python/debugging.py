from calib import *
import os

# os.chdir('..')
wd = os.getcwd()
os_fold = os_sep()

if __name__ == '__main__':
# helpful code for debugging purposes
    year = '2018'
    model_folder = wd + os_fold + 'models' + os_fold + year + os_fold + 'init'
    data_path = wd + os_fold + "data" + os_fold + year
    fail_time = 300
    regions = {
        'Default': np.r_[1:10],
        'Thomas': np.r_[10:28],
        'South': np.r_[46:51],
        'North': np.r_[28:46, 51:63]
    }
    problem = cequal()
    problem.write_control(0.25, "EXH2O", model_folder, 0)
    problem.write_control(104.3, "ESTR", model_folder, 0)
    problem.write_wsc([0.625, 0.625, 0.625], regions, model_folder)
    problem.run_cequal(model_folder, fail_time)
    error_bn = problem.Error_BN(model_folder, data_path, year)
    error_ci = problem.Error_CI(model_folder, data_path, year)
    print([error_bn, error_ci])
    # subprocess.call('/home/js17a/CE-QUAL-W2-Linux/w2_exe_linux /home/js17a/jmet/2017/init')
    # subprocess.call('/home/js17a/CE-QUAL-W2-Linux/w2_exe_linux /home/js17a/jmet/2017/init', 150)
    # subprocess.call(['./CE-QUAL-W2-Linux/w2_exe_linux', model_folder])
    # subprocess.call(['w2_v4_64.exe', model_folder])
