
# Overview
CEQUAL-Opt is a framework that allows for the automatic calibration of CE-QUAL-W2 Wachusett Reservoir water quality models. It utilizes the [jMetalpy](https://github.com/jMetal/jMetalPy) package in Python and is currently set up for calibration using the [NSGA-II](https://www.iitk.ac.in/kangal/Deb_NSGA-II.pdf) multi-objective genetic algorithm. In its current state, it has the capacity to calibrate water quality models for temperature and specific conductivity but may be extended for other water quality constituents and eventually, other CE-QUAL-W2 models, as necessary.

<p align="center">
  <b> Plan View of Model Grid</b>
</p>
<p align="center">
  <img src="figures/CEQUAL_GRIDS.png" width="400" height="400">
</p>

## Getting Started
**Download Instructions**
1. Clone/download this repository
2. Download jMetalPy using the pip command:
```
  pip install jmetalpy
```
3. Install other package dependencies when prompted

**Notes**

Given the computational demand associated with calibrating CE-QUAL-W2 models, use of a cluster (e.g. UMass GHPCC) is recommended for multi-year models. A separate Linux-based executable for CE-QUAL-W2 must be compiled to run on Linux machines. The Linux compilation procedure and documentation is provided in the [WQDSS](https://github.com/WQDSS/CE-QUAL-W2-Linux) GitHub repository.

## Usage
Several modifications necessary for model calibration are summarized below.
#### Input Files
1. Update the model executable (e.g. "w2_v4_64.exe") in the main directory to the desired version
2. Add the baseline model files into the following directory:
  ```
    models/[year]/init

  ```
3. Add observations from the Basin North and Cosgrove Intake into the following directory. Make sure filenames and data format are in agreement with previous model years.
```
  data/[year]
```
#### Calibration
The algorithm objectives are set to minimize the root mean square error (RMSE) between simulated and observed temperature (T) and specific conductivity (SC) at the Basin North (BN) and Cosgrove Intake (CI).  The model parameters (decision variables) selected for calibration and corresponding bounds are defined in the [calib.py](Python/calib.py) file under the `def __init__` problem initiation and summarized below:


Table 1. Summary of model parameters used in calibration

|Parameter Name  | Parameter Abbr. | Lower Bound | Upper Bound |
| :-: | :-: | :-: | :-: |
| Light extinction coefficient | EXH2O | 0.25 | 0.65 |
| CI withdrawal elevation | ESTR | 104.3 | 110.6 |
| WSC Thomas | WSC | 0.5 | 1.0 |
| WSC South | WSC | 0.5 | 1.0 |
| WSC North  | WSC | 0.5 | 1.0 |

 The algorithm will seek to find the optimal parameter sets from Table 1 that form the pareto front. For calibration of model years prior to 2016, it is recommended that the SC objective at the CI be dropped due to probe measurement errors.
##### 'evaluate' function
Several script parameters directly following the 'evaluate' function definition need to be modified prior to calibration. These include:
1. year - Model year(s) as a string
2. trial_name - Output file ID as a string
3. fail_time - Maximum allowable model run-time (seconds)
4. regions - WSC delineation by segment. Use the wsc_viewer.py script to visualize the delineation
5. debug - If turned on, will raise an error if model run-time exceeds fail_time

For example, the following code...

```
def evaluate(self, solution: FloatSolution) -> FloatSolution:
  year = "2018"
  trial_name = "2018_debug"
  fail_time = 400
  regions = {
      'Default': np.r_[1:10],
      'Thomas': np.r_[10:28],
      'South': np.r_[46:51],
      'North': np.r_[28:46, 51:63]
  }
  debug = "off"
```
... will set up a calibration for the 2018 model located here: `models/2018/init`, using observed data located here `data/2018/`, with a maximum allowable model run time of 400 seconds and the following WSC delineation:

<p align="center">
  <b> WSC Delineation </b>
</p>
<p align="center">
  <img src="figures/WSC_example.png" width="300" height="300">
</p>

##### Other
Further changes to the script will be required to introduce additional objective functions (e.g. RMSE for dissolved oxygen at BN) or decision variables (other CE-QUAL-W2 parameters).
#### Output Files
Several output files will be generated following a successful calibration:

1. Pareto front decision variables (CE-QUAL parameters) in order defined by script:
  ```
    results/[year]/VAR_NSGAII_[trial_name].txt'
  ```
2. Pareto front objectives (RMSE values) in order defined by script:
  ```
    results/[year]/OBJ_NSGAII_[trial_name].txt
  ```
3. Record of all decision variables and objectives tested during optimization:
  ```
    results/[year]/[trial_name]_all_runs.txt
  ```
4. The 'best' performing model from each iteration is recorded from the most recent calibration run and output to the [jmetalpy.log](Python/jmetalpy.log) file

### Notes
* The calib.py script should be called using a terminal directly from the calib.py directory `./Python`
* It is recommended that the Anaconda distribution be used for Python package management
* jMetalPy is an actively managed package. Frequent updates may occur that change the naming/placement of several function locations
