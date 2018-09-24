"""
This code estimates heart rate and computes accuracy statistics
(compared to reference heart rate measurements) for the PPG-only 
algorithm and the PPG-area algorithm. 

The input file HREstimates.csv, which may be the output of
colorAreaHeartRate.py, should have data in 8 columns:
subject_number
video_number
FFT_PPG_HR: Heart rate estimate (bpm) using FFT to estimate
    the spectrum and only including the green color signal.
FFT_area_HR: Heart rate estimate (bpm) using FFT to estimate
    the spectrum and only including the ellipse area signal.
LASSO_PPG_HR: Heart rate estimate (bpm) using Group LASSO to
    estimate the spectrum and only including the green color
    signal.
LASSO_area_HR: Heart rate estimate (bpm) using Group LASSO to
    estimate the spectrum and only including the ellipse area
    signal.
FFT_PPG_area_HR: Heart rate estimate (bpm) using FFT to estimate the
    spectrum of each signal, and taking the elementwise product
    in the frequency domain of the color and area coefficient
    magnitudes.
LASSO_PPG_area_HR: Heart rate estimate (bpm) using Group LASSO to estimate
    the spectrum of each signal, and taking the elementwise
    product in the frequency domain of the color and area 
    coefficient magnitudes.

The input file reference.csv should have data in 3 columns:
subject_number
video_number
reference_HR: The reference heart rate, measured in bpm.

Data in the two input files should be parallel.

Two output figures are produced in the current directory:
BlandAltmanPPG.png: Bland-Altman plot comparing the heart rate
	estimated using PPG-only to the reference heart rate.
BlandAltmanPPGArea.png: Bland-Altman plot comparing the heart
	rate estimated using PPG-area to the reference heart rate.

Run this code using the command:
python heartRateAlgorithmComparison.py HREstimates.csv reference.csv
"""

import numpy as np
import sys
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family':'serif', 'serif':['Computer Modern Roman'], 
                                'monospace': ['Computer Modern Typewriter']})
params = {'text.usetex': True, 'font.weight': 'bold', 'axes.labelweight': 'bold'}
matplotlib.rcParams.update(params)

def plotBlandAltman(measurement, reference, fig_name, algorithm_name):
	mean = np.mean([measurement, reference], axis=0)
	diff = measurement - reference
	md = np.mean(diff)
	sd = np.std(diff)

	fig = plt.figure(figsize=(15,15))
	plt.ylim((-50,50))
	plt.plot(mean, diff, 'bo', markersize=10)
	plt.axhline(md, color='xkcd:dark grey', linestyle='--', linewidth=6)
	plt.axhline(md + 1.96*sd, color='xkcd:dark grey', linestyle='--', linewidth=6)
	plt.axhline(md - 1.96*sd, color='xkcd:dark grey', linestyle='--', linewidth=6)

	print(algorithm_name + " mean diff: ", md)
	print(algorithm_name + " mean diff + 1.96*sd: ", md + 1.96*sd)
	print(algorithm_name + " mean diff - 1.96*sd: ", md - 1.96*sd)
	r = np.corrcoef(measurement, reference)
	r = r[0,1]
	n = np.size(measurement)
	t = r*np.sqrt(n-2)/np.sqrt(1-r*r)
	print(algorithm_name + " r: ", r)
	print(algorithm_name + " n: ", n)
	print(algorithm_name + " t: ", t)

	plt.xlabel('\\textbf{(Arm Cuff HR + ' + algorithm_name + ' HR)/2 (bpm)}', fontsize = 50)
	plt.ylabel('\\textbf{' + algorithm_name + ' HR - Arm Cuff HR (bpm)}', fontsize = 50)
	plt.tick_params(axis='both', which='major', labelsize=40)
	plt.tight_layout()
	fig.savefig(fig_name)
	plt.close('all')

results = np.loadtxt(sys.argv[1])
subject_nums = results[:,0]
FFT_PPG_HR = results[:,2]
LASSO_PPG_HR = results[:,4]
FFT_PPG_area_HR = results[:,6]
LASSO_PPG_area_HR = results[:,7]

reference_HR = np.loadtxt(sys.argv[2])
reference_HR = reference_HR[:,2]

# Separate training and testing data, and compute MSE on 
# training data to do weighted average for testing data
num_subjects = np.size(np.unique(subject_nums))
random.seed(a = 4) # For reproducibility
training_subjects = random.sample(range(1, num_subjects), 
	int(num_subjects/2))
training_indices = np.in1d(subject_nums, training_subjects)
testing_indices = np.in1d(subject_nums, training_subjects, 
	invert=True)

training_FFT_PPG_HR = FFT_PPG_HR[training_indices]
training_LASSO_PPG_HR = LASSO_PPG_HR[training_indices]
testing_FFT_PPG_HR = FFT_PPG_HR[testing_indices]
testing_LASSO_PPG_HR = LASSO_PPG_HR[testing_indices]

training_FFT_PPG_area_HR = FFT_PPG_area_HR[training_indices]
training_LASSO_PPG_area_HR = LASSO_PPG_area_HR[training_indices]
testing_FFT_PPG_area_HR = FFT_PPG_area_HR[testing_indices]
testing_LASSO_PPG_area_HR = LASSO_PPG_area_HR[testing_indices]

training_reference_HR = reference_HR[training_indices]
testing_reference_HR = reference_HR[testing_indices]

# Remove indices where FFT or LASSO rejected the video
good_FFT_PPG_HR = training_FFT_PPG_HR > 0
good_LASSO_PPG_HR = training_LASSO_PPG_HR > 0
training_FFT_PPG_HR_error = training_FFT_PPG_HR[good_FFT_PPG_HR] - training_reference_HR[good_FFT_PPG_HR]
training_LASSO_PPG_HR_error = training_LASSO_PPG_HR[good_LASSO_PPG_HR] - training_reference_HR[good_LASSO_PPG_HR]

good_FFT_PPG_area_HR = training_FFT_PPG_area_HR > 0
good_LASSO_PPG_area_HR = training_LASSO_PPG_area_HR > 0
training_FFT_PPG_area_HR_error = training_FFT_PPG_area_HR[good_FFT_PPG_area_HR] - training_reference_HR[good_FFT_PPG_area_HR]
training_LASSO_PPG_area_HR_error = training_LASSO_PPG_area_HR[good_LASSO_PPG_area_HR] - training_reference_HR[good_LASSO_PPG_area_HR]

training_FFT_PPG_HR_MSE = np.mean(training_FFT_PPG_HR_error**2)
training_LASSO_PPG_HR_MSE = np.mean(training_LASSO_PPG_HR_error**2)

training_FFT_PPG_area_HR_MSE = np.mean(training_FFT_PPG_area_HR_error**2)
training_LASSO_PPG_area_HR_MSE = np.mean(training_LASSO_PPG_area_HR_error**2)

print("training FFT PPG-only MSE: ", training_FFT_PPG_HR_MSE)
print("training LASSO PPG-only MSE: ", training_LASSO_PPG_HR_MSE)

print("training FFT PPG-area MSE: ", training_FFT_PPG_area_HR_MSE)
print("training LASSO PPG-area MSE: ", training_LASSO_PPG_area_HR_MSE)

# Now apply these values to do a weighted average on the testing data
acceptable_delta = 5.0  # This determines how similar FFT and LASSO must be
good_indices_PPG = []
for index in range(0, np.size(testing_reference_HR)):
	if testing_LASSO_PPG_HR[index] == 0:
		continue
	if testing_FFT_PPG_HR[index] == 0:
		continue
	if np.abs(testing_LASSO_PPG_HR[index] - testing_FFT_PPG_HR[index]) > acceptable_delta:
		continue
	good_indices_PPG.append(index)
good_indices_PPG = np.asarray(good_indices_PPG)  

good_indices_PPG_area = []
for index in range(0, np.size(testing_reference_HR)):
	if testing_LASSO_PPG_area_HR[index] == 0:
		continue
	if testing_FFT_PPG_area_HR[index] == 0:
		continue
	if np.abs(testing_LASSO_PPG_area_HR[index] - testing_FFT_PPG_area_HR[index]) > acceptable_delta:
		continue
	good_indices_PPG_area.append(index)
good_indices_PPG_area = np.asarray(good_indices_PPG_area)        

print("Accepted percentage of testing videos using PPG-only: ", np.size(good_indices_PPG)*
	100/np.size(testing_reference_HR))
print("Accepted percentage of testing videos using PPG-area: ", np.size(good_indices_PPG_area)*
	100/np.size(testing_reference_HR))

FFT_PPG_weight = training_LASSO_PPG_HR_MSE / (training_LASSO_PPG_HR_MSE + training_FFT_PPG_HR_MSE)
LASSO_PPG_weight = training_FFT_PPG_HR_MSE / (training_LASSO_PPG_HR_MSE + training_FFT_PPG_HR_MSE)

testing_PPG_HR = FFT_PPG_weight*testing_FFT_PPG_HR[good_indices_PPG] + LASSO_PPG_weight*testing_LASSO_PPG_HR[good_indices_PPG]
testing_PPG_HR_error = testing_PPG_HR - testing_reference_HR[good_indices_PPG]
testing_PPG_HR_AE = np.abs(testing_PPG_HR_error)
testing_PPG_HR_MAE = np.mean(testing_PPG_HR_AE)

FFT_PPG_area_weight = training_LASSO_PPG_area_HR_MSE / (training_LASSO_PPG_area_HR_MSE + training_FFT_PPG_area_HR_MSE)
LASSO_PPG_area_weight = training_FFT_PPG_area_HR_MSE / (training_LASSO_PPG_area_HR_MSE + training_FFT_PPG_area_HR_MSE)

testing_PPG_area_HR = FFT_PPG_area_weight*testing_FFT_PPG_area_HR[good_indices_PPG_area] + LASSO_PPG_area_weight*testing_LASSO_PPG_area_HR[good_indices_PPG_area]
testing_PPG_area_HR_error = testing_PPG_area_HR - testing_reference_HR[good_indices_PPG_area]
testing_PPG_area_HR_AE = np.abs(testing_PPG_area_HR_error)
testing_PPG_area_HR_MAE = np.mean(testing_PPG_area_HR_AE)

# Make Bland-Altman plots
plotBlandAltman(testing_PPG_HR, testing_reference_HR[good_indices_PPG], "BlandAltmanPPG.png", "PPG-only")
plotBlandAltman(testing_PPG_area_HR, testing_reference_HR[good_indices_PPG_area], "BlandAltmanPPGArea.png", "PPG-area")

# Standard error of mean absolute error
testing_std_err_PPG = np.std(testing_PPG_HR_AE) / np.sqrt(np.size(testing_PPG_HR_AE))
testing_std_err_PPG_area = np.std(testing_PPG_area_HR_AE) / np.sqrt(np.size(testing_PPG_area_HR_AE))

print("mean absolute testing error using PPG-only: ", testing_PPG_HR_MAE)
print("standard error of mean absolute testing error using PPG-only: ", testing_std_err_PPG)
print("mean absolute testing error using PPG-area: ", testing_PPG_area_HR_MAE)
print("standard error of mean absolute testing error using PPG-area: ", testing_std_err_PPG_area)

# Check what percentage of non-rejected videos have absolute error >10 bpm, >20 bpm, and >40 bpm
testing_PPG_10 = np.sum(testing_PPG_HR_AE > 10.0)
testing_PPG_20 = np.sum(testing_PPG_HR_AE > 20.0)
testing_PPG_40 = np.sum(testing_PPG_HR_AE > 40.0)
testing_PPG_area_10 = np.sum(testing_PPG_area_HR_AE > 10.0)
testing_PPG_area_20 = np.sum(testing_PPG_area_HR_AE > 20.0)
testing_PPG_area_40 = np.sum(testing_PPG_area_HR_AE > 40.0)

print("Percentage of testing videos off by > 10 bpm using PPG-only: ", 
	testing_PPG_10*100/np.size(testing_PPG_HR_AE))
print("Percentage of testing videos off by > 20 bpm using PPG-only: ", 
	testing_PPG_20*100/np.size(testing_PPG_HR_AE))
print("Percentage of testing videos off by > 40 bpm using PPG-only: ", 
	testing_PPG_40*100/np.size(testing_PPG_HR_AE))
print("Percentage of testing videos off by > 10 bpm using PPG-area: ", 
	testing_PPG_area_10*100/np.size(testing_PPG_area_HR_AE))
print("Percentage of testing videos off by > 20 bpm using PPG-area: ", 
	testing_PPG_area_20*100/np.size(testing_PPG_area_HR_AE))
print("Percentage of testing videos off by > 40 bpm using PPG-area: ", 
	testing_PPG_area_40*100/np.size(testing_PPG_area_HR_AE))




