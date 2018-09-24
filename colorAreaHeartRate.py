"""
This code batch processes extracted area and color time series
to estimate heart rate, using 6 algorithm variations.

The provided directory should contain data files of the form
subject_number.video_number.area.csv and 
subject_number.video_number.color.csv, the same format
produced by running extractSignals.py. The following output
files are produced for each pair of 
subject_number.video_number.area.csv and
subject_number.video_number.color.csv files:
subject_number.video_number.FFT.png: A plot of FFT coefficient
    magnitudes between 30 and 180 bpm for both the green color
    and ellipse area signals.
subject_number.video_number.LASSO.png: A plot of the Group LASSO
    coefficient magnitudes between 30 and 180 bpm for both the
    green color and ellipse area signals. This is essentially a 
    sparser version of the FFT representation.

An additional output file HREstimates.csv is produced in the
input directory. This file has data in the following columns:
subject_number
video_number
FFT_color_HR: Heart rate estimate (bpm) using FFT to estimate
    the spectrum and only including the green color signal.
FFT_area_HR: Heart rate estimate (bpm) using FFT to estimate
    the spectrum and only including the ellipse area signal.
LASSO_color_HR: Heart rate estimate (bpm) using Group LASSO to
    estimate the spectrum and only including the green color
    signal.
LASSO_area_HR: Heart rate estimate (bpm) using Group LASSO to
    estimate the spectrum and only including the ellipse area
    signal.
FFT_HR: Heart rate estimate (bpm) using FFT to estimate the
    spectrum of each signal, and taking the elementwise product
    in the frequency domain of the color and area coefficient
    magnitudes.
LASSO_HR: Heart rate estimate (bpm) using Group LASSO to estimate
    the spectrum of each signal, and taking the elementwise
    product in the frequency domain of the color and area 
    coefficient magnitudes.

Run this code using the command:
python colorAreaHeartRate.py directory
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cvxpy
import glob
import sys
from cvxpy import *

def PlotLassoColorArea(greens, areas, threshold_factor, figname):

    # Prepare data and parameters
    green_window = greens
    area_window = areas
    if np.size(greens)%2 == 1:
        green_window = greens[0:np.size(greens) - 1]
        area_window = areas[0:np.size(greens) - 1]
    
    window_size = np.size(area_window)
    window_size_secs = window_size / frame_rate
    hz = np.arange(0, int(window_size/2+1))
    hz = hz * frame_rate / window_size
    bpm_window = hz * 60
    lower_index = int(lower_bpm*window_size_secs/60.0)
    upper_index = int(upper_bpm*window_size_secs/60.0)
    lower_thresh_index = int((lower_bpm + 10)*
        window_size_secs/60.0) - lower_index
    upper_thresh_index = int((upper_bpm - 10)*
        window_size_secs/60.0) - lower_index

    # Group LASSO on both (green) color and area

    # Only use sine and cosine components at frequencies 
    # between 30 and 180 bpm.
    # For each frequency component, include sin(2*pi*f*frame) 
    # and cos(2*pi*f*frame).
    # Frequency spacing between components is the same as in the FFT.

    # Prepare the X matrix (each column is a frequency component)
    frame_numbers = np.arange(0, window_size)
    freq = lower_bpm / (60.0 * frame_rate)
    sin = np.sin(2*np.pi*freq*frame_numbers)
    cos = np.cos(2*np.pi*freq*frame_numbers)
    X = np.concatenate((sin[:, np.newaxis], cos[:, np.newaxis]), 1)
    max_freq = upper_bpm / (60.0 * frame_rate)
    steps = upper_index - lower_index 
    for index in range(2, steps + 1):
        freq = freq + 1.0/window_size
        sin = np.sin(2*np.pi*freq*frame_numbers)
        cos = np.cos(2*np.pi*freq*frame_numbers)
        X = np.concatenate((X, sin[:, np.newaxis]), 1)
        X = np.concatenate((X, cos[:, np.newaxis]), 1)
    green_param = 50
    area_param = 200
    n = np.shape(X)[1]

    green_B = Variable(n) # Weight vectors
    area_B = Variable(n)
    # Construct the group weight penalties
    green_B_norm = 0
    area_B_norm = 0
    for index in range(0,150,2):
        green_B_norm = green_B_norm + cvxpy.pnorm(
            green_B[index:index + 2], 2)
        area_B_norm = area_B_norm + cvxpy.pnorm(
            area_B[index:index + 2], 2)

    # Bandpass filter the green color and area
    # For simplicity, filtering is done naively using FFT
    # and edge effects up to 10 bpm are ignored (below)
    green_fft = np.fft.rfft(green_window)
    green_bandpass_fft = np.copy(green_fft)
    green_bandpass_fft[bpm_window < lower_bpm] = 0
    green_bandpass_fft[bpm_window > upper_bpm] = 0
    green_bandpass_filtered = np.fft.irfft(green_bandpass_fft)

    area_fft = np.fft.rfft(area_window)
    area_bandpass_fft = np.copy(area_fft)
    area_bandpass_fft[bpm_window < lower_bpm] = 0
    area_bandpass_fft[bpm_window > upper_bpm] = 0
    area_bandpass_filtered = np.fft.irfft(area_bandpass_fft)    

    # Normalize green_bandpass_filtered and area_bandpass_filtered to be 
    # zero mean and unit standard deviation
    std_green = green_bandpass_filtered - np.mean(green_bandpass_filtered)
    std_green = std_green / np.sqrt(np.var(std_green))

    std_area = area_bandpass_filtered - np.mean(area_bandpass_filtered)
    std_area = std_area / np.sqrt(np.var(std_area))

    # Solve Group LASSO for both green color and area
    green_objective = Minimize(sum_squares(std_green - X*green_B) + 
        green_param*green_B_norm)
    constraints = []
    green_prob = Problem(green_objective, constraints)
    green_result = green_prob.solve(solver=CVXOPT)

    area_objective = Minimize(sum_squares(std_area - X*area_B) + 
        area_param*area_B_norm)
    area_prob = Problem(area_objective, constraints)
    area_result = area_prob.solve(solver=CVXOPT)

    # Calculate magnitude of each frequency component
    green_magnitudes = np.zeros((steps, 1))
    area_magnitudes = np.zeros((steps, 1))
    green_Bval = np.absolute(green_B.value)
    area_Bval = np.absolute(area_B.value)
    for index in range(0,steps*2,2):
        green_magnitudes[int(index/2)] = np.linalg.norm(
            green_Bval[index:index + 2])
        area_magnitudes[int(index/2)] = np.linalg.norm(
            area_Bval[index:index + 2])

    # Plot coefficient magnitudes and product of magnitudes
    threshold = np.max([green_magnitudes, area_magnitudes])*threshold_factor
    product_magnitudes = green_magnitudes * area_magnitudes
    fig = plt.figure(figsize=(15,6))
    plt.plot(bpm_window[lower_index:upper_index], green_magnitudes, 'go-', 
        linewidth=2, label='Green Color')
    plt.plot(bpm_window[lower_index:upper_index], area_magnitudes, 'ro-', 
        linewidth=2, label='Ellipse Area')
    plt.plot(bpm_window[lower_index:upper_index], product_magnitudes, 'bo-', 
        linewidth=2, label='Green * Area')
    plt.plot(bpm_window[lower_index:upper_index], np.ones((steps, 1)) * 
        threshold, 'k-', linewidth=2, label='Threshold')
    plt.xlabel('Frequency (bpm)', fontsize = 16)
    plt.ylabel('Magnitude of Group LASSO Coefficient', fontsize = 16)
    plt.legend(prop={'size': 20})
    fig.savefig(figname)
    plt.close('all')

    # Ignore edge effects up to 10 bpm
    green_magnitudes[0:lower_thresh_index + 1] = 0.0
    green_magnitudes[upper_thresh_index:np.size(green_magnitudes)] = 0.0
    # Compute peak frequency of green magnitude, and store
    green_peak_index = np.argmax(green_magnitudes)
    green_peak_value = np.max(green_magnitudes)
    green_peak_bpm = green_peak_index * 60.0 / window_size_secs + lower_bpm

    # Ignore edge effects up to 10 bpm
    area_magnitudes[0:lower_thresh_index + 1] = 0.0
    area_magnitudes[upper_thresh_index:np.size(area_magnitudes)] = 0.0
    # Compute peak frequency of area magnitude, and store
    area_peak_index = np.argmax(area_magnitudes)
    area_peak_value = np.max(area_magnitudes)
    area_peak_bpm = area_peak_index * 60.0 / window_size_secs + lower_bpm      

    # Ignore edge effects up to 10 bpm
    product_magnitudes[0:lower_thresh_index + 1] = 0.0
    product_magnitudes[upper_thresh_index:np.size(product_magnitudes)] = 0.0
    # Compute peak frequency of product of magnitudes, and store
    peak_index = np.argmax(product_magnitudes)
    peak_value = np.max(product_magnitudes)
    peak_bpm = peak_index * 60.0 / window_size_secs + lower_bpm
    # Enforce adaptive threshold
    if peak_value < threshold:
        peak_bpm = 0.0   

    return [green_peak_bpm, area_peak_bpm, peak_bpm]

def PlotFourierColorArea(greens, areas, threshold_factor, figname):

    # Prepare data and parameters
    green_window = greens
    area_window = areas

    if np.size(greens)%2 == 1:
        green_window = greens[0:np.size(greens) - 1]
        area_window = areas[0:np.size(greens) - 1]
    
    window_size = np.size(area_window)
    window_size_secs = window_size / frame_rate
    hz = np.arange(0, int(window_size/2+1))
    hz = hz * frame_rate / window_size
    bpm_window = hz * 60
    lower_index = int(lower_bpm*window_size_secs/60.0)
    upper_index = int(upper_bpm*window_size_secs/60.0)
    lower_thresh_index = int((lower_bpm + 10)*
        window_size_secs/60.0) - lower_index
    upper_thresh_index = int((upper_bpm - 10)*
        window_size_secs/60.0) - lower_index
    steps = upper_index - lower_index

    # Bandpass filter the green color and area
    # For simplicity, filtering is done naively using FFT
    # and edge effects up to 10 bpm are ignored (below)
    green_fft = np.fft.rfft(green_window)
    green_bandpass_fft = np.copy(green_fft)
    green_bandpass_fft[bpm_window < lower_bpm] = 0
    green_bandpass_fft[bpm_window > upper_bpm] = 0
    green_bandpass_filtered = np.fft.irfft(green_bandpass_fft)

    area_fft = np.fft.rfft(area_window)
    area_bandpass_fft = np.copy(area_fft)
    area_bandpass_fft[bpm_window < lower_bpm] = 0
    area_bandpass_fft[bpm_window > upper_bpm] = 0
    area_bandpass_filtered = np.fft.irfft(area_bandpass_fft)    

    # Normalize green_bandpass_filtered and area_bandpass_filtered to be 
    # zero mean and unit standard deviation
    std_green = green_bandpass_filtered - np.mean(green_bandpass_filtered)
    std_green = std_green / np.sqrt(np.var(std_green))

    std_area = area_bandpass_filtered - np.mean(area_bandpass_filtered)
    std_area = std_area / np.sqrt(np.var(std_area))
    
    # Calculate magnitude of each frequency component (FFT)
    green_magnitudes = np.absolute(
        np.fft.rfft(std_green))[lower_index:upper_index]
    area_magnitudes = np.absolute(
        np.fft.rfft(std_area))[lower_index:upper_index]

    # Plot FFT magnitudes and product of magnitudes
    threshold = np.max([green_magnitudes, area_magnitudes])*threshold_factor
    product_magnitudes = green_magnitudes * area_magnitudes
    fig = plt.figure(figsize=(15,6))
    plt.plot(bpm_window[lower_index:upper_index], green_magnitudes, 'go-', 
        linewidth=2, label='Green Color')
    plt.plot(bpm_window[lower_index:upper_index], area_magnitudes, 'ro-', 
        linewidth=2, label='Ellipse Area')
    plt.plot(bpm_window[lower_index:upper_index], product_magnitudes/1000.0, 
        'bo-', linewidth=2, label='Green * Area')
    plt.plot(bpm_window[lower_index:upper_index], np.ones((steps, 1)) * 
        threshold/1000.0, 
        'k-', linewidth=2, label='Threshold')
    plt.xlabel('Frequency (bpm)', fontsize = 16)
    plt.ylabel('Magnitude of FFT Coefficient', fontsize = 16)
    plt.legend(prop={'size': 20})
    fig.savefig(figname)
    plt.close('all')

    # Ignore edge effects up to 10 bpm
    green_magnitudes[0:lower_thresh_index + 1] = 0.0
    green_magnitudes[upper_thresh_index:np.size(green_magnitudes)] = 0.0
    # Compute peak frequency of green magnitude, and store
    green_peak_index = np.argmax(green_magnitudes)
    green_peak_value = np.max(green_magnitudes)
    green_peak_bpm = green_peak_index * 60.0 / window_size_secs + lower_bpm

    # Ignore edge effects up to 10 bpm
    area_magnitudes[0:lower_thresh_index + 1] = 0.0
    area_magnitudes[upper_thresh_index:np.size(area_magnitudes)] = 0.0
    # Compute peak frequency of area magnitude, and store
    area_peak_index = np.argmax(area_magnitudes)
    area_peak_value = np.max(area_magnitudes)
    area_peak_bpm = area_peak_index * 60.0 / window_size_secs + lower_bpm      

    # Ignore edge effects up to 10 bpm
    product_magnitudes[0:lower_thresh_index + 1] = 0.0
    product_magnitudes[upper_thresh_index:np.size(product_magnitudes)] = 0.0
    # Compute peak frequency of product of magnitudes, and store
    peak_index = np.argmax(product_magnitudes)
    peak_value = np.max(product_magnitudes)
    peak_bpm = peak_index * 60.0 / window_size_secs + lower_bpm
    # Enforce adaptive threshold
    if peak_value < threshold:
        peak_bpm = 0.0   

    return [green_peak_bpm, area_peak_bpm, peak_bpm]

frame_rate = 30
lower_bpm = 30
upper_bpm = 180

# Take the name of the folder as input
area_names = glob.glob(sys.argv[1] + "/*.area.csv")
area_names.sort()
output_name = sys.argv[1] + "/HREstimates.csv"

peaks = []

for area_name in area_names:
    print(area_name)
    color_name = area_name[0:-8] + "color.csv"
    FFT_figname = area_name[0:-8] + "FFT.png"
    LASSO_figname = area_name[0:-8] + "LASSO.png"
    subject_number = int(area_name.split("/")[1].split(".")[0])
    video_number = int(area_name.split("/")[1].split(".")[1])

    area = np.loadtxt(area_name)
    color = np.loadtxt(color_name)

    # Skip videos that are less than 20 seconds of decent data
    if np.size(area) < 20 * frame_rate:
        peaks.append([subject_number, video_number, 0, 0, 0, 0, 0, 0])
        continue
    green = color[:,1]

    # On videos where ellipse estimation fails, 
    # there may be one extra color value
    if np.size(green) != np.size(area):
        green = green[0:np.size(area)]

    # Estimate heart rate using the six algorithms. Threshold factors
    # for FFT (100) and Group LASSO (0.05) are heuristic.
    [green_bpm_FFT, area_bpm_FFT, bpm_FFT] = PlotFourierColorArea(
        green, area, 100, FFT_figname)
    [green_bpm_LASSO, area_bpm_LASSO, bpm_LASSO] = PlotLassoColorArea(
        green, area, 0.05, LASSO_figname)

    peaks.append([subject_number, video_number, green_bpm_FFT, 
        area_bpm_FFT, green_bpm_LASSO, area_bpm_LASSO, 
        bpm_FFT, bpm_LASSO])

np.savetxt(output_name, peaks)	










