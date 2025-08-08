import wfdb
import matplotlib.pyplot as plt
from scipy.signal import lfilter, freqz, group_delay
import numpy as np
import warnings

# Suppress specific runtime warnings from signal analysis functions for cleaner output
warnings.filterwarnings('ignore', 'divide by zero encountered in log10')
warnings.filterwarnings('ignore', 'invalid value encountered in divide')
warnings.filterwarnings('ignore', 'The group delay is singular at frequencies')

# --- Custom zplane plotting function ---
def zplane(b, a, ax=None, title='Pole-Zero Plot'):
    """
    Plots the poles and zeros of a digital filter given its numerator (b)
    and denominator (a) coefficients.
    Optionally takes an axes object 'ax' to plot onto.
    """
    z = np.roots(b)
    p = np.roots(a)

    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)

    # Plot unit circle
    unit_circle = plt.Circle((0, 0), radius=1, color='gray', lw=1, fill=False)
    ax.add_patch(unit_circle)
    ax.set_aspect('equal', adjustable='box')

    # Plot the zeros
    ax.plot(z.real, z.imag, 'o', markersize=6, color='blue', label='Zeros') # Reduced markersize for subplot

    # Plot the poles
    ax.plot(p.real, p.imag, 'x', markersize=6, color='red', label='Poles') # Reduced markersize

    # Add axes
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Real', fontsize=8) # Abbreviated labels for subplots
    ax.set_ylabel('Imag', fontsize=8) # Abbreviated labels for subplots

    # Set limits
    max_val = max(1.1, np.max(np.abs(np.concatenate((z, p)))))
    ax.set_xlim((-max_val, max_val))
    ax.set_ylim((-max_val, max_val))

    # Add grid and legend
    ax.grid(True)
    ax.legend()
    ax.set_title(title, fontsize=10) # Reduced fontsize

# --- Helper functions for detection methods (copied from evaluation script) ---

# Adaptive Pan-Tompkins Detection Method
def detect_qrs_pan_tompkins_adaptive(integrated_signal, filtered_bandpass, integrated_peaks_indices, integrated_peaks_amplitudes, sampling_frequency):
    spki, npki, threshold_i1, threshold_i2 = 0.0, 0.0, 0.0, 0.0
    spkf, npkf, threshold_f1, threshold_f2 = 0.0, 0.0, 0.0, 0.0
    refractory_period_ms = 200
    refractory_period_samples = int(refractory_period_ms * sampling_frequency / 1000)
    last_qrs_time = 0
    detected_qrs = []

    initialization_samples_pt = min(5 * sampling_frequency, len(integrated_signal))
    initial_peak_indices_pt = [idx for idx in integrated_peaks_indices if idx < initialization_samples_pt]
    initial_peak_amplitudes_pt = [integrated_signal[idx] for idx in initial_peak_indices_pt]
    if len(initial_peak_amplitudes_pt) >= 2:
        sorted_initial_peaks_pt = sorted(initial_peak_amplitudes_pt)
        initial_spki_pt = sorted_initial_peaks_pt[-1]
        initial_npki_pt = sorted_initial_peaks_pt[0]
        if initial_spki_pt <= initial_npki_pt: initial_npki_pt, initial_spki_pt = 1e-6, 1e-6 * 2
        spki, npki = initial_spki_pt, initial_npki_pt
        initial_filtered_amplitudes_at_peaks_pt = [filtered_bandpass[idx] for idx in initial_peak_indices_pt]
        if len(initial_filtered_amplitudes_at_peaks_pt) > 0:
            initial_spkf_pt = np.max(initial_filtered_amplitudes_at_peaks_pt)
            initial_npkf_pt = np.min(initial_filtered_amplitudes_at_peaks_pt)
            if abs(initial_spkf_pt) <= abs(initial_npkf_pt) and initial_spkf_pt * initial_npkf_pt >= 0: initial_npkf_pt, initial_spkf_pt = 1e-6, -1e-6 if initial_spkf_pt < 0 else 1e-6
            spkf, npkf = initial_spkf_pt, initial_npkf_pt
        else: spkf, npkf = 1.0, -1.0
        signal_noise_diff_i_pt = max(spki - npki, 1e-6)
        threshold_i1 = npki + 0.25 * signal_noise_diff_i_pt
        threshold_i2 = 0.5 * threshold_i1
        signal_noise_diff_f_paper_pt = spkf - npkf
        threshold_f1 = npkf + 0.25 * max(signal_noise_diff_f_paper_pt, 1e-6)
        threshold_f2 = 0.5 * threshold_f1
        try: last_qrs_time = initial_peak_indices_pt[initial_peak_amplitudes_pt.index(initial_spki_pt)]
        except ValueError: last_qrs_time = 0
    else:
         spki, npki, threshold_i1, threshold_i2 = 1.0, 0.1, 0.2, 0.1
         spkf, npkf, threshold_f1, threshold_f2 = 1.0, -1.0, 0.2, 0.1

    for i, peak_index in enumerate(integrated_peaks_indices):
        peak_amplitude_integrated = integrated_signal[peak_index]
        peak_amplitude_filtered   = filtered_bandpass[peak_index]
        time_since_last_qrs = peak_index - last_qrs_time
        if time_since_last_qrs < refractory_period_samples and last_qrs_time != 0: continue
        is_qrs = False
        if peak_amplitude_integrated >= threshold_i1 and abs(peak_amplitude_filtered) >= abs(threshold_f1): is_qrs = True
        elif peak_amplitude_integrated >= threshold_i2 and abs(peak_amplitude_filtered) >= abs(threshold_f2): is_qrs = True
        if is_qrs:
            spki = 0.125 * peak_amplitude_integrated + 0.875 * spki
            spkf = 0.125 * peak_amplitude_filtered + 0.875 * spkf
            last_qrs_time = peak_index
            detected_qrs.append(peak_index)
        else:
            npki = 0.125 * peak_amplitude_integrated + 0.875 * npki
            npkf = 0.125 * filtered_bandpass[peak_index] + 0.875 * npkf
        signal_noise_diff_i = max(spki - npki, 1e-6)
        threshold_i1 = npki + 0.25 * signal_noise_diff_i
        threshold_i2 = 0.5 * threshold_i1
        signal_noise_diff_f_paper = spkf - npkf
        threshold_f1 = npkf + 0.25 * max(signal_noise_diff_f_paper, 1e-6)
        threshold_f2 = 0.5 * threshold_f1

    return detected_qrs

# Static Threshold Detection Method
def detect_qrs_static(integrated_signal, integrated_peaks_indices, integrated_peaks_amplitudes):
    static_threshold_i = np.max(integrated_signal) * 0.3
    detected_qrs = [peak_index for peak_index, peak_amplitude in zip(integrated_peaks_indices, integrated_peaks_amplitudes)
                       if peak_amplitude >= static_threshold_i]
    return detected_qrs

# LMS-based Adaptive Threshold Detection Method
def detect_qrs_lms_adaptive_threshold(integrated_signal, filtered_bandpass, integrated_peaks_indices, integrated_peaks_amplitudes, sampling_frequency):
    mu_lms = 0.01
    refractory_period_ms = 200
    refractory_period_samples = int(refractory_period_ms * sampling_frequency / 1000)

    detected_qrs = []
    last_qrs_time_lms = 0

    initialization_samples_lms_f = min(5 * sampling_frequency, len(filtered_bandpass))
    initial_peak_indices_lms_f = [idx for idx in integrated_peaks_indices if idx < initialization_samples_lms_f]
    initial_filtered_amplitudes_lms = [filtered_bandpass[idx] for idx in initial_peak_indices_lms_f]
    if len(initial_filtered_amplitudes_lms) > 0:
        initial_spkf_lms = np.max(initial_filtered_amplitudes_lms)
        initial_npkf_lms = np.min(initial_filtered_amplitudes_lms)
        signal_noise_diff_f_lms = initial_spkf_lms - initial_npkf_lms
        threshold_f1_for_lms_validation = initial_npkf_lms + 0.25 * max(signal_noise_diff_f_lms, 1e-6)
    else:
         threshold_f1_for_lms_validation = 0.2

    initialization_samples_lms_i = min(5 * sampling_frequency, len(integrated_signal))
    initial_peak_indices_lms_i = [idx for idx in integrated_peaks_indices if idx < initialization_samples_lms_i]
    initial_peak_amplitudes_lms_i = [integrated_signal[idx] for idx in initial_peak_indices_lms_i]
    if len(initial_peak_amplitudes_lms_i) > 0:
         T_lms = np.max(initial_peak_amplitudes_lms_i) * 0.5
    else: T_lms = 1.0

    for i, peak_index in enumerate(integrated_peaks_indices):
        peak_amplitude_integrated = integrated_signal[peak_index]
        peak_amplitude_filtered   = filtered_bandpass[peak_index]

        is_qrs_lms = False
        if peak_amplitude_integrated >= T_lms:
            if abs(peak_amplitude_filtered) >= abs(threshold_f1_for_lms_validation):
                is_qrs_lms = True

        time_since_last_qrs_lms = peak_index - last_qrs_time_lms
        if is_qrs_lms and time_since_last_qrs_lms < refractory_period_samples and last_qrs_time_lms != 0:
            is_qrs_lms = False

        if is_qrs_lms:
            detected_qrs.append(peak_index)
            last_qrs_time_lms = peak_index

        T_lms += mu_lms * (peak_amplitude_integrated - T_lms)


    return detected_qrs


# =============================================================================
# Reproduction Demo (Continued)
# Main execution starts here
# =============================================================================

record_name = '100'
demo_samples = 5000
print(f"--- Running Reproduction Demo for Record: {record_name} ({demo_samples} samples) ---")

# Stage 1: Data Loading
print("Loading record data...")
try:
    record = wfdb.rdrecord(record_name, sampfrom=0, sampto=demo_samples, pn_dir='mitdb')
    print("Record loaded successfully.")
    ecg_signal = record.p_signal[:, 0]
    sampling_frequency = record.fs
    print(f"Sampling frequency: {sampling_frequency} Hz")
    print(f"Number of samples loaded: {len(ecg_signal)}")
    time_vector = np.arange(len(ecg_signal)) / sampling_frequency
    plt.figure(figsize=(12, 4))
    plt.plot(time_vector, ecg_signal, color='mediumpurple')
    plt.title(f'Raw ECG Signal - MIT-BIH Record {record_name}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"Error loading record {record_name}: {e}")
    exit()

# Stage 2: Bandpass Filtering
print("Applying Bandpass Filter...")
a_lp = [1, -2, 1]
b_lp = [1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1]
filtered_lp = lfilter(b_lp, a_lp, ecg_signal)
a_hp = [1, 1]
b_hp = np.zeros(33)
b_hp[0] = -1
b_hp[16] = 32
b_hp[32] = -1
filtered_bandpass = lfilter(b_hp, a_hp, filtered_lp)
print("Bandpass filter applied.")
plt.figure(figsize=(12, 4))
plt.plot(time_vector, filtered_bandpass, color='mediumpurple')
plt.title(f'Bandpass Filtered ECG Signal - MIT-BIH Record {record_name}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Stage 3: Differentiation
print("Applying Derivative Filter...")
b_deriv = [1, 2, 0, -2, -1]
a_deriv = [1]
scaling_factor_deriv = sampling_frequency / 8
filtered_deriv_raw = lfilter(b_deriv, a_deriv, filtered_bandpass)
filtered_deriv = filtered_deriv_raw * scaling_factor_deriv
print("Derivative filter applied.")
plt.figure(figsize=(12, 4))
plt.plot(time_vector, filtered_deriv, color='mediumpurple')
plt.title(f'Derivative ECG Signal - MIT-BIH Record {record_name}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (Rate of Change)')
plt.grid(True)
plt.ylim([-40000, 40000])
plt.show()

# Stage 4: Squaring
print("Applying Squaring...")
squared_signal = filtered_deriv ** 2
print("Squaring applied.")
plt.figure(figsize=(12, 4))
plt.plot(time_vector, squared_signal, color='mediumpurple')
plt.title(f'Squared ECG Signal - MIT-BIH Record {record_name}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (Squared Rate of Change)')
plt.grid(True)
plt.ylim([-0.05 * np.max(squared_signal), 1.05 * np.max(squared_signal)])
plt.show()

# Stage 5: Moving Window Integration
print("Applying Moving Window Integration...")
N = int(0.15 * sampling_frequency)
b_integ = np.ones(N) / float(N)
a_integ = [1]
integrated_signal = lfilter(b_integ, a_integ, squared_signal)
print("Moving Window Integration applied.")
plt.figure(figsize=(12, 4))
plt.plot(time_vector, integrated_signal, color='mediumpurple')
plt.title(f'Integrated ECG Signal - MIT-BIH Record {record_name}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (Integrated Squared Signal)')
plt.grid(True)
plt.ylim([0, 1.1 * np.max(integrated_signal)])
plt.show()

# Stage 6: Peak Detection in Integrated Signal
print("Detecting peaks in the integrated signal...")
integrated_peaks_indices = []
integrated_peaks_amplitudes = []
for i in range(1, len(integrated_signal) - 1):
    if integrated_signal[i] > integrated_signal[i-1] and integrated_signal[i] > integrated_signal[i+1]:
        integrated_peaks_indices.append(i)
        integrated_peaks_amplitudes.append(integrated_signal[i])
print(f"Detected {len(integrated_peaks_indices)} peaks in the integrated signal.")
plt.figure(figsize=(12, 4))
plt.plot(time_vector, integrated_signal, color='mediumpurple', label='Integrated Signal')
plt.plot(np.array(integrated_peaks_indices) / sampling_frequency, integrated_peaks_amplitudes, 'ro', markersize=5, label='Detected Peaks')
plt.title(f'Integrated ECG Signal with Detected Peaks - MIT-BIH Record {record_name}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (Integrated Squared Signal)')
plt.grid(True)
plt.ylim([0, 1.1 * np.max(integrated_signal)])
plt.legend()
plt.show()


# =============================================================================
# DSP Analysis (Task 3) - Plots consolidated into single figures
# =============================================================================
print("\n--- Analyzing Filter DSP properties ---")

# Bandpass Filter Analysis
print("Analyzing Bandpass Filter properties...")
b_bandpass = np.convolve(b_lp, b_hp)
a_bandpass = np.convolve(a_lp, a_hp)
w_bandpass, h_bandpass = freqz(b_bandpass, a_bandpass, worN=8192, fs=sampling_frequency)
magnitude_response_bandpass = 20 * np.log10(abs(h_bandpass))
phase_response_bandpass = np.unwrap(np.angle(h_bandpass))
w_gd_bandpass, gd_bandpass = group_delay((b_bandpass, a_bandpass), fs=sampling_frequency)

fig_bp, axes_bp = plt.subplots(2, 2, figsize=(10, 8))
axes_bp[0, 0].plot(w_bandpass, magnitude_response_bandpass, color='mediumpurple')
axes_bp[0, 0].set_title('Bandpass Mag Response (dB)', fontsize=10)
axes_bp[0, 0].set_xlabel('Freq (Hz)', fontsize=8)
axes_bp[0, 0].set_ylabel('Magnitude (dB)', fontsize=8)
axes_bp[0, 0].grid(True)
axes_bp[0, 0].set_xlim([0, sampling_frequency / 2])
axes_bp[0, 0].set_ylim([-100, 50])
axes_bp[0, 0].tick_params(labelsize=8)

axes_bp[0, 1].plot(w_bandpass, phase_response_bandpass, color='mediumpurple')
axes_bp[0, 1].set_title('Bandpass Phase Response (rad)', fontsize=10)
axes_bp[0, 1].set_xlabel('Freq (Hz)', fontsize=8)
axes_bp[0, 1].set_ylabel('Phase (rad)', fontsize=8)
axes_bp[0, 1].grid(True)
axes_bp[0, 1].set_xlim([0, sampling_frequency / 2])
axes_bp[0, 1].tick_params(labelsize=8)

axes_bp[1, 0].plot(w_gd_bandpass, gd_bandpass, color='mediumpurple')
axes_bp[1, 0].set_title('Bandpass Group Delay (samples)', fontsize=10)
axes_bp[1, 0].set_xlabel('Freq (Hz)', fontsize=8)
axes_bp[1, 0].set_ylabel('Group Delay (samples)', fontsize=8)
axes_bp[1, 0].grid(True)
axes_bp[1, 0].set_xlim([0, sampling_frequency / 2])
axes_bp[1, 0].tick_params(labelsize=8)

zplane(b_bandpass, a_bandpass, ax=axes_bp[1, 1], title='Bandpass Pole-Zero Plot')
axes_bp[1, 1].tick_params(labelsize=8)
plt.tight_layout()
plt.suptitle('Bandpass Filter Analysis', y=1.02, fontsize=14)
plt.show()


# Derivative Filter Analysis
print("Analyzing Derivative Filter properties...")
w_deriv, h_deriv = freqz(b_deriv, a_deriv, worN=8192, fs=sampling_frequency)
magnitude_response_deriv = 20 * np.log10(abs(h_deriv))
phase_response_deriv = np.unwrap(np.angle(h_deriv))
w_gd_deriv, gd_deriv = group_delay((b_deriv, a_deriv), fs=sampling_frequency)

fig_deriv, axes_deriv = plt.subplots(2, 2, figsize=(10, 8))
axes_deriv[0, 0].plot(w_deriv, magnitude_response_deriv, color='mediumpurple')
axes_deriv[0, 0].set_title('Derivative Mag Response (dB)', fontsize=10)
axes_deriv[0, 0].set_xlabel('Freq (Hz)', fontsize=8)
axes_deriv[0, 0].set_ylabel('Magnitude (dB)', fontsize=8)
axes_deriv[0, 0].grid(True)
axes_deriv[0, 0].set_xlim([0, sampling_frequency / 2])
axes_deriv[0, 0].tick_params(labelsize=8)

axes_deriv[0, 1].plot(w_deriv, phase_response_deriv, color='mediumpurple')
axes_deriv[0, 1].set_title('Derivative Phase Response (rad)', fontsize=10)
axes_deriv[0, 1].set_xlabel('Freq (Hz)', fontsize=8)
axes_deriv[0, 1].set_ylabel('Phase (rad)', fontsize=8)
axes_deriv[0, 1].grid(True)
axes_deriv[0, 1].set_xlim([0, sampling_frequency / 2])
axes_deriv[0, 1].tick_params(labelsize=8)

axes_deriv[1, 0].plot(w_gd_deriv, gd_deriv, color='mediumpurple')
axes_deriv[1, 0].set_title('Derivative Group Delay (samples)', fontsize=10)
axes_deriv[1, 0].set_xlabel('Freq (Hz)', fontsize=8)
axes_deriv[1, 0].set_ylabel('Group Delay (samples)', fontsize=8)
axes_deriv[1, 0].grid(True)
axes_deriv[1, 0].set_xlim([0, sampling_frequency / 2])
axes_deriv[1, 0].set_ylim([0, gd_deriv[0] + 0.5])
axes_deriv[1, 0].tick_params(labelsize=8)

zplane(b_deriv, a_deriv, ax=axes_deriv[1, 1], title='Derivative Pole-Zero Plot')
axes_deriv[1, 1].tick_params(labelsize=8)
plt.tight_layout()
plt.suptitle('Derivative Filter Analysis', y=1.02, fontsize=14)
plt.show()

# Integrator Filter Analysis
print("Analyzing Integrator Filter properties...")
w_integ, h_integ = freqz(b_integ, a_integ, worN=8192, fs=sampling_frequency)
magnitude_response_integ = 20 * np.log10(abs(h_integ))
phase_response_integ = np.unwrap(np.angle(h_integ))
w_gd_integ, gd_integ = group_delay((b_integ, a_integ), fs=sampling_frequency)

fig_integ, axes_integ = plt.subplots(2, 2, figsize=(10, 8))
axes_integ[0, 0].plot(w_integ, magnitude_response_integ, color='mediumpurple')
axes_integ[0, 0].set_title('Integrator Mag Response (dB)', fontsize=10)
axes_integ[0, 0].set_xlabel('Freq (Hz)', fontsize=8)
axes_integ[0, 0].set_ylabel('Magnitude (dB)', fontsize=8)
axes_integ[0, 0].grid(True)
axes_integ[0, 0].set_xlim([0, sampling_frequency / 2])
axes_integ[0, 0].tick_params(labelsize=8)

axes_integ[0, 1].plot(w_integ, phase_response_integ, color='mediumpurple')
axes_integ[0, 1].set_title('Integrator Phase Response (rad)', fontsize=10)
axes_integ[0, 1].set_xlabel('Freq (Hz)', fontsize=8)
axes_integ[0, 1].set_ylabel('Phase (rad)', fontsize=8)
axes_integ[0, 1].grid(True)
axes_integ[0, 1].set_xlim([0, sampling_frequency / 2])
axes_integ[0, 1].tick_params(labelsize=8)

axes_integ[1, 0].plot(w_gd_integ, gd_integ, color='mediumpurple')
axes_integ[1, 0].set_title('Integrator Group Delay (samples)', fontsize=10)
axes_integ[1, 0].set_xlabel('Freq (Hz)', fontsize=8)
axes_integ[1, 0].set_ylabel('Group Delay (samples)', fontsize=8)
axes_integ[1, 0].grid(True)
axes_integ[1, 0].set_xlim([0, sampling_frequency / 2])
axes_integ[1, 0].set_ylim([0, gd_integ[0] + 1])
axes_integ[1, 0].tick_params(labelsize=8)

zplane(b_integ, a_integ, ax=axes_integ[1, 1], title='Integrator Pole-Zero Plot')
axes_integ[1, 1].tick_params(labelsize=8)
plt.tight_layout()
plt.suptitle('Integrator Filter Analysis', y=1.02, fontsize=14)
plt.show()

print("Completed DSP Analysis.")


# =============================================================================
# Run Detection Methods on Demo Segment & Plot Comparison
# This section comes AFTER all signal processing and peak detection
# =============================================================================

print("\n--- Running all three detection methods on demo segment ---")

# Ensure the necessary variables are available from the preceding stages:
# integrated_signal, filtered_bandpass, integrated_peaks_indices, integrated_peaks_amplitudes, sampling_frequency

# Run the three detection methods on the demo segment
detected_qrs_pt_adaptive = detect_qrs_pan_tompkins_adaptive(integrated_signal, filtered_bandpass, integrated_peaks_indices, integrated_peaks_amplitudes, sampling_frequency)
detected_qrs_static = detect_qrs_static(integrated_signal, integrated_peaks_indices, integrated_peaks_amplitudes)
detected_qrs_lms_adaptive = detect_qrs_lms_adaptive_threshold(integrated_signal, filtered_bandpass, integrated_peaks_indices, integrated_peaks_amplitudes, sampling_frequency)

print(f"Detected QRS (Adaptive Pan-Tompkins): {len(detected_qrs_pt_adaptive)}")
print(f"Detected QRS (Static Threshold):      {len(detected_qrs_static)}")
print(f"Detected QRS (LMS Adaptive Threshold):{len(detected_qrs_lms_adaptive)}")


# Load ground truth for demo plot
try:
    ann = wfdb.rdann(record_name, 'atr', sampto=len(ecg_signal), pn_dir='mitdb')
    true_beats = np.array(ann.sample)
    print(f"Loaded {len(true_beats)} ground truth annotations for demo segment.")
except Exception as e:
     print(f"Warning: Could not load ground truth for demo plot: {e}")
     true_beats = np.array([])


# Final Plot for Demo (Comparison of all three methods)
print("\nPlotting comparison of detected QRS locations for demo segment...")
plt.figure(figsize=(12, 6))
plt.plot(time_vector, ecg_signal, color='mediumpurple', label='Raw ECG Signal')
ymin_plot, ymax_plot = np.min(ecg_signal), np.max(ecg_signal)

# Plot detections for each method
if detected_qrs_pt_adaptive:
    times_pt = np.array(detected_qrs_pt_adaptive) / sampling_frequency
    # Plotting slightly higher to distinguish from others if needed
    plt.vlines(times_pt, ymin=ymin_plot, ymax=ymax_plot, colors='red', linestyles='--', label='Adaptive Pan-Tompkins')

if detected_qrs_static:
     times_static = np.array(detected_qrs_static) / sampling_frequency
     # Plot slightly lower than PT
     plt.vlines(times_static, ymin=ymin_plot, ymax=ymax_plot * 0.9, colors='blue', linestyles=':', label='Static Threshold')

if detected_qrs_lms_adaptive: # THIS INCLUDES THE LMS PLOT
     times_lms = np.array(detected_qrs_lms_adaptive) / sampling_frequency
     # Plot even lower than static
     plt.vlines(times_lms, ymin=ymin_plot, ymax=ymax_plot * 0.8, colors='green', linestyles='-.', label='LMS Adaptive Threshold')

# Plot Ground Truth Annotations for reference
if len(true_beats) > 0:
    true_beat_times = true_beats / sampling_frequency
    # Plot ground truth as gray solid lines at the full height
    plt.vlines(true_beat_times, ymin=ymin_plot, ymax=ymax_plot, colors='gray', linestyles='-', linewidth=2, label='Ground Truth', alpha=0.7)


plt.title(f'QRS Detections Comparison - MIT-BIH Record {record_name} ({demo_samples} samples)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.ylim([ymin_plot, ymax_plot])
plt.show()


print("\n--- Demo Script finished ---")