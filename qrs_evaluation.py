import wfdb
import matplotlib.pyplot as plt
from scipy.signal import lfilter # Only need lfilter for processing stages
import numpy as np
import warnings

# Suppress specific runtime warnings
warnings.filterwarnings('ignore', 'divide by zero encountered in log10')
warnings.filterwarnings('ignore', 'invalid value encountered in divide')

# --- Evaluation Function ---
def score(detected, truth, sampling_frequency, tol_ms, processing_delay_samples):
    """
    Scores detected beat locations against ground truth annotations.
    Accounts for processing delay by shifting detected times.
    """
    tol_samps = int(tol_ms * sampling_frequency / 1000)
    TP = FP = 0
    matched_truth_indices = set()

    adjusted_detected = np.array(detected) - processing_delay_samples
    adjusted_detected = adjusted_detected[adjusted_detected >= 0]

    for d_adj in adjusted_detected:
        match = next((t for t in truth if abs(t - d_adj) <= tol_samps and t not in matched_truth_indices), None)
        if match is not None:
            TP += 1
            matched_truth_indices.add(match)
        else:
            FP += 1

    FN = len(truth) - TP

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    ppv  = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1   = (2 * sensitivity * ppv / (sensitivity + ppv)) if (sensitivity + ppv) > 0 else 0.0

    return TP, FP, FN, sensitivity, ppv, f1

# --- Pan-Tompkins Processing Pipeline (Helper function to apply stages 2-5) ---
def apply_pan_tompkins_pipeline(ecg_signal, sampling_frequency):
    # Stage 2: Bandpass Filter
    a_lp = [1, -2, 1]
    b_lp = [1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1]
    filtered_lp = lfilter(b_lp, a_lp, ecg_signal)
    a_hp = [1, 1]
    b_hp = np.zeros(33)
    b_hp[0] = -1
    b_hp[16] = 32
    b_hp[32] = -1
    filtered_bandpass = lfilter(b_hp, a_hp, filtered_lp)

    # Stage 3: Differentiation
    b_deriv = [1, 2, 0, -2, -1]
    a_deriv = [1]
    filtered_deriv_raw = lfilter(b_deriv, a_deriv, filtered_bandpass)
    scaling_factor_deriv = sampling_frequency / 8
    filtered_deriv = filtered_deriv_raw * scaling_factor_deriv

    # Stage 4: Squaring
    squared_signal = filtered_deriv ** 2

    # Stage 5: Moving Window Integration
    N = int(0.15 * sampling_frequency)
    b_integ = np.ones(N) / float(N)
    a_integ = [1]
    integrated_signal = lfilter(b_integ, a_integ, squared_signal)

    # Peak Detection in Integrated Signal
    integrated_peaks_indices = []
    integrated_peaks_amplitudes = []
    for i in range(1, len(integrated_signal) - 1):
        if integrated_signal[i] > integrated_signal[i-1] and integrated_signal[i] > integrated_signal[i+1]:
            integrated_peaks_indices.append(i)
            integrated_peaks_amplitudes.append(integrated_signal[i])

    return integrated_signal, filtered_bandpass, integrated_peaks_indices, integrated_peaks_amplitudes

# --- Adaptive Pan-Tompkins Detection Method ---
def detect_qrs_pan_tompkins(integrated_signal, filtered_bandpass, integrated_peaks_indices, integrated_peaks_amplitudes, sampling_frequency):
    spki, npki, threshold_i1, threshold_i2 = 0.0, 0.0, 0.0, 0.0
    spkf, npkf, threshold_f1, threshold_f2 = 0.0, 0.0, 0.0, 0.0
    refractory_period_ms = 200
    refractory_period_samples = int(refractory_period_ms * sampling_frequency / 1000)
    last_qrs_time = 0
    detected_qrs_pt_adaptive = []

    # Initialization
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
         # print("Warning: Not enough initial peaks for reliable PT initialization.") # For evaluation, avoid prints
         spki, npki, threshold_i1, threshold_i2 = 1.0, 0.1, 0.2, 0.1
         spkf, npkf, threshold_f1, threshold_f2 = 1.0, -1.0, 0.2, 0.1

    # Main Processing Loop
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
            detected_qrs_pt_adaptive.append(peak_index)
        else:
            npki = 0.125 * peak_amplitude_integrated + 0.875 * npki
            npkf = 0.125 * filtered_bandpass[peak_index] + 0.875 * npkf # Use filtered at peak index
        signal_noise_diff_i = max(spki - npki, 1e-6)
        threshold_i1 = npki + 0.25 * signal_noise_diff_i
        threshold_i2 = 0.5 * threshold_i1
        signal_noise_diff_f_paper = spkf - npkf
        threshold_f1 = npkf + 0.25 * max(signal_noise_diff_f_paper, 1e-6)
        threshold_f2 = 0.5 * threshold_f1

    return detected_qrs_pt_adaptive

# --- Static Threshold Detection Method ---
def detect_qrs_static(integrated_signal, integrated_peaks_indices, integrated_peaks_amplitudes, sampling_frequency):
    # Use a single, fixed threshold based on the overall max integrated signal peak.
    # This threshold is NOT adaptive and uses only the integrated signal.
    static_threshold_i = np.max(integrated_signal) * 0.3 # Tuned factor
    detected_qrs_static = [peak_index for peak_index, peak_amplitude in zip(integrated_peaks_indices, integrated_peaks_amplitudes)
                           if peak_amplitude >= static_threshold_i]
    return detected_qrs_static

# --- LMS-based Adaptive Threshold Detection Method ---
# Renamed from 'Simplified LMS Attempt' for professionalism while acknowledging its basis
def detect_qrs_lms_adaptive_threshold(integrated_signal, filtered_bandpass, integrated_peaks_indices, integrated_peaks_amplitudes, sampling_frequency):
    mu_lms = 0.01 # LMS-like adaptation rate (tuned parameter)
    refractory_period_ms = 200
    refractory_period_samples = int(refractory_period_ms * sampling_frequency / 1000)

    detected_qrs_lms_adaptive = []
    last_qrs_time_lms = 0

    # Need threshold_f1 for validation, initialize based on initial peaks of filtered signal
    initialization_samples_lms_f = min(5 * sampling_frequency, len(filtered_bandpass))
    initial_peak_indices_lms_f = [idx for idx in integrated_peaks_indices if idx < initialization_samples_lms_f] # Use peaks from integrated signal for timing
    initial_filtered_amplitudes_lms = [filtered_bandpass[idx] for idx in initial_peak_indices_lms_f] # Get filtered amplitudes at these times
    if len(initial_filtered_amplitudes_lms) > 0:
        initial_spkf_lms = np.max(initial_filtered_amplitudes_lms)
        initial_npkf_lms = np.min(initial_filtered_amplitudes_lms)
        signal_noise_diff_f_lms = initial_spkf_lms - initial_npkf_lms
        threshold_f1_for_lms_validation = initial_npkf_lms + 0.25 * max(signal_noise_diff_f_lms, 1e-6)
    else:
         threshold_f1_for_lms_validation = 0.2 # Default minimal

    # Initialize the LMS-based threshold T_lms. Could use an initial guess or based on initial peaks of integrated signal.
    initialization_samples_lms_i = min(5 * sampling_frequency, len(integrated_signal))
    initial_peak_indices_lms_i = [idx for idx in integrated_peaks_indices if idx < initialization_samples_lms_i]
    initial_peak_amplitudes_lms_i = [integrated_signal[idx] for idx in initial_peak_indices_lms_i]
    if len(initial_peak_amplitudes_lms_i) > 0:
         T_lms = np.max(initial_peak_amplitudes_lms_i) * 0.5 # Start with 50% of max initial peak as threshold guess
    else: T_lms = 1.0 # Default minimal

    # Processing Loop: Iterate through the integrated peaks
    for i, peak_index in enumerate(integrated_peaks_indices):
        peak_amplitude_integrated = integrated_signal[peak_index]
        peak_amplitude_filtered   = filtered_bandpass[peak_index]

        # --- Apply LMS-based Threshold and Validate ---
        is_qrs_lms = False
        if peak_amplitude_integrated >= T_lms:
            if abs(peak_amplitude_filtered) >= abs(threshold_f1_for_lms_validation):
                is_qrs_lms = True

        # --- Refractory period check ---
        time_since_last_qrs_lms = peak_index - last_qrs_time_lms
        if is_qrs_lms and time_since_last_qrs_lms < refractory_period_samples and last_qrs_time_lms != 0:
            is_qrs_lms = False

        # --- Record QRS and Update LMS refractory ---
        if is_qrs_lms:
            detected_qrs_lms_adaptive.append(peak_index)
            last_qrs_time_lms = peak_index

        # --- LMS Threshold Update ---
        # This updates T_lms towards the peak amplitude processed, influencing the threshold for the *next* peak
        T_lms += mu_lms * (peak_amplitude_integrated - T_lms)


    return detected_qrs_lms_adaptive


# =============================================================================
# Main Evaluation Loop
# =============================================================================

record_list = ['100', '101', '108', '203', '228']
samples_to_process = 30000
estimated_processing_delay_samples = 35
evaluation_tolerance_ms = 50 # ms

print("--- Starting Evaluation Across Records ---")
print("\nMethod                   TP    FP    FN    Sens    PPV     F1")
print("-" * 50)

for record_name in record_list:
    # =============================================================================
    # Data Loading and Pan-Tompkins Pipeline (Stages 1-5)
    # =============================================================================
    try:
        record = wfdb.rdrecord(record_name, sampfrom=0, sampto=samples_to_process, pn_dir='mitdb')
        ecg_signal = record.p_signal[:, 0]
        sampling_frequency = record.fs
        time_vector = np.arange(len(ecg_signal)) / sampling_frequency
        ann = wfdb.rdann(record_name, 'atr', sampto=len(ecg_signal), pn_dir='mitdb')
        true_beats = np.array(ann.sample)

        # Apply Pan-Tompkins Processing Pipeline
        integrated_signal, filtered_bandpass, integrated_peaks_indices, integrated_peaks_amplitudes = \
            apply_pan_tompkins_pipeline(ecg_signal, sampling_frequency)

    except Exception as e:
        print(f"Error processing record {record_name}: {e}")
        print("Skipping this record.")
        continue

    # =============================================================================
    # Apply Different Detection Methods
    # =============================================================================
    detected_pt_adaptive = detect_qrs_pan_tompkins(integrated_signal, filtered_bandpass, integrated_peaks_indices, integrated_peaks_amplitudes, sampling_frequency)
    detected_static = detect_qrs_static(integrated_signal, integrated_peaks_indices, integrated_peaks_amplitudes, sampling_frequency)
    detected_lms_adaptive = detect_qrs_lms_adaptive_threshold(integrated_signal, filtered_bandpass, integrated_peaks_indices, integrated_peaks_amplitudes, sampling_frequency)

    # =============================================================================
    # Evaluation and Printing Results
    # =============================================================================
    tp_pt, fp_pt, fn_pt, sens_pt, ppv_pt, f1_pt = score(detected_pt_adaptive, true_beats, sampling_frequency, evaluation_tolerance_ms, estimated_processing_delay_samples)
    tp_static, fp_static, fn_static, sens_static, ppv_static, f1_static = score(detected_static, true_beats, sampling_frequency, evaluation_tolerance_ms, estimated_processing_delay_samples)
    tp_lms, fp_lms, fn_lms, sens_lms, ppv_lms, f1_lms = score(detected_lms_adaptive, true_beats, sampling_frequency, evaluation_tolerance_ms, estimated_processing_delay_samples)

    # Print evaluation row for current record
    print(f"{record_name:<20} {tp_pt:4d}  {fp_pt:4d}  {fn_pt:4d}   {sens_pt:.3f}  {ppv_pt:.3f}  {f1_pt:.3f}")
    print(f"{' ':20} {tp_static:4d}  {fp_static:4d}  {fn_static:4d}   {sens_static:.3f}  {ppv_static:.3f}  {f1_static:.3f}")
    print(f"{' ':20} {tp_lms:4d}  {fp_lms:4d}  {fn_lms:4d}   {sens_lms:.3f}  {ppv_lms:.3f}  {f1_lms:.3f}")

    # =============================================================================
    # Final Plotting Comparison (Optional: Plotting every record adds many windows)
    # Uncomment if you want to see the plot for each record.
    # Or add logic to only plot specific records (e.g., record_name in ['100', '108'])
    # =============================================================================
    # plt.figure(figsize=(12, 6))
    # plt.plot(time_vector, ecg_signal, color='mediumpurple', label='Raw ECG Signal')
    # ymin_plot, ymax_plot = np.min(ecg_signal), np.max(ecg_signal)
    # if detected_pt_adaptive:
    #     times_pt = np.array(detected_pt_adaptive) / sampling_frequency
    #     plt.vlines(times_pt, ymin=ymin_plot, ymax=ymax_plot, colors='red', linestyles='--', label='Adaptive Pan-Tompkins')
    # if detected_static:
    #      times_static = np.array(detected_static) / sampling_frequency
    #      plt.vlines(times_static, ymin=ymin_plot, ymax=ymax_plot * 0.8, colors='blue', linestyles=':', label='Static Threshold')
    # if detected_lms_adaptive:
    #      times_lms = np.array(detected_lms_adaptive) / sampling_frequency
    #      plt.vlines(times_lms, ymin=ymin_plot, ymax=ymax_plot * 0.6, colors='green', linestyles='-.', label='LMS Adaptive Threshold')
    # if len(true_beats) > 0:
    #     true_beat_times = true_beats / sampling_frequency
    #     plt.vlines(true_beat_times, ymin=ymin_plot * 0.9, ymax=ymax_plot * 0.9, colors='gray', linestyles='-', linewidth=2, label='Ground Truth', alpha=0.7)
    # plt.title(f'QRS Detections Comparison - MIT-BIH Record {record_name}')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.grid(True)
    # plt.legend()
    # plt.ylim([ymin_plot, ymax_plot])
    # plt.show()


print("\n--- Evaluation complete for all records ---")
print("Script finished.")