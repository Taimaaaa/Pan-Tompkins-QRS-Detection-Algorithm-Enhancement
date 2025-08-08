# Pan-Tompkins QRS Detection Algorithm – Reproduction & Enhancement

**Authors:** Lara Foqaha, Taima Nasser, Veronica Wakileh  
**Course:** Digital Signal Processing – Birzeit University  

## Overview
This project reproduces the **classic Pan-Tompkins QRS Detection Algorithm** for ECG signal analysis and implements an enhancement using a **Least Mean Squares (LMS)–based adaptive thresholding method** to improve detection robustness in noisy environments.

The project includes:
- **Signal preprocessing pipeline** (bandpass filtering, differentiation, squaring, moving window integration).
- **Three QRS detection methods**:
  1. Adaptive Pan-Tompkins (original)
  2. Static thresholding
  3. LMS-based adaptive thresholding (proposed enhancement)
- **Performance evaluation** on clean and noisy ECG signals from the MIT-BIH Arrhythmia Database.

## Project Files
- **qrs_demo.py** – Step-by-step reproduction with visualizations
- **qrs_evaluation.py** – Batch evaluation across multiple ECG records
- **dsp-report.pdf** – Full project report with methodology & results
- **README.md** – Project documentation

## Methodology

### 1. Data Import & Preprocessing
- Import ECG signals using the `wfdb` Python package.
- Apply **bandpass filtering** (5–15 Hz) to remove baseline wander and high‑frequency noise.
- Apply a **derivative filter** to highlight rapid slope changes in QRS complexes.
- **Square** the signal to emphasize larger slope changes and remove sign ambiguity.
- Apply **moving window integration** to capture QRS width and energy information.

### 2. Detection Methods
- **Adaptive Pan‑Tompkins:** Uses dual adaptive thresholds with signal/noise peak tracking.
- **Static Threshold:** Applies a fixed threshold based on the max integrated peak amplitude.
- **LMS Adaptive Threshold (proposed):** Dynamically updates the threshold using an LMS‑based adaptation rule to respond to signal variation.

### 3. Evaluation
- Compare detections against ground‑truth annotations from the MIT‑BIH Arrhythmia Database.
- Metrics: **Sensitivity**, **Positive Predictive Value (PPV)**, and **F1 score**.
- Test on both **clean** (Records 100, 101) and **noisy** (Records 108, 203, 228) signals.

## Results Summary

- **Adaptive Pan‑Tompkins:** Balanced performance, robust to noise.
- **Static Threshold:** Poor performance in noisy signals (high FP).
- **LMS Adaptive Threshold:** High precision but lower sensitivity in highly variable/noisy data.

For detailed analysis, see [`dsp-report.pdf`](./dsp-report.pdf).

## Technologies Used

- **Programming Language:** Python  
- **Libraries:** NumPy, SciPy, Matplotlib, WFDB  
- **DSP Concepts:** Bandpass filtering, derivative filtering, squaring, moving window integration, pole-zero analysis  
- **Dataset:** MIT-BIH Arrhythmia Database (via PhysioNet)  
- **Evaluation Metrics:** Sensitivity, Positive Predictive Value (PPV), F1 Score

