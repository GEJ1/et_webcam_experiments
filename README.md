# et_webcam_precision_experiment
Data Analysis for the Eye tracking precision experiment using webcam

More info on our bioRxiv:
[Web-based eye-tracking for remote cognitive assessments: The anti-saccade task as a case study](https://www.biorxiv.org/content/10.1101/2023.07.11.548447v1.abstract)

## Reproducing the figures

Requirements: Python 3.10+ with `pyxations`, `pandas`, `numpy`, `scipy`, `seaborn`, `statsmodels`, `pyarrow`.

### Figure 2 (accuracy assessment)
Run `precision_experiment/` notebooks. Raw data is loaded from the BIDS structure built by `pyxations`.

### Figure 3 (anti-saccade task)
From `antisaccade_experiment/`, run `antisaccades_pyxations_blocked.ipynb` end-to-end. The notebook:
1. Builds the BIDS dataset from `raw_data/` and computes derivatives.
2. Preprocesses each subject per block (interpolation to 30 Hz, baseline subtraction, min-max normalization, mirroring, rejection of trials with `|x| > 1.5`).
3. Aggregates error rates and RTs across subjects, both for all blocks (panels B, C) and for the first half excluding block 1 (panels D, E).
4. Picks a representative subject and plots the pixel, degrees, and normalized views (panel A).
5. Saves the composite figure to `result_plots/figure3.png` and prints the Wilcoxon rank-sum tests reported in the paper.

