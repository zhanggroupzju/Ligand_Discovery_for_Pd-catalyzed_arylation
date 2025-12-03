# Ligand_Discovery_for_Pd-catalyzed_--arylation
---

## Description

This code uses an SVR model to predict the enantioselectivity of a palladium-catalyzed α-arylation reaction.

---

## “SVR Model.py” — User Guide

### System Requirements

The code can run on Windows and has been tested on the following system:

* **Windows 11**

  * Python version: **3.12.3**
  * Platform: **Windows-11-10.0.26100-SP0 (AMD64, 64-bit)**

---

### Installation Guide

Download and install the following programs:

* **Python (≥ 3.12)**
  Download: [https://www.python.org/](https://www.python.org/)
* **PyCharm**
  Download: [https://www.jetbrains.com/pycharm/](https://www.jetbrains.com/pycharm/)

Typical installation time on a regular desktop computer: **~15 minutes**.

---

### Usage Instructions

1. Open **“SVR Model.py”** in PyCharm and place it in the same working directory as **train_data.xlsx**, **test_data.xlsx**, and **val_data.xlsx**.

2. Install the required dependencies (run in PyCharm Terminal):

   ```
   pip install numpy==1.26.4 pandas==2.2.2 matplotlib==3.10.6 scikit-learn==1.5.1 optuna==4.4.0 joblib==1.5.1
   ```

3. Run the script.
   The code will perform the following tasks:

   * Load **train_data.xlsx**, **test_data.xlsx**, and **val_data.xlsx**
   * Perform data preprocessing
   * Train the SVR model using the training set
   * Calculate **R²_LOOCV** and **MAE_LOOCV** on the test set
   * Compute **R²** and **MAE** for both the test set and validation set
   * Plot the **calibration plot** of the model

---

## “Stratified Train–Test Split.py” — User Guide

### System requirements and installation guide

Same as for **“SVR Model.py”**.

### Usage Instructions

1. Open **’’Stratified Train–Test Split.py”** in PyCharm and place it in the same working directory as **30 exp data.xlsx**.
2. Install dependencies (installation only needs to be done once per computer).
3. Run the script.
   It will output **train_data.xlsx** and **test_data.xlsx**.


