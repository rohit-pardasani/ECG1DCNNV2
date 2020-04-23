# ECG1DCNNV2
CinC Challenge 2020 (Team AI Strollers, Classification of ECG Signal using 1D CNN, Version 2)
This is version 2
# Classifier for CinC2020 using 1D CNN by AI Strollers

#### Contributors: Rohit Pardasani, Navchetan Awasthi

## Contents

This classifier uses two scripts:

* `run_12ECG_classifier.py` makes the classification of the clinical 12-Leads ECG. Add your classification code to the `run_12ECG_classifier` function. To reduce your code's run time, add any code to the `load_12ECG_model` function that you only need to run once, such as loading weights for your model.
* `driver.py` calls `load_12ECG_model` once and `run_12ECG_classifier` many times. Both functions are in `run_12ECG_classifier.py` file. This script also performs all file input and output. Please **do not** edit this script or we may be unable to evaluate your submission.

## Use

You can run this classifier by installing the packages in the `requirements.txt` file and running

    python driver.py input_directory output_directory

where `input_directory` is a directory for input data files and `output_directory` is a directory for output classification files. The PhysioNet/CinC 2020 webpage provides a training database with data files and a description of the contents and structure of these files.

## Submission

The `driver.py`and `run_12ECG_classifier.py` scripts need to be in the base or root path of the Github repository. If they are inside a subfolder, then the submission will fail.

## Details
The model file is .h5 file and it is loaded using load_12ECG_model function. It uses different thresholds for different classes unlike version 1 which uses same threshold.

