# Comparison of sleep-wake classification using accelerometry data from Actiwatch and Apple Watch devices
## IFN646 Biomedical Data Science - Group 9

This repository contains data and code to reproduce the results shown in the project report of group 9 of the QUT course "Biomedical Data Science". Here, the authors propose a comparison of sleep/wake classification of accelerometry data acquired from Apple Watch and Actiwatch wearable devices. The underlying data of 27 nights of actigraphy is contained in the folder `raw_data` as CSV files.

### Reproduction guide
First, make sure `pip` is present on your machine and then install all necessary dependencies by running `pip install -r requirements.txt` from the project root. The authors suggest to create a virtual environment to install the dependenies locally. Generate the pre-processed data files split into full, training and test sets by running `python preprocess.py`. This will run all pre-processing steps and then pickle the processed data for convenience in a folder called `cached_data` in the project root. Then simply run the Jupyter Notebook `submission.ipynb` which will generate all intermediate and final results.

### Authors
Matthias Eder, 11378093
Kai Guetzlaff, 10823590
Julia O'Brien, 8858381
