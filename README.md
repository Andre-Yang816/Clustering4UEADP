# On the Relative Value of Clustering Techniques for Unsupervised Effort-Aware Defect Prediction

## Requirements

### Experimental Environment:
- Python 3.6 (Not recommended to use higher versions as some libraries may not support them)
- R 4.3.1 (Recommended to use the latest version)

Library Functions
Execute requirements.txt

## Usage
### Get result tables:

``$ python './Code/CUDP.py'``

``$ python './Code/supervisedMethod.py'``

After running the above code, experiment results for 22 methods on cross-version validation datasets will be generated in the Output directory.

``$ python './Code/getResultData_RQ1.py'``

``$ python './Code/getResultData_RQ2.py'``

The above code is used to statistically analyze preliminary results. It generates summaries of various indicators in the Result directory.
These summaries are the direct sources for RQ1 and RQ2 data, which will be used for calculations and plotting in subsequent steps.

### Calculate p-value and draw pictures
Execute the Python and R files in from RQ1/code to RQ5/code directory. Detailed steps can be found in each subdirectory.

### Dataset

The PROMISE data sets used in RQ1 to RQ4 are in the Data directory and in the provided CrossersionData.zip. 

The NASA and SOFTLAB data sets of our RQ5 experiment are provided in the RQ5 directory. 

When conducting experiments, you need to unzip the CrossersionData.zip data to the current directory.
