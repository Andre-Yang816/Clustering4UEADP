# On the Relative Value of Clustering Techniques for Unsupervised Effort-Aware Defect Prediction

## Requirements

### Experimental Environment:
- Python 3.6 (Not recommended to use higher versions as some libraries may not support them)
- R 4.3.1 (Version doesn't matter, but it's recommended to use the latest version)

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
Execute the Python and R files in RQ1-RQ5/code directory. Detailed steps can be found in each subdirectory.
