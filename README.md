# On the Relative Value of Clustering Techniques for Unsupervised Effort-Aware Defect Prediction

## Requirements

### Experimental Environment:
- Python 3.6 (Not recommended to use higher versions as some libraries may not support them)
- R 4.3.1 (Recommended to use the latest version)

## Usage
### Get result tables:

``$ python './Code/CUDP.py'``

``$ python './Code/supervisedMethod.py'``

After running the above code, experiment results for 22 methods on cross-version validation datasets will be generated in the Output directory.

``$ python './Code/getResultData_RQ1.py'``

``$ python './Code/getResultData_RQ2.py'``

The above code is used to statistically analyze preliminary results. It generates summaries of various indicators in the Result directory.

### Calculate p-value and draw pictures
Execute the Python and R files in from RQ1/code to RQ5/code directory. Detailed steps can be found in each subdirectory.

**Note** that there is no RQ3 in our directory. This is because the results of RQ2 and RQ3 can be obtained through the same piece of code, which will reduce a certain amount of work. Therefore, we have merged the RQ3 code into RQ2, which you can see in the RQ2 directory. More detailed explanation.
### Dataset

The PROMISE data sets used in RQ1 to RQ4 are in the Data directory and in the provided CrossersionData.zip. 

The NASA and SOFTLAB data sets of our RQ5 experiment are provided in the RQ5 directory. 

When conducting experiments, you need to unzip the CrossersionData.zip data to the current directory.

### Code
- ``CUDP.py``: The 22 unsupervised clustering methods are clustered and evaluated on the PROMISE dataset.
- ``supervisedMethod.py``: The 6 supervised EADP methods are trained and evaluated on the PROMISE dataset.
- ``getResultData_RQ1.py`` and ``getResultData_RQ2.py``: In order to facilitate subsequent operations, these two files further process the evaluation results for RQ1 and RQ2 and integrate all results corresponding to each evaluation indicator into the same table.
