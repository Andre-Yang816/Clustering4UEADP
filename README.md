# On the Relative Value of Clustering Techniques for Unsupervised Effort-Aware Defect Prediction

这是论文'On the Relative Value of Clustering Techniques for Unsupervised Effort-Aware Defect Prediction'的开源代码和数据。

## Requirements

实验环境： 
> Python 3.6 (不建议版本过高，可能有一些库不支持)
> R 4.3.1 (版本没有影响，建议用最新版)

库函数
执行 ``requirements.txt``
## Usage 
1. Get result tables: 
``$ python './Code/CUDP.py' `` 
``$ python './Code/supervisedMethod.py' `` 

  执行上述代码后，在Output目录下会生成22个方法在跨版本验证数据集对上的实验结果 

``$ python './Code/getResultData_RQ1.py' `` 
``$ python './Code/ getResultData_RQ2.py' `` 

  上述代码是对初步结果进行统计，在Result目录下生成各指标数据集的汇总，这也是RQ1和RQ2数据的直接来源，后续会被用于各个问题的计算和作图。 

2. calculate p-value and draw pictures 
   需要执行各RQ目录下的code子目录下的python和R文件。
