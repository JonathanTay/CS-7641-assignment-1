CS 7641 Spring 2017 Assignment 1

This file describes the structure of this assignment submission. 
The assignment code is written in Python 3.5.1. Library dependencies are: 
scikit-learn 0.18.1
numpy 0.11.1
pandas 0.19.2
matplotlib 1.5.3

Other libraries used are part of the Python standard library. 

The main folder contains the following files:
1. adult.*, madelon_*.* -> These are the original datasets, as downloaded from the UCI Machine Learning Repository http://archive.ics.uci.edu/ml/
2. datasets.hdf -> A pre-processed/cleaned up copy of the datasets. This file is created by "parse data.py"
3. "parse data.py" -> This python script pre-processes the original UCI ML repo files into a cleaner form for the experiments
4. "jtay6-analysis.pdf" -> The analysis for this assignment.
5. helpers.py -> A collection of helper functions used for this assignment
6. ANN.py -> Code for the Neural Network Experiments
7. Boosting.py -> Code for the Boosted Tree experiments
8. "Decision Tree.py" -> Code for the Decision Tree experiments
9. KNN.py -> Code for the K-nearest Neighbours experiments
10. SVM.py -> Code for the Support Vector Machine (SVM) experiments
11. plotter.py -> Code to plot the learning and validation curves in the report
12. README.txt -> This file

There is also a subfolder called "output". This folder contains the experimental results. 
Here, I use DT/ANN/BT/KNN/SVM_Lin/SVM_RBF to refer to decision trees, artificial neural networks, boosted trees, K-nearest neighbours, linear and RBF kernel SVMs respectively. A suffix of _OF indicates a deliberately "overfitted" version of the model where regularisation is turned off.
The datasets are adult/madelon referring to the two datasets used (the UCI Adult dataset and the UCI Madelon dataset)
There are 83 files in this folder. They come the following types:
1. <Algorithm>_<dataset>_reg.csv -> The validation curve tests for <algorithm> on <dataset>
2. <Algorithn>_<dataset>_LC_train.scv -> Table of # of examples vs. CV training accuracy (for 5 folds) for <algorithm> on <dataset>. For learning curves
3. <Algorithn>_<dataset>_LC_test.csv -> Table of # of examples vs. CV testing accuracy (for 5 folds) for <algorithm> on <dataset>. For learning curves
4. <Algorithm>_<dataset>_timing.csv -> Table of fraction of training set vs. training and evaluation times. If the fulll training set is of size T and a fraction f are used for training, then the evaluation set is of size (T-fT)= (1-f)T
5. ITER_base_<Algorithm>_<dataset>.csv -> Table of results for learning curves based on number of iterations/epochs.
6. ITERtestSET_<Algorithm>_<dataset>.csv -> Table showing training and test set accuracy as number of iterations/epochs is varied. NOT USED in report.
7. "test results.csv" -> Table showing the optimal hyper-parameters chosen, as well as the final accuracy on the held out test set.
8. "test results Madelon No feature selection.csv" -> Table showing the optimal hyper-parameters chosen, as well as the final accuracy on the held out test set on Madelon with feature selection turned off. (Feature selection can be turned off my removing the "Cull<X>" stages in the experiment pipelines (pipeM objects). Note that these results were done before random seeds were fixed throughout the code, so any attempt to regenerate them will be slightly different due to different random seeds.