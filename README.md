# CZ1115/SC1015 DSAI Mini Project

NTU AY22/23 Semester 2 CZ1115/SC1015 DSAI Mini Project - Student Success Prediction: Enhancing Retention, Performance, and Accessibility using Machine Learning


## Links


- [GitHub Repository](https://github.com/TheonlyLB/CZ1115-Project)
- [Presentation Video](https://youtu.be/vmEp6Zl6Hs0)

## Team Members

Team 1 Lab Group A121

| Name         | Github Account                                  |
|--------------|-------------------------------------------------|
| Wong Si Kei Wynette     | [wynettte](https://github.com/wynettte)          
| Phun Wei Cheng Russell | [russellphun](https://github.com/russellphun) 
| Zon Liew Hur Zhen  | [TheonlyLB](https://github.com/TheonlyLB)
  
## Describe, justify problem
Predicting students’ dropout and academic success allows for early intervention and support to be provided to at risk students. Dropout reduces job prospects and lowers lifetime earnings, while increasing healthcare costs and reducing economic productivity for society. The problem is formulated as a three category classification task (dropout, enrolled, and graduate) for XGB and as an anomaly(dropout) detection problem for Isolation Forest.

## EDA
The dataset is from the Polytechnic Institute of Portalegre related to students enrolled in different undergraduate degrees. It contains 4424 records with 35 attributes. 

There is no ordinality in the values of variables such as ‘Mother/Father’s qualifications’ and ‘Mother/Father’s occupation’.

The data set also contains data points where the ‘Mother’s occupation’ and ‘Father’s occupation are unknown.

## Preprocessing
After identifying the incomplete data points from the exploratory data analysis, we replace the unknown qualification levels with the median qualification level correspondingly. 

## Feature Engineering with FeatureTools

ML Methods
## XGBoost: eXtreme Gradient Boosting
XGBoost uses boosting with gradient descent to produce highly accurate results on structured data in both classification and regression problems. It is an ensemble learning method, where base learners such as shallow decision trees are trained on subsets of data and combined sequentially to correct the errors of the previous one.

## Isolation Forests
Isolation Forest shows better time complexity/scalability compared to other anomaly detection methods, as measured by the training time required/performance as data dimensionality increases. 

It is an unsupervised anomaly detection algorithm based on random forests to detect outliers in the dataset. It partitions the data points until each observation is isolated.

Regular data points require more partitions to be isolated than an anomaly data point. The number of partitions needed/anomaly score is computed for all data points and compared to a threshold value.

## Optuna Hyperparameter Tuning for XGB
The search spaces for the hyperparameters in Optuna tuning were set as such after several trials:
params = {
        'eval_metric': 'mlogloss',
        'objective': 'multi:softmax',
        'max_depth': trial.suggest_int('max_depth', 5, 8),
        'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 1.0, 0.05),
        'n_estimators': trial.suggest_int('n_estimators', 1800, 4000, 50),
        'eta': trial.suggest_loguniform('eta', 0.005, 0.1),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1e-5),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-6, 0.5),
        'min_split_loss': trial.suggest_loguniform('gamma', 1e-5, 1), 
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 15),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.1, 0.5)}

Best trial hyperparameters:
    max_depth: 5
    subsample: 1.0
    n_estimators: 3800
    eta: 0.09952475990974284
    reg_alpha: 9.778562186840526e-07
    reg_lambda: 0.05099229063943839
    gamma: 0.0037642422353443006
    min_child_weight: 8
    colsample_bytree: 0.4609258070899312
Resulting mlogloss: 0.5269664308760176

## Results 
XGB
Achieved mlogloss of 0.527 compared to dumb log-loss benchmark of 1.02 
Accuracy of 0.79

ISOLATION FOREST
Accuracy of 0.74

## Contributions:
Wynette: Parameter Tuning,Script, Video
Russell: EDA, Feature Engineering, Script, Slides
Zon: Models, Experimentation, Visualisation

