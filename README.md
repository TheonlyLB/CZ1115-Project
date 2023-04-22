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
Predicting students’ dropout and academic success allows for early intervention and support to be provided to at risk students. Dropout reduces job prospects and lowers lifetime earnings, while increasing healthcare costs and reducing economic productivity for society. The problem is formulated as a three category classification task (dropout, enrolled, and graduate) for XGB and as an anomaly (dropout) detection problem for Isolation Forest.

## Exploratory data analysis (EDA)
The dataset is from the Polytechnic Institute of Portalegre related to students enrolled in different undergraduate degrees. It contains 4424 records with 35 attributes. 

There are 34 predictor variables and 1 response variable labelled "Target" which consists of categorical variables labelled 'Graduate', 'Enrolled' and 'Dropout'.

There is no ordinality in the values of variables such as ‘Mother/Father’s qualifications’ and ‘Mother/Father’s occupation’.

The data set also contains data points where the ‘Mother/Father’s occupation’ and ‘Mother/Father’s qualification' are unknown.

### Preprocessing
After identifying the incomplete data points from the EDA, we replace the unknown qualification levels and occupations with their respective median qualification levels / occupations correspondingly. 

We also normalised the numerical features with StandardScaler to reduce the differences between different columns.

## Feature Engineering with FeatureTools

FeatureTools is an open-source Python library for automated feature engineering, which automatically generates new features from our dataset by performing deep feature synthesis. Feature engineering is the process of transforming raw data into features that can be used by machine learning algorithms to improve their predictive accuracy.

## ML Methods
### XGBoost: eXtreme Gradient Boosting
XGBoost uses boosting with gradient descent to produce highly accurate results on structured data in both classification and regression problems. It is an ensemble learning method, where base learners such as shallow decision trees are trained on subsets of data and combined sequentially to correct the errors of the previous one.

### Isolation Forests
Isolation Forest shows better time complexity/scalability compared to other anomaly detection methods, as measured by the training time required/performance as data dimensionality increases. 

It is an unsupervised anomaly detection algorithm based on random forests to detect outliers in the dataset. It partitions the data points until each observation is isolated.

Regular data points require more partitions to be isolated than an anomaly data point. The number of partitions needed/anomaly score is computed for all data points and compared to a threshold value.

## Optuna Hyperparameter Tuning for XGB
Optuna is an automatic hyperparameter optimisation framework that is capable of generating the optimal set of hyperparameters.

The search spaces for the hyperparameters in Optuna tuning were set as such after several trials:

params = {<br>
        'eval_metric': 'mlogloss', <br>
        'objective': 'multi:softmax',<br>
        'max_depth': trial.suggest_int('max_depth', 5, 8),<br>
        'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 1.0, 0.05),<br>
        'n_estimators': trial.suggest_int('n_estimators', 1800, 4000, 50),<br>
        'eta': trial.suggest_loguniform('eta', 0.005, 0.1),<br>
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1e-5),<br>
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-6, 0.5),<br>
        'min_split_loss': trial.suggest_loguniform('gamma', 1e-5, 1), <br>
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 15),<br>
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.1, 0.5)}<br>

Best trial hyperparameters:<br>
    max_depth: 5<br>
    subsample: 1.0<br>
    n_estimators: 3800<br>
    eta: 0.09952475990974284<br>
    reg_alpha: 9.778562186840526e-07<br>
    reg_lambda: 0.05099229063943839<br>
    gamma: 0.0037642422353443006<br>
    min_child_weight: 8<br>
    colsample_bytree: 0.4609258070899312<br>
Resulting mlogloss: 0.5269664308760176<br>

## Results 
### XGB
Achieved mlogloss of 0.527 compared to dumb log-loss benchmark of 1.02 <br>
Accuracy of 0.79

### ISOLATION FOREST
Accuracy of 0.74

## Further analysis using XGB
Since XGB was a better, more accurate model, we used their feature importance function to generated the most important features used to predict student dropout. The results generated showed that 'Tution fees up to date', 'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (approved)' were the top 3 most important features respectively. 

## Insights
Financial factors and workload have the most impact on student dropout rates.

## Contributions:
Wynette: EDA, Parameter Tuning, Script, Video<br>
Russell: Feature Engineering, Script, Slides<br>
Zon: Models, Experimentation, Visualisation<br>

## References: 
https://www.graphpad.com/support/faq/what-is-the-difference-between-ordinal-interval-and-ratio-variables-why-should-i-care/
https://en.wikipedia.org/wiki/Education_in_Portugal
https://thebestschools.org/degrees/college-degree-levels/
https://optuna.readthedocs.io/en/stable/
https://practicaldatascience.co.uk/machine-learning/how-to-use-optuna-for-xgboost-hyperparameter-tuning


