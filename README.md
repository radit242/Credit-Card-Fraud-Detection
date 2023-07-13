# Credit-Card-Fraud-Detection

**Introduction **

Credit card fraud is a major concern for banks and financial institutions. Fraudsters use various techniques to steal credit card information and make unauthorized transactions. In this project, we will explore a dataset containing credit card transactions and build models to predict fraudulent transactions.

We will use the Kaggle dataset Credit Card Fraud Detection which contains credit card transactions made by European cardholders. The dataset consists of 284,807 transactions, out of which 492 are fraudulent. The data contains only numerical input variables which are a result of Principal Component Analysis (PCA) transformations due to confidentiality issues. The features include 'Time', 'Amount', and 'V1' through 'V28', as well as the 'Class' variable, which is the target variable indicating whether the transaction is fraudulent (1) or not (0).

In this project, we will start with exploratory data analysis (EDA) to get a better understanding of the data. Next, we will perform data processing and modeling, where we will build several classification models to predict fraudulent transactions. We will also address the issue of imbalanced classes by using undersampling. Finally, we will evaluate the performance of the models and choose the best one based on various evaluation metrics such as precision, recall, F1-score, and accuracy.

Exploratory Data Analysis

This part of analysis includes 
1. loading the csv file in a dataframe 
2. observing its shape
3. getting anover view of the types of data and observing if there is any null value present
4. getting a statistical inside of each coloums

Fraud detection 

we can observe that the dataset is highly imbalanced, with a vast majority of transactions being non-fraudulent (class 0) and a relatively small number of transactions being fraudulent (class 1). This indicates that the dataset has a class imbalance problem, which may affect the performance of a model trained on this dataset. It may be necessary to use techniques such as oversampling, undersampling, or class weighting to handle the class imbalance problem when building a model for fraud detection.

correlation

![image](https://github.com/radit242/Credit-Card-Fraud-Detection/assets/107355525/4cfea693-679c-4151-b831-d419999da208)


From the heatmap, it can be observed that there are no strong positive or negative correlations between any pairs of variables in the dataset. The strongest correlations are found:

Time and V3, with a correlation coefficient of -0.42
Amount and V2, with a correlation coefficient of -0.53
Amount and V4, with a correlation coefficient of 0.4.

Modelling 

The "Credit Card Fraud Detection" dataset has credit card transactions labeled as fraudulent or not. The dataset is imbalanced, so it needs a model that can accurately detect fraudulent transactions without wrongly flagging non-fraudulent transactions.

1. Dataset is divided into train, test, validation in 60 :20 :20 ratio 
2. since the units of measurement inside data is different for each coloums, we used to standardscalar() to standardise each measurements.StandardScaler standardizes data by giving it a mean of 0 and a standard deviation of 1, which results in a normal distribution. This technique works well when dealing with a wide range of amounts and time. To scale the data, the training set is used to initialize the fit, and the train, validation, and test sets are then scaled before running them into the models.
3. Initially we installed both RandomUnderSampling and RandomOverSampling to balance the imbalanced dataset, how ever we sticked to RandomUnderSampling as RandomOverSampling was returning a very large dataset which took longer time to train the models 
4. Models used - logistic regression, Random Forest, SVM, XGboost

Logistic Regression 

The following parameters are used for logistic Regression

The 'penalty' parameter determines the type of regularization to be applied in the logistic regression model. 

The 'tol' parameter specifies the tolerance for stopping criteria. It represents the tolerance for the change in the loss function or the coefficients between iterations. If the change in the loss or coefficients is below this tolerance, the optimization algorithm stops. 

The 'C' parameter determines the inverse of the regularization strength. A smaller 'C' value corresponds to a stronger regularization, while a larger 'C' value means weaker regularization. It basically determines the complexity of the model.  a smaller 'C' value results in stronger regularization and a simpler model. 

Random Forest

Parameters 

n_estimators:This parameter determines the number of decision trees to be created in an ensemble learning method, such as Random Forest.

criteria : The 'criteria' parameter specifies the criterion used for splitting data in decision trees. Gini quantifies the impurity or the probability of misclassification at a particular node in a decision tree.Entropy is a measure of impurity or disorder in a set of samples within a node of a decision tree or any other classification algorithm

max_depth : max_depth' determines the maximum depth or maximum number of levels that a decision tree can grow. 

min_samples_split: This parameter specifies the minimum number of samples required to split an internal node in a decision tree.

min_samples_leaf: 'min_samples_leaf' determines the minimum number of samples required to be present in a leaf node

max_features:'max_features' determines the number of features to consider when looking for the best split at each node in a decision tree. It controls the randomness in feature selection, which can improve generalization and reduce overfitting. In your case, it specifies two options: 'sqrt' (square root of the total number of features) and 'log2' (logarithm base 2 of the total number of features).


SVM

Parameters 

The 'kernel' parameter specifies the type of kernel function to be used in the SVM model. It defines the type of decision boundary created by the model. The options provided in your code are 'linear', 'poly' (polynomial), 'rbf' (Radial Basis Function), and 'sigmoid'.

The 'C' parameter determines the inverse of the regularization strength. A smaller 'C' value corresponds to a stronger regularization, while a larger 'C' value means weaker regularization. It basically determines the complexity of the model.  a smaller 'C' value results in stronger regularization and a simpler model. 

The 'gamma' parameter determines the kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels. It defines the influence of a single training example. 'scale' and 'auto' are two options for setting the gamma value. 'scale' sets gamma as 1 / (n_features * X.var()) and 'auto' sets gamma as 1 / n_features. Gamma values influence the flexibility and smoothness of the decision boundary.

Xgboost

The 'n_estimators' parameter determines the number of boosting rounds or decision trees to be created in the XGBoost model

The 'max_depth' parameter determines the maximum depth or maximum number of levels that a tree can grow in the XGBoost model. It limits the complexity and size of each decision tree.

The 'learning_rate' parameter controls the step size or shrinkage applied to each boosting iteration. It scales the contribution of each tree in the ensemble. A smaller learning rate value requires more boosting rounds to achieve the same level of model performance. In your code, it generates a list of three values evenly spaced between 0.1 and 1, representing different learning rate values to be tested.

The 'gamma' parameter is the minimum loss reduction required to make a further partition on a leaf node of the tree. It adds regularization by reducing the number of splits. 

The 'subsample' parameter determines the fraction of samples to be used for training each tree.

The 'colsample_bytree' parameter determines the fraction of features to be used for training each tree

Model Evaluation 

Recall (True Positive Rate): This metric measures the percentage of all fraudulent transactions that the model correctly identifies as fraudulent.

Precision: This metric indicates the percentage of items that the model labels as fraud that are actually fraudulent.

False Positive Rate: This metric measures the percentage of non-fraudulent transactions that the model incorrectly labels as fraudulent.

Accuracy: This metric reflects how often the model is correct in its predictions overall. However, it can be misleading in the case of imbalanced data or fraud detection.

F1 score: This metric is a combination of precision and recall, taking both false positives and false negatives into account. It's a weighted average of precision and recall and is usually more useful than accuracy, especially when dealing with uneven classes.

F1 = 2 * (precision * recall) / (precision + recall)

where

precision = TP / (TP + FP)
recall = TP / (TP + FN)
TP: True Positive (model predicts positive and it is positive)
FP: False Positive (model predicts positive but it is negative)
FN: False Negative (model predicts negative but it is positive)

Roc_auc_score :  It measures the performance of a model in terms of its ability to discriminate between positive and negative classes.

Threshold :  a threshold is a value that is used to separate the predicted probabilities or scores into the positive class and the negative class. It acts as a decision boundary for classifying the samples.

![image](https://github.com/radit242/Credit-Card-Fraud-Detection/assets/107355525/8b39c738-aeda-4790-a2cb-a0be005ee853)
![image](https://github.com/radit242/Credit-Card-Fraud-Detection/assets/107355525/aa8fdc48-a758-40ce-8d7c-c0ae545cba63)

Confusion Matrix 

Random Forest has a higher TP% than Xgboost while Xgboost has a higher TN% compared to Random Forest

![image](https://github.com/radit242/Credit-Card-Fraud-Detection/assets/107355525/e62f7074-d127-42aa-8a2b-16520b5c9968)
confusion matrix of random Forest

![image](https://github.com/radit242/Credit-Card-Fraud-Detection/assets/107355525/5beeb0ed-76f0-42ca-8005-312c369c9382)
confusion matrix of xgboost

Conclusion:
From the results, its is observed that random forest and xgboost are the best models as they have higher mean_score, roc_auc_score, f1_score, and accuracy_score compared to other models. Random Forest has a higher TP% while Xgboost has a higher TN% compared to each other.



Dataset from - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud




