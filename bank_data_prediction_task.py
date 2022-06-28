# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:06:08 2022

@author: Krystyna
"""
# import libraries 
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# loading data into pandas dataframe
data = 'bank_data_prediction_task.csv'
df = pd.read_csv(data, header=[0], low_memory=False)

# data engineering: selecting only the data from campaign group, 
df = df.loc[df['test_control_flag'] == "campaign group"]

# setting the expected result to client subscribing a term deposit
df['engaged'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)
print('Ratio of people, who subscribed a term deposit after the campaign %0.4f' % df['engaged'].mean())

# selecting  features, that describe the client
continuous_features = ['age']

# splitting categorical features to separate columns
columns_to_encode = ['job', 'marital', 'education', 'default', 'housing', 
    'loan']
categorical_features = []
for col in columns_to_encode:
    encoded_df = pd.get_dummies(df[col])
    encoded_df.columns = [col.replace(' ', '.') + '.' + x for x in encoded_df.columns]
    
    categorical_features += list(encoded_df.columns)
    
    df = pd.concat([df, encoded_df], axis=1)

# combining data back into one data frame   
all_features = continuous_features + categorical_features
response = 'engaged'
sample = df[all_features + [response]]
sample.columns = [x.replace(' ', '.') for x in sample.columns]
all_features = [x.replace(' ', '.') for x in all_features]
sample.head()

# splitting the data into train/test groups
x_train, x_test, y_train, y_test = train_test_split(sample[all_features], sample[response], test_size=0.3)

# building random forest model
rf_model = RandomForestClassifier(n_estimators=200,max_depth=5)

# features
X = x_train

# output
y = y_train

# fitting model to training data
rf_model.fit(X, y)

# examining what RF thinks are important features
rf_model.feature_importances_
feature_importance_df = pd.DataFrame(list(zip(rf_model.feature_importances_, all_features)))
feature_importance_df.columns = ['feature.importance', 'feature']
featsorted = feature_importance_df.sort_values(by='feature.importance', ascending=False)

featsortedtop10 = featsorted.head(10)
featsortedtop10.plot(kind='bar', x='feature')

in_sample = rf_model.predict(x_train)
out_sample = rf_model.predict(x_test)

# evaluating the model
cm = confusion_matrix(y_test, out_sample)

print('Evaluation of Random tree classification model:')
print('Confusion matrix\n\n', cm)

print(classification_report(y_test, out_sample))

print('In-Sample Accuracy: %0.4f' % accuracy_score(y_train, in_sample))
print('Out-of-Sample Accuracy: %0.4f' % accuracy_score(y_test, out_sample))

print('In-Sample Precision: %0.4f' % precision_score(y_train, in_sample, zero_division=1))
print('Out-of-Sample Precision: %0.4f' % precision_score(y_test, out_sample, zero_division=1))

print('In-Sample Recall: %0.4f' % recall_score(y_train, in_sample, zero_division=1))
print('Out-of-Sample Recall: %0.4f' % recall_score(y_test, out_sample, zero_division=1))

# ROC and AUC curves
in_sample = rf_model.predict_proba(x_train)[:,1]
out_sample = rf_model.predict_proba(x_test)[:,1]
in_sample_fpr, in_sample_tpr, in_sample_thresholds = roc_curve(y_train, in_sample)
out_sample_fpr, out_sample_tpr, out_sample_thresholds = roc_curve(y_test, out_sample)
in_sample_roc_auc = auc(in_sample_fpr, in_sample_tpr)
out_sample_roc_auc = auc(out_sample_fpr, out_sample_tpr)
print('In-Sample AUC: %0.4f' % in_sample_roc_auc)
print('Out-Sample AUC: %0.4f' % out_sample_roc_auc)

plt.figure(figsize=(10,7))
plt.plot(
    out_sample_fpr, out_sample_tpr, color='darkorange', label='Out-Sample ROC curve (area = %0.4f)' % in_sample_roc_auc
)
plt.plot(
    in_sample_fpr, in_sample_tpr, color='navy', label='In-Sample ROC curve (area = %0.4f)' % out_sample_roc_auc
)
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.grid()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RandomForest Model ROC Curve')
plt.legend(loc="lower right")
plt.show()