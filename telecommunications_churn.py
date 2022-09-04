# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 13:22:34 2022

@author: Nasir
"""

import pandas as pd
import sweetviz as sv
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from __future__ import division
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score,RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from numpy import mean,std
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest,chi2



df=pd.read_csv(r"C:\Users\Nasir\Desktop\Data_Science\telecommunications_churn\telecommunications_churn.csv")
df.info()
df['churn'].value_counts()
plt.pie(df['churn'].value_counts(),autopct='%.2f')
plt.show()
#sweet_report = sv.analyze(df,target_feat='churn')
#sweet_report.show_html('churn_data_EDA.html')
plt.figure(figsize=(15,15))
sns.heatmap(round(df.iloc[:,:-1].corr(),2),annot=True)
plt.show()

X = df.drop(labels='churn',axis=1)
y = df[['churn']]
X.shape,y.shape


# define Undersampling strategy
undersample = RandomUnderSampler(sampling_strategy='majority')
# fit and apply the transform
X_under, y_under = undersample.fit_resample(X, y)
X.shape,y.shape,X_under.shape,y_under.shape


# define oversampling strategy
oversample = RandomOverSampler(sampling_strategy='minority')
# fit and apply the transform
X_over, y_over = oversample.fit_resample(X, y)
X.shape,y.shape,X_over.shape, y_over.shape
plt.pie(y_over.value_counts(),autopct='%0.2f')
plt.show()
X_over['churn'] = y_over
X_over.shape
X_over.to_csv(r'C:\Users\Nasir\Desktop\Data_Science\telecommunications_churn\oversample.csv')
oversample_df=pd.read_csv(r"C:\Users\Nasir\Desktop\Data_Science\telecommunications_churn\oversample.csv")
oversample_df.info()
del oversample_df['Unnamed: 0']
#sweet_report = sv.analyze(oversample_df,target_feat='churn')
#sweet_report.show_html('churn_data_EDA.html')
plt.figure(figsize=(15,15))
sns.heatmap(round(oversample_df.iloc[:,:-1].corr(),2),annot=True)
plt.show()

std_sclr = StandardScaler()
X = oversample_df.drop(labels='churn',axis=1)
cols = X.columns
X = pd.DataFrame(std_sclr.fit_transform(X))
X.columns = cols
y = oversample_df[['churn']]
X.shape,y.shape

# create pipeline
rfe = RFECV(estimator=DecisionTreeClassifier())
model = DecisionTreeClassifier()
pipeline = Pipeline(steps=[('s',rfe),('m',model)])
# evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
rfe.fit(X,y)
X.columns[rfe.get_support()]
len(X.columns[rfe.get_support()])

rf_model = RandomForestClassifier()

rf_model.fit(X,y)
X.columns
importances = pd.DataFrame({'features':X.columns,'importance':rf_model.feature_importances_}).sort_values('importance')
importances
importances.plot(kind='barh')


X2 = df.drop(labels='churn',axis=1)
y2 = df[['churn']]
X2.shape,y2.shape
selector = SelectKBest(score_func=chi2,k=10)
new_data = selector.fit_transform(X2,y2)
new_data.shape

best10features = X.columns[selector.get_support(indices=True)]
best10features
X_new = selector.transform(X)
new_data = pd.DataFrame(X_new,columns=best10features)
new_data.shape
