# Repo link: https://github.com/evidentlyai/evidently/blob/main/examples/sample_notebooks/evidently_metric_presets.ipynb
import pandas as pd
import numpy as np

from sklearn import datasets, ensemble, model_selection

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metric_preset import DataQualityPreset
from evidently.metric_preset import RegressionPreset
from evidently.metric_preset import ClassificationPreset
from evidently.metric_preset import TargetDriftPreset

#Dataset for Data Quality and Integrity
adult_data = datasets.fetch_openml(name='adult', version=2, as_frame='auto')
adult = adult_data.frame

adult_ref = adult[~adult.education.isin(['Some-college', 'HS-grad', 'Bachelors'])]
adult_cur = adult[adult.education.isin(['Some-college', 'HS-grad', 'Bachelors'])]

adult_cur.iloc[:2000, 3:5] = np.nan

#Dataset for regression
housing_data = datasets.fetch_california_housing(as_frame='auto')
housing = housing_data.frame

housing.rename(columns={'MedHouseVal': 'target'}, inplace=True)
housing['prediction'] = housing_data['target'].values + np.random.normal(0, 3, housing.shape[0])

housing_ref = housing.sample(n=5000, replace=False)
housing_cur = housing.sample(n=5000, replace=False)

#Dataset for Binary Probabilistic Classifcation
bcancer_data = datasets.load_breast_cancer(as_frame='auto')
bcancer = bcancer_data.frame

bcancer_ref = bcancer.sample(n=300, replace=False)
bcancer_cur = bcancer.sample(n=200, replace=False)

bcancer_label_ref = bcancer_ref.copy(deep=True)
bcancer_label_cur = bcancer_cur.copy(deep=True)

model = ensemble.RandomForestClassifier(random_state=1, n_estimators=10)
model.fit(bcancer_ref[bcancer_data.feature_names.tolist()], bcancer_ref.target)

bcancer_ref['prediction'] = model.predict_proba(bcancer_ref[bcancer_data.feature_names.tolist()])[:, 1]
bcancer_cur['prediction'] = model.predict_proba(bcancer_cur[bcancer_data.feature_names.tolist()])[:, 1]

bcancer_label_ref['prediction'] = model.predict(bcancer_label_ref[bcancer_data.feature_names.tolist()])
bcancer_label_cur['prediction'] = model.predict(bcancer_label_cur[bcancer_data.feature_names.tolist()])

#Dataset for multiclass classifcation
iris_data = datasets.load_iris(as_frame='auto')
iris = iris_data.frame

iris_ref = iris.sample(n=150, replace=False)
iris_cur = iris.sample(n=150, replace=False)

model = ensemble.RandomForestClassifier(random_state=1, n_estimators=3)
model.fit(iris_ref[iris_data.feature_names], iris_ref.target)

iris_ref['prediction'] = model.predict(iris_ref[iris_data.feature_names])
iris_cur['prediction'] = model.predict(iris_cur[iris_data.feature_names])

data_drift_report = Report(metrics=[
    DataDriftPreset(num_stattest='ks', cat_stattest='psi', num_stattest_threshold=0.2, cat_stattest_threshold=0.2),
])

data_drift_report.run(reference_data=adult_ref, current_data=adult_cur)
data_drift_report

data_drift_report.json()

data_quality_report = Report(metrics=[
    DataQualityPreset(),
])

data_quality_report.run(reference_data=adult_ref, current_data=adult_cur)
data_quality_report

regression_performance_report = Report(metrics=[
    RegressionPreset(),
])

regression_performance_report.run(reference_data=housing_ref.sort_index(), current_data=housing_cur.sort_index())
regression_performance_report

classification_performance_report = Report(metrics=[
    ClassificationPreset(probas_threshold=0.7),
])

classification_performance_report.run(reference_data=bcancer_ref, current_data=bcancer_cur)

classification_performance_report

num_target_drift_report = Report(metrics=[
    TargetDriftPreset(num_stattest='ks', cat_stattest='psi'),
])

num_target_drift_report.run(reference_data=housing_ref, current_data=housing_cur)
num_target_drift_report

multiclass_cat_target_drift_report = Report(metrics=[
    TargetDriftPreset(num_stattest='ks', cat_stattest='psi'),
])

multiclass_cat_target_drift_report.run(reference_data=iris_ref, current_data=iris_cur)
multiclass_cat_target_drift_report

binary_cat_target_drift_report = Report(metrics=[
    TargetDriftPreset(num_stattest='ks', cat_stattest='psi'),
])

binary_cat_target_drift_report.run(reference_data=bcancer_label_ref, current_data=bcancer_label_cur)
binary_cat_target_drift_report

prob_binary_cat_target_drift_report = Report(metrics=[
    TargetDriftPreset(num_stattest='ks', cat_stattest='psi'),
])

prob_binary_cat_target_drift_report.run(reference_data=bcancer_ref, current_data=bcancer_cur)
prob_binary_cat_target_drift_report