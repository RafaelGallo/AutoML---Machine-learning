from pprint import pprint

import sklearn.datasets
import sklearn.metrics

import autosklearn.classification

X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)

automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=120,
                                                          per_run_time_limit=30,
                                                          tmp_folder="autosklearn_classification_example_tmp",)

automl.fit(X_train, 
           y_train, 
           dataset_name="breast_cancer")

print(automl.leaderboard())
pprint(automl.cv_results_)
pprint(automl.show_models(), indent=4)

predictions = automl.predict(X_test)

print("Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))
print("Precision", sklearn.metrics.precision_score(y_test, predictions))
print("Recall", sklearn.metrics.recall_score(y_test, predictions))

from sklearn.metrics import confusion_matrix

matrix_confusion = confusion_matrix(y_test, predictions)
matrix_confusion
