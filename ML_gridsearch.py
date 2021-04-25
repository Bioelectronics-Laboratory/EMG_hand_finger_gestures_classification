import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


#############################################################################
'''
Here, the datasets we used in the paper can not be released for personal information protection.
Instead, you can identify a sample dataset.
please refer to "sample_dataset.csv"
'''

# Load dataset
data = pd.read_csv("your_own_dataset.csv")
X_data = data.drop(["label"], axis=1)
y_data = data["label"]

X_data = X_data.values
y_data = y_data.values

# split into training and test datasets
x_trainval, x_test, y_trainval, y_test = train_test_split(X_data, y_data,
                                                          test_size=0.1,
                                                          stratify=y_data,
                                                          random_state=1004)

# normalize features
scaler = StandardScaler()
scaler.fit(x_trainval)
x_trainval = scaler.transform(x_trainval)
x_test = scaler.transform(x_test)

#############################################################################

from sklearn.svm import SVC

svm = SVC()

##### predefine candidate for finding best hyperparameter - Support Vector Machine
parameter = {'kernel' : ['linear', 'rbf'], 
             "C": [1, 10, 100, 1000], 
             'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4]}

grid_search = GridSearchCV(estimator= svm, 
                           param_grid=parameter, n_jobs=-1, cv=10)
grid_search.fit(x_trainval, y_trainval)

svm_test_score = grid_search.score(x_test, y_test)
print("SVM Test Score : {0}".format(svm_test_score))

result = grid_search.cv_results_
df_result = pd.DataFrame(result)
df_result.to_csv("./svm_grid_result.csv", encoding="utf-8-sig", index=False)

#############################################################################

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier()

##### predefine candidate for finding best hyperparameter - Random Forest
parameter = {'n_estimators' : [100, 500, 1000], 
             "class_weight" : ["balanced", "balanced_subsample", None]}

grid_search = GridSearchCV(estimator= random_forest, 
                           param_grid=parameter, n_jobs=-1, cv=10)
grid_search.fit(x_trainval, y_trainval)

rf_test_score = grid_search.score(x_test, y_test)
print("RF Test Score : {0}".format(rf_test_score))

result = grid_search.cv_results_
df_result = pd.DataFrame(result)
df_result.to_csv("./rf_grid_result.csv", encoding="utf-8-sig", index=False)

#############################################################################

from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression()

##### predefine candidate for finding best hyperparameter - Logistic Regression
parameter = {'penalty' : ['l1', 'l2', 'elasticnet', 'none'], 
             'C' : [1, 1e-1, 1e-2, 1e-3, 1e-4], 
             'class_weight' : ['balanced', None], 
             'solver' : ['lbfgs', 'saga']}

grid_search = GridSearchCV(estimator= logistic, 
                           param_grid=parameter, n_jobs=-1, cv=10)
grid_search.fit(x_trainval, y_trainval)

logistic_test_score = grid_search.score(x_test, y_test)
print("LR Test Score : {0}".format(logistic_test_score))

result = grid_search.cv_results_
df_result = pd.DataFrame(result)
df_result.to_csv("./lr_grid_result.csv", encoding="utf-8-sig", index=False)