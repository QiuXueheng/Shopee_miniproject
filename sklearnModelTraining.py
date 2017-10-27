import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from imblearn.combine import SMOTETomek
import os
import numpy as np

traindata = pd.read_csv('traindata.csv', index_col=0)
traindata = traindata.as_matrix()
parpath = ['voucher', 'repur15', 'repur30', 'repur60', 'repur90']

if not os.path.exists(r'.\models'):
    os.makedirs(r'.\models')

for tar in range(5):
    print(tar)
    X, y = np.concatenate((traindata[:, 7:], np.reshape(traindata[:, 0], (traindata.shape[0], 1))), axis=1), traindata[
                                                                                                             :, tar + 1]
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=True)

    # oversampling and undersampling for first two datasets
    if tar < 2:
        smote_tomek = SMOTETomek(random_state=True)
        trainX, trainy = smote_tomek.fit_sample(trainX, trainy)

    pca = PCA(n_components=2)
    mms = MinMaxScaler()

    trainX_norm = mms.fit_transform(trainX)
    testX_norm = mms.transform(testX)

    trainX_norm_pca = pca.fit_transform(trainX_norm[:, 7:193])
    testX_norm_pca = pca.transform(testX_norm[:, 7:193])

    trainX_final = np.concatenate((trainX_norm[:, 0:7], trainX_norm[:, 193:], trainX_norm_pca), axis=1)
    testX_final = np.concatenate((testX_norm[:, 0:7], testX_norm[:, 193:], testX_norm_pca), axis=1)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, criterion='entropy', n_jobs=-1,
                                random_state=True, verbose=1)

    param_range_depth = [5, 10, 15, 20, 30, 40]
    param_range_feature = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    param_grid_rf = [{'max_depth': param_range_depth,
                      'max_features': param_range_feature}]


    class run():
        def __init__(self):
            gs = GridSearchCV(estimator=rf,
                              param_grid=param_grid_rf,
                              scoring='accuracy',
                              cv=5,
                              n_jobs=1)

            gs = gs.fit(trainX_final, trainy)
            print(gs.best_estimator_)
            print(gs.best_params_)
            clf = gs.best_estimator_
            clf.fit(trainX_final, trainy)
            joblib.dump(clf, r'.\models\rf_%s.pkl' % parpath[tar])
            print('Test accuracy: %.3f' % clf.score(testX_final, testy))


    if __name__ == '__main__':
        run()

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=True)

    param_range_dt = [1, 2, 3, 4, 5, 6, 7, 10, 15, 30, 40]
    param_range_cri = ['entropy', 'gini']
    param_grid_dt = [{'max_depth': param_range_dt,
                      'criterion': param_range_cri}]


    class run2():
        def __init__(self):
            gs = GridSearchCV(estimator=dt,
                              param_grid=param_grid_dt,
                              scoring='accuracy',
                              cv=5,
                              n_jobs=1)

            gs = gs.fit(trainX_final, trainy)
            print(gs.best_estimator_)
            print(gs.best_params_)
            clf = gs.best_estimator_
            clf.fit(trainX_final, trainy)
            joblib.dump(clf, r'.\models\dt_%s.pkl' % parpath[tar])
            print('Test accuracy: %.3f' % clf.score(testX_final, testy))


    if __name__ == '__main__':
        run2()

    # Logistic Regression
    lr = LogisticRegression(random_state=True, verbose=1)

    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    param_grid = [{'C': param_range,
                   'penalty': ['l1']},
                  {'C': param_range,
                   'penalty': ['l2']}]


    class run3():
        def __init__(self):
            gs = GridSearchCV(estimator=lr,
                              param_grid=param_grid,
                              scoring='accuracy',
                              cv=5,
                              n_jobs=1)

            gs = gs.fit(trainX_final, trainy)
            print(gs.best_estimator_)
            print(gs.best_params_)
            clf = gs.best_estimator_
            clf.fit(trainX_final, trainy)
            joblib.dump(clf, r'.\models\lr_%s.pkl' % parpath[tar])
            print('Test accuracy: %.3f' % clf.score(testX_final, testy))


    if __name__ == '__main__':
        run3()
