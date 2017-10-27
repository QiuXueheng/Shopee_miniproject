import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from imblearn.combine import SMOTETomek
import os
import time


def acc(real, predict):
    correct = 0
    for i in range(len(real)):
        if predict[i] > 0.5:
            pre_temp = 1
        else:
            pre_temp = 0
        if pre_temp == real[i]:
            correct += 1
    return correct / len(real) * 100


def lstm_reshapedata(X):
    X_lstm = np.zeros((X.shape[0], 31, 16))
    X_final = np.zeros((X.shape[0], 31, 16))
    for i in range(X.shape[0]):
        for j in range(31):
            for k in range(16):
                if k == 0:
                    X_lstm[i, j, k] = X[i, 7 + j + k]
                elif k < 6:
                    X_lstm[i, j, k] = X[i, 37 + j * 5 + k]
                elif k < 13:
                    X_lstm[i, j, k] = X[i, k - 6]
                else:
                    X_lstm[i, j, k] = X[i, k + 180]

    for j in range(31):
        X_final[:, j, :] = X_lstm[:, 30 - j, :]
    return X_final


traindata = pd.read_csv('traindata.csv', index_col=0)
traindata = traindata.as_matrix()
parpath = ['voucher', 'repur15', 'repur30', 'repur60', 'repur90']

if not os.path.exists(r'.\models'):
    os.makedirs(r'.\models')

for tar in range(5):
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

    trainX_lstm = lstm_reshapedata(trainX_norm)
    testX_lstm = lstm_reshapedata(testX_norm)

    trainX_norm_pca = pca.fit_transform(trainX_norm[:, 7:193])
    testX_norm_pca = pca.transform(testX_norm[:, 7:193])

    trainX_final = np.concatenate((trainX_norm[:, 0:7], trainX_norm[:, 193:], trainX_norm_pca), axis=1)
    testX_final = np.concatenate((testX_norm[:, 0:7], testX_norm[:, 193:], testX_norm_pca), axis=1)

    # Neural Network
    model = Sequential()
    model.add(Dense(input_dim=trainX_final.shape[1],
                    activation='sigmoid',
                    output_dim=50))
    model.add(Dense(output_dim=1,
                    activation='sigmoid'))

    start = time.time()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print('compilation time:', time.time() - start)

    model.fit(trainX_final, trainy, batch_size=100, epochs=100, validation_split=0.05, verbose=1)

    train_pred = model.predict(trainX_final)
    test_pred = model.predict(testX_final)
    train_score_nn = np.zeros(5)
    test_score_nn = np.zeros(5)
    train_score_nn[tar] = acc(trainy, train_pred)
    test_score_nn[tar] = acc(testy, test_pred)

    model.save(r'./models/nn_%s.h5' % parpath[tar])

    # LSTM
    lstm = Sequential()
    lstm.add(LSTM(50, return_sequences=True, input_shape=(31, 16)))
    lstm.add(LSTM(100, return_sequences=False))
    lstm.add(Dense(output_dim=1,
                   activation='sigmoid'))

    start = time.time()
    lstm.compile(loss='binary_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    print('compilation time:', time.time() - start)

    lstm.fit(trainX_lstm, trainy, batch_size=100, nb_epoch=10, validation_split=0.05, verbose=1)

    train_pred = lstm.predict(trainX_lstm)
    test_pred = lstm.predict(testX_lstm)

    train_score_lstm = np.zeros(5)
    test_score_lstm = np.zeros(5)
    train_score_lstm[tar] = acc(trainy, train_pred)
    test_score_lstm[tar] = acc(testy, test_pred)

    lstm.save(r'./models/lstm_%s.h5' % parpath[tar])
