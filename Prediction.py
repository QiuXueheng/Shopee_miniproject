import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import numpy as np


def time_encode(dataframe):
    shopeedata = dataframe
    for idx in range(shopeedata.shape[0]):
        unixtime = datetime.strptime(shopeedata.get_value(idx, 'registration_time'), '%Y-%m-%d %H:%M:%S').timestamp()
        shopeedata.set_value(idx, 'registration_time', unixtime)
        if pd.isnull(shopeedata.get_value(idx, 'birthday')):
            shopeedata.set_value(idx, 'birthday', datetime(2038, 1, 1).timestamp())
        elif datetime.strptime(shopeedata.get_value(idx, 'birthday'), '%Y-%m-%d') < datetime(1970, 1, 3):
            shopeedata.set_value(idx, 'birthday', datetime(1970, 1, 3).timestamp())
        else:
            unixtime2 = datetime.strptime(shopeedata.get_value(idx, 'birthday'), '%Y-%m-%d').timestamp()
            shopeedata.set_value(idx, 'birthday', unixtime2)
    return shopeedata


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

# Use pretrained models? 0: yes, 1: no
pathselect = 0
modelpath = ['Pretrained_models', 'models']

# Construct New Input Feature Matrix
trainframe = pd.read_csv(r'.\Shopee\predict.csv')
user_profile = pd.read_csv(r'.\Shopee\user_profiles_MY.csv')
likes = pd.read_csv(r'.\Shopee\likes.csv')
vouchers = pd.read_csv(r'.\Shopee\Voucher_mechanics.csv')
activedate = pd.read_csv(r'.\Shopee\voucher_distribution_active_date.csv')

trainframe = pd.merge(trainframe, user_profile, left_on='userid', right_on='userid')

trainframe = pd.merge(trainframe, vouchers, left_on='promotionid_received', right_on='promotionid_received')

trainframe = pd.merge(trainframe, activedate,
                      left_on=['userid', 'promotionid_received', 'voucher_code_received', 'voucher_received_time'],
                      right_on=['userid', 'promotionid_received', 'voucher_code_received', 'voucher_received_time'])

for lags in range(31):
    log_temp = pd.read_csv(r'.\view_log\view_log_%d.csv' % lags, index_col=0)
    trainframe = pd.merge(trainframe, log_temp, left_on=['userid', 'voucher_received_date'],
                          right_on=['userid', 'voucher_received_date'])

trans_log = pd.read_csv(r'.\Shopee\transactions_MY.csv')
for idx in range(trans_log.shape[0]):
    for col in ['voucher_code_used', 'promotionid_used']:

        if pd.isnull(trans_log.get_value(idx, col)):
            trans_log.set_value(idx, col, 0)
        else:
            trans_log.set_value(idx, col, 1)

trans_log = trans_log.drop(['orderid', 'shopid', 'order_time'], axis=1)
transaction = trans_log.groupby(['userid']).sum().reset_index()
trainframe2 = pd.merge(trainframe, transaction, left_on=['userid'],
                       right_on=['userid'], how='left')

for idx in range(trainframe2.shape[0]):
    if trainframe2.get_value(idx, 'total_price') == 0:
        trainframe2.set_value(idx, 'promotionid_used', 0)

trainframe3 = time_encode(trainframe2)

traindata = trainframe3.drop(['voucher_code_received', 'voucher_received_date'], axis=1)

traindata = traindata.fillna(0)

traindata.to_csv('newtraindata.csv')

# Normalization and PCA
traindata = traindata.as_matrix()

X = np.concatenate((traindata[:, 4:], np.reshape(traindata[:, 2], (traindata.shape[0], 1))), axis=1)

pca = PCA(n_components=2)
mms = MinMaxScaler()

X_norm = mms.fit_transform(X)
X_lstm = lstm_reshapedata(X_norm)

X_norm_pca = pca.fit_transform(X_norm[:, 7:193])
X_final = np.concatenate((X_norm[:, 0:7], X_norm[:, 193:], X_norm_pca), axis=1)

# Use Pretrained Model to Predict
parpath = ['voucher', 'repur15', 'repur30', 'repur60', 'repur90']

for tar in range(5):

    rf = joblib.load(r'.\%s\rf_%s.pkl' % (modelpath[pathselect], parpath[tar]))
    nn = load_model(r'.\%s\nn_%s.h5' % (modelpath[pathselect], parpath[tar]))
    lstm = load_model(r'.\%s\lstm_%s.h5' % (modelpath[pathselect], parpath[tar]))

    pred_rf = rf.predict_proba(X_final)[:, 1]
    pred_lstm = lstm.predict(X_lstm)
    pred_nn = nn.predict(X_final)

    pred = (pred_lstm[:, 0] + pred_rf + pred_nn[:, 0]) / 3

    pred_label = np.zeros(len(pred))

    for i in range(len(pred)):
        if pred[i] > 0.5:
            pred_label[i] = 1
        else:
            pred_label[i] = 0
    X = np.concatenate((X, np.reshape(pred_label, (pred_label.shape[0], 1))), axis=1)

Label = np.concatenate((traindata[:, 0:2], X[:, 196:]), axis=1)

Label2 = pd.DataFrame(Label)

Ori_trainframe = pd.read_csv(r'.\Shopee\predict.csv')

Label2.columns = ['userid', 'promotionid_received', 'used?', 'repurchase_15?', 'repurchase_30?', 'repurchase_60?',
                  'repurchase_90?']

PredictFrame = pd.merge(Ori_trainframe, Label2, left_on=['userid', 'promotionid_received'],
                        right_on=['userid', 'promotionid_received'], how='left')

# Final Output CSV File
PredictFrame.to_csv('predict_result.csv', index=False)
