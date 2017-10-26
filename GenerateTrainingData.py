import pandas as pd
import dateutil
from datetime import datetime
import os
import numpy as np
from imblearn.combine import SMOTEENN


def view_log_read(path, lags):
    action1 = pd.read_csv(path)

    action1_tempdummy = pd.get_dummies(action1['event_name'], prefix='%d' % lags)

    action1_frame = pd.merge(action1, action1_tempdummy, left_index=True, right_index=True)

    for idx in range(action1_frame.shape[0]):
        for act in action1_tempdummy.columns.values.tolist():
            if action1_frame.get_value(idx, act) == 1:
                action1_frame.set_value(idx, act, action1_frame.get_value(idx, 'count'))

    action1_frame = action1_frame.drop(['event_name', 'count'], axis=1)

    if lags > 0:
        lag_label = '%d_day_before' % lags

        action1_frame = action1_frame.groupby(['userid', lag_label]).sum().reset_index()

        action1_frame[lag_label] = pd.to_datetime(action1_frame[lag_label])

        for idx in range(action1_frame.shape[0]):
            action1_frame.set_value(idx, lag_label,
                                    action1_frame.get_value(idx, lag_label) + dateutil.relativedelta.relativedelta(
                                        days=lags))

        action1_frame[lag_label] = action1_frame[lag_label].dt.strftime('%Y-%m-%d')
        action1_frame = action1_frame.rename(columns={lag_label: 'voucher_received_date'})
    else:
        action1_frame = action1_frame.groupby(['userid', 'voucher_received_date']).sum().reset_index()

    print('log file %d has been imported' % lags)
    return action1_frame


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


trainframe = pd.read_csv(r'.\Shopee\training.csv')
user_profile = pd.read_csv(r'.\Shopee\user_profiles_MY.csv')
likes = pd.read_csv(r'.\Shopee\likes.csv')
vouchers = pd.read_csv(r'.\Shopee\Voucher_mechanics.csv')
activedate = pd.read_csv(r'.\Shopee\voucher_distribution_active_date.csv')

trainframe = pd.merge(trainframe, user_profile, left_on='userid', right_on='userid')

trainframe = pd.merge(trainframe, vouchers, left_on='promotionid_received', right_on='promotionid_received')

trainframe = pd.merge(trainframe, activedate,
                      left_on=['userid', 'promotionid_received', 'voucher_code_received', 'voucher_received_time'],
                      right_on=['userid', 'promotionid_received', 'voucher_code_received', 'voucher_received_time'])

if not os.path.exists(r'.\view_log'):
    os.makedirs(r'.\view_log')

for lags in range(31):
    log_temp = view_log_read(r'.\Shopee\view_log_%d.csv' % lags, lags)
    log_temp.to_csv(r'.\view_log\view_log_%d.csv' % lags)
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
trainframe = pd.merge(trainframe, transaction, left_on=['userid'],
                      right_on=['userid'], how='left')

for idx in range(trainframe.shape[0]):
    if trainframe.get_value(idx, 'total_price') == 0:
        trainframe.set_value(idx, 'promotionid_used', 0)

trainframe = time_encode(trainframe)

trainframe = trainframe.drop(['userid', 'promotionid_received', 'voucher_code_received', 'voucher_received_date'],
                             axis=1)
trainframe = trainframe.fillna(0)

trainframe.to_csv('traindata.csv')

