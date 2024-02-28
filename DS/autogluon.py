from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import numpy as np
import math

train_data = TabularDataset('~/facebook_train.csv')
date, label = 'Collection_date', 'Price'
predictor = TabularPredictor(label=label).fit(
    train_data.drop(columns=date)
)

test_data = TabularDataset('~/facebook_test.csv')
preds = predictor.predict(test_data.drop(columns=[date, 'Price']))
comparsion = pd.DataFrame({'true_price' : test_data['Price'], 'pred_price':preds})

def mse(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    diff = np.subtract(actual, predicted)
    sq_diff = np.square(diff)
    return sq_diff.mean()

print(math.sqrt(mse(comparsion['true_price'], comparsion['pred_price']))) #86787.5

predictor3 = TabularPredictor(label=label).fit(
    train_data.drop(columns=date), 
    hyperparameters='multimodal',
    num_stack_levels=1, num_bag_folds=2
)

preds3 = predictor3.predict(test_data.drop(columns=[date, 'Price']))
comparsion3 = pd.DataFrame({'true_price' : test_data['Price'], 'pred_price':preds3})

print(math.sqrt(mse(comparsion3['true_price'], comparsion3['pred_price']))) #85138.6

# Second runing after more cleaning

train_data_sec = TabularDataset('~/facebook_train.csv')
label = 'Price'
predictor_sec = TabularPredictor(label=label, eval_metric='root_mean_squared_error').fit(
    train_data_sec,
    hyperparameters='multimodal',
    num_stack_levels=1, num_bag_folds=3
)

test_data_sec = TabularDataset('~/facebook_test.csv')
preds_2 = predictor_sec.predict(test_data_sec.drop(columns='Price'))
comparsion_2 = pd.DataFrame({'true_price' : test_data_sec['Price'], 'pred_price':preds_2})

def mse(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    diff = np.subtract(actual, predicted)
    sq_diff = np.square(diff)
    return sq_diff.mean()

print(math.sqrt(mse(comparsion_2['true_price'], comparsion_2['pred_price']))) #87746.3

# Third running after adding log

train_data_3 = TabularDataset('~/facebook_train.csv')
label = 'Price'
predictor_3 = TabularPredictor(label=label, eval_metric='root_mean_squared_error').fit(
    train_data_3,
    hyperparameters='multimodal',
    num_stack_levels=1, num_bag_folds=3
)

test_data_3 = TabularDataset('~/facebook_test.csv')
preds_3 = predictor_3.predict(test_data_3.drop(columns='Price'))
comparsion_3 = pd.DataFrame({'true_price' : test_data_3['Price'], 'pred_price':preds_3})

print(math.sqrt(mse(comparsion_3['true_price'], comparsion_3['pred_price']))) #90512.5

# Fourth running after filtering prices

train_data_4 = TabularDataset('~/facebook_train.csv')
label = 'Price'
predictor_4 = TabularPredictor(label=label, eval_metric='root_mean_squared_error').fit(
    train_data_4,
    hyperparameters='multimodal',
    num_stack_levels=1, num_bag_folds=3
)

test_data_4 = TabularDataset('~/facebook_test.csv')
preds_4 = predictor_4.predict(test_data_4.drop(columns='Price'))
comparsion_4 = pd.DataFrame({'true_price' : test_data_4['Price'], 'pred_price':preds_4})

print(math.sqrt(mse(comparsion_4['true_price'], comparsion_4['pred_price']))) #28220.9

# Fifty running after feature engineering

train_data_5 = TabularDataset('~/facebook_train.csv')
label = 'Price'
predictor_5 = TabularPredictor(label=label, eval_metric='root_mean_squared_error').fit(
    train_data_5,
    hyperparameters='multimodal',
    num_stack_levels=1, num_bag_folds=3
)

test_data_5 = TabularDataset('~/facebook_test.csv')
preds_5 = predictor_5.predict(test_data_5.drop(columns='Price'))
comparsion_5 = pd.DataFrame({'true_price' : test_data_5['Price'], 'pred_price':preds_5})

print(math.sqrt(mse(comparsion_5['true_price'], comparsion_5['pred_price']))) #27432.2

# Sixth running after brand extraction

train_data_6 = TabularDataset('~/facebook_train.csv')
label = 'Price'
predictor_6 = TabularPredictor(label=label, eval_metric='root_mean_squared_error').fit(
    train_data_6,
    hyperparameters='multimodal',
    num_stack_levels=1, num_bag_folds=3
)

test_data_6 = TabularDataset('~/facebook_test.csv')
preds_6 = predictor_6.predict(test_data_6.drop(columns='Price'))
comparsion_6 = pd.DataFrame({'true_price' : test_data_6['Price'], 'pred_price':preds_6})

print(math.sqrt(mse(comparsion_6['true_price'], comparsion_6['pred_price']))) #22847.7

# 7th running after brand extraction

train_data_7 = TabularDataset('~/facebook_train.csv')
label = 'Price'
predictor_7 = TabularPredictor(label=label, eval_metric='root_mean_squared_error').fit(
    train_data_7,
    hyperparameters='multimodal',
    num_stack_levels=1, num_bag_folds=3
)

test_data_7 = TabularDataset('~/facebook_test.csv')
preds_7 = predictor_7.predict(test_data_7.drop(columns='Price'))
comparsion_7 = pd.DataFrame({'true_price' : test_data_7['Price'], 'pred_price':preds_7})

print(math.sqrt(mse(comparsion_7['true_price'], comparsion_7['pred_price']))) #9455.8

# 8th all cleaned up
train_data_8 = TabularDataset('facebook_train.csv')
label = 'Price'
predictor_8 = TabularPredictor(label=label, eval_metric='root_mean_squared_error').fit(
    train_data_8,
    hyperparameters='multimodal',
    num_stack_levels=1, num_bag_folds=5
)

test_data_8 = TabularDataset('facebook_test.csv')
preds_8 = predictor_8.predict(test_data_8.drop(columns='Price'))
comparsion_8 = pd.DataFrame({'true_price' : test_data_8['Price'], 'pred_price':preds_8})

print(math.sqrt(mse(comparsion_8['true_price'], comparsion_8['pred_price']))) #8715.6

# 9th Final one
train_data_9 = TabularDataset('facebook_train.csv')
label = 'Price'
predictor_9 = TabularPredictor(label=label, eval_metric='root_mean_squared_error').fit(
    train_data_9,
    hyperparameters='multimodal',
    num_stack_levels=1, num_bag_folds=5
)

test_data_9 = TabularDataset('facebook_test.csv')
preds_9 = predictor_9.predict(test_data_9.drop(columns='Price'))
comparsion_9 = pd.DataFrame({'true_price' : test_data_9['Price'], 'pred_price':preds_9})

print(math.sqrt(mse(comparsion_9['true_price'], comparsion_9['pred_price']))) #8389.6
