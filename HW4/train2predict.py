import pandas as pd
import lightgbm as lgb

train_df = pd.read_csv('train.csv')
train_df.drop(columns=['Unnamed: 0'], inplace=True)

# Train data
X_train = train_df.loc[train_df['date_block_num'] < 33]
X_train = X_train.drop(columns=['item_cnt_month'])
y_train = train_df.loc[train_df['date_block_num'] < 33, 'item_cnt_month']

# Validation data
X_valid = train_df.loc[train_df['date_block_num'] == 33]
X_valid = X_valid.drop(columns=['item_cnt_month'])
y_valid = train_df.loc[train_df['date_block_num'] == 33, 'item_cnt_month']

# Prepare datasets
train = lgb.Dataset(X_train, y_train)
valid = lgb.Dataset(X_valid, y_valid)

# Parameters
params = {
    'metric': ['rmse'],
    'num_leaves': [400],
    'learning_rate': [0.005],
    'max_depth': [-1],
    'bagging_freq': [5],
    'random_state': [10]
}

# Start training
bst = lgb.train(
    params=params, 
    train_set=train, 
    num_boost_round=2000, 
    valid_sets=(train, valid),
    early_stopping_rounds=200,
    categorical_feature=['shop_id', 'item_id', 'item_category_id', 'shop_category', 'shop_city', 'item_category_type', 'item_category_product', 'month', 'year'],
    verbose_eval=False
)

# Save model
bst.save_model('model.txt', num_iteration=bst.best_iteration)

# Test data
X_test = train_df.loc[train_df['date_block_num'] == 34]
X_test = X_test.drop(columns=['item_cnt_month'])

# Predict result
y_test = bst.predict(X_test, num_iteration=bst.best_iteration).clip(0, 20)

# Output
data = [[i, y_test[i]] for i in range(len(y_test))]
submission_df = pd.DataFrame(data, columns=['ID', 'item_cnt_month'])
submission_df.to_csv('submission.csv', index=False)