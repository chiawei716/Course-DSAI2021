import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Read data
sales = pd.read_csv('data/sales_train.csv', parse_dates=['date'])
items = pd.read_csv('data/items.csv')
item_cates = pd.read_csv('data/item_categories.csv')
shops = pd.read_csv('data/shops.csv')
test = pd.read_csv('data/test.csv')

# Clean outlier
sales = sales[(sales['item_cnt_day'] > 0) & (sales['item_cnt_day'] < 1000)]
sales = sales[(sales['item_price'] > 0) & (sales['item_price'] < 100000)]

# Only extract data needed in test dataset
shopids = test['shop_id'].unique()
sales = sales[sales['shop_id'].isin(shopids)]

# Combine similiar shops
sales.loc[sales['shop_id'] == 0, 'shop_id'] = 57
sales.loc[sales['shop_id'] == 1, 'shop_id'] = 58
sales.loc[sales['shop_id'] == 10, 'shop_id'] = 11
sales.loc[sales['shop_id'] == 39, 'shop_id'] = 40
test.loc[test['shop_id'] == 0, 'shop_id'] = 57
test.loc[test['shop_id'] == 1, 'shop_id'] = 58
test.loc[test['shop_id'] == 10, 'shop_id'] = 11
test.loc[test['shop_id'] == 39, 'shop_id'] = 40

# Extract shops feature - city/category
types = ['ТЦ', 'ТРК', 'ТРЦ', 'МТРЦ', 'ТК']
for i in range(shops.shape[0]):
    shops.loc[i, 'city'] = shops.loc[i, 'shop_name'].split()[0]
    shops.loc[i, 'category'] = 'other'
    for _type in types:
        if shops.loc[i, 'shop_name'].find(_type) != -1:
            shops.loc[i, 'category'] = _type
            break

# Special case
shops.loc[shops['city'] =='!Якутск', 'city'] = 'Якутск'

# Numerical
shops['shop_category'] = LabelEncoder().fit_transform(shops['category'])
shops['shop_city'] = LabelEncoder().fit_transform(shops['city'])
shops = shops.loc[:, ['shop_id', 'shop_category', 'shop_city']]

# Get first sale date block of each item
items['first_sale_date_block'] = sales.groupby('item_id').agg({'date_block_num': 'min'})['date_block_num']
items['first_sale_date_block'] = items['first_sale_date_block'].fillna(34)

# Extract category feature - type/relative product
for i in range(item_cates.shape[0]):
    cate = item_cates.loc[i, 'item_category_name'].split(' - ')
    item_cates.loc[i, 'type'] = cate[0]
    if len(cate) == 1:
        item_cates.loc[i, 'product'] = 'other'
    else:
        item_cates.loc[i, 'product'] = cate[1]

# Numerical
item_cates['item_category_type'] = LabelEncoder().fit_transform(item_cates['type'])
item_cates['item_category_product'] = LabelEncoder().fit_transform(item_cates['product'])
item_cates = item_cates.loc[:, ["item_category_id", "item_category_type", "item_category_product"]]

# Extract transaction feature
sales['trans_month'] = sales['item_cnt_day'] * sales['item_price']

# Group by date_block/item_id/shop_id
sales = sales.groupby(by=['date_block_num', 'item_id', 'shop_id'],as_index=False).agg({
    'item_cnt_day': 'sum',
    'trans_month': 'sum',
    'item_price': 'mean'
})
sales['item_cnt_month'] = sales['item_cnt_day']
sales.drop(columns='item_cnt_day', inplace=True)

# Fill missing months
month_n = 34
train_df = []
for i in range(month_n):
    _shopids = sales.loc[sales['date_block_num'] == i]['shop_id'].unique()
    _itemids = sales.loc[sales['date_block_num'] == i]['item_id'].unique()
    for shopid in _shopids:
        for itemid in _itemids:
            train_df.append([i, shopid, itemid])
train_df = pd.DataFrame(train_df, columns=['date_block_num', 'shop_id', 'item_id'])
train_df = train_df.merge(
    sales,
    how='left',
    on=['date_block_num', 'shop_id', 'item_id']
).fillna(0)

# Concat test data
test.drop(columns=['ID'], inplace=True)
test['date_block_num'] = 34
train_df = pd.concat([train_df, test], ignore_index=True)
train_df = train_df.fillna(0)

# Merge features
train_df = train_df.merge(items, how='left', on='item_id')
train_df = train_df.merge(shops, how='left', on='shop_id')
train_df = train_df.merge(item_cates, how='left', on='item_category_id')

# Extract month
train_df['month'] = train_df['date_block_num'] % 12 + 1
train_df['year'] = train_df['date_block_num'] // 12 + 2013

# Extract time since start saling
train_df['on_sale_time'] = train_df['date_block_num'] - train_df['first_sale_date_block']

# Get mean of item_cnt_month group by provided index
def get_month_cnt_mean(cols, suffixes):
    df = train_df[cols + ['item_cnt_month']].groupby(cols).mean()
    df = train_df.merge(df, how='left', on=cols, suffixes=suffixes)
    return df
train_df = get_month_cnt_mean(['date_block_num', 'item_category_id', 'shop_id'], ('', '_mean_cate_shop'))
train_df = get_month_cnt_mean(['date_block_num', 'item_id'], ('', '_mean_item'))
train_df = get_month_cnt_mean(['date_block_num', 'item_id', 'shop_city'], ('', '_mean_item_city'))

# Get lag data (1, 2, 3 months)
def get_lag_feature(target):
    lags = [1, 2, 3]
    for lag in lags:
        col_name = target + '_lag' + str(lag)
        train_df[col_name] = train_df.sort_values('date_block_num').groupby(['shop_id', 'item_id'])[target].shift(lag)
        train_df[col_name].fillna(0, inplace=True)
get_lag_feature('item_cnt_month')
get_lag_feature('item_price')
get_lag_feature('item_cnt_month_mean_cate_shop')
get_lag_feature('item_cnt_month_mean_item')
get_lag_feature('item_cnt_month_mean_item_city')

cnt_cols = []
for col in train_df.columns:
    if '_cnt' in col:
        cnt_cols.append(col)
        
for col in cnt_cols:
    train_df[col] = train_df[col].clip(0, 20)

# Get mean of item_cnt_month lags
train_df['item_cnt_month_lag_mean'] = train_df[['item_cnt_month_lag1', 'item_cnt_month_lag2', 'item_cnt_month_lag3']].mean(axis=1)
train_df['item_cnt_month_lag_mean'].fillna(0, inplace=True)

# Get slope of lag features
train_df['slope1'] = train_df['item_cnt_month_lag1'] / train_df['item_cnt_month_lag2']
train_df['slope1'] = train_df['slope1'].replace([np.inf, -np.inf], np.nan)
train_df['slope1'] = train_df['slope1'].fillna(0)
train_df['slope2'] = train_df['item_cnt_month_lag2'] / train_df['item_cnt_month_lag3']
train_df['slope2'] = train_df['slope2'].replace([np.inf, -np.inf], np.nan)
train_df['slope2'] = train_df['slope2'].fillna(0)

# Drop first three date blocks
train_df = train_df.loc[train_df['date_block_num'] >= 3]

# Drop useless columns
drop_cols = ['item_price', 'trans_month','item_name', 'first_sale_date_block','item_cnt_month_mean_cate_shop', 'item_cnt_month_mean_item', 'item_cnt_month_mean_item_city']
train_df.drop(columns=drop_cols, inplace=True)

train_df.to_csv('train.csv')