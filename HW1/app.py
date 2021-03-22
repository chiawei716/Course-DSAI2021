# !/usr/bin/python
# coding:utf-8

if __name__ == '__main__':

    # Argument parsing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                        default='training_data.csv',
                        help='input training data file name')
    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()

    # Import modules
    import datetime
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR

    # Read dataset
    df = pd.read_csv(args.training, sep=',')

    # Convert date column into type DateTime
    df.loc[:, '日期'] = pd.to_datetime(df.loc[:, '日期'], format='%Y%m%d')
    df = df.set_index('日期')

    # Feature extraction
    df_target = df.loc[:, ['備轉容量(MW)']]
    df_target.loc[:, 'Yesterday'] = df_target.loc[:, '備轉容量(MW)'].shift()
    df_target.loc[:, 'Yesterday_diff'] = df_target.loc[:, 'Yesterday'].diff()
    df_target.loc[:, 'LastWeek'] = df_target.loc[:, '備轉容量(MW)'].shift(7)
    df_target.loc[:, 'LastMonth'] = df_target.loc[:, '備轉容量(MW)'].shift(30)
    df_target = df_target.dropna()

    X_train = df_target.drop(['備轉容量(MW)'], axis=1).reset_index()
    y_train = df_target.loc[:, '備轉容量(MW)'].reset_index()

    # Train with data in 2020
    model = MLPRegressor(hidden_layer_sizes=(200, ), activation='relu', solver='adam', alpha=0.001, learning_rate='adaptive', max_iter=10000, early_stopping=True, verbose=False)
    tscv = TimeSeriesSplit(n_splits=100, test_size=1)
    for train_idx, test_idx in tscv.split(X_train):
        model.fit(X_train.loc[train_idx, ['Yesterday', 'Yesterday_diff', 'LastWeek', 'LastMonth']], y_train.loc[train_idx, '備轉容量(MW)'])
    
    # Predict
    X_result = df_target
    X_result = X_result.reset_index()
    for i in range(60):
        date = X_result.iloc[-1]['日期'] + datetime.timedelta(days=1)
        yesterday = X_result.iloc[-1]['備轉容量(MW)']
        yesterday_diff = X_result.iloc[-1]['備轉容量(MW)'] - X_result.iloc[-2]['備轉容量(MW)']
        last_week = X_result.iloc[-7]['備轉容量(MW)']
        last_month = X_result.iloc[-30]['備轉容量(MW)']
        res = model.predict([[yesterday, yesterday_diff, last_week, last_month]])
        X_result = X_result.append({'日期':date, '備轉容量(MW)':res[0], 'Yesterday':yesterday, 'Yesterday_diff':yesterday_diff, 'LastWeek':last_week, 'LastMonth':last_month}, ignore_index=True)

    start_date = pd.to_datetime('20210323', format='%Y%m%d')
    end_date = pd.to_datetime('20210329', format='%Y%m%d')
    mask = (X_result['日期'] >= start_date) & (X_result['日期'] <= end_date)
    df_output = X_result.loc[mask, ['日期', '備轉容量(MW)']]
    df_output = df_output.rename(columns={'日期':'date', '備轉容量(MW)':'operating_reserve(MW)'})
    df_output.loc[:, 'date'] = df_output.loc[:, 'date'].dt.strftime("%Y%m%d")
    
    df_output.reset_index(inplace=True, drop=True)
    df_output.to_csv('submission.csv')
