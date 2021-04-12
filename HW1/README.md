# Course-DSAI2021 HW1

## Introduction
* The goal is to predict the operating reserve of electrical power from 2021/3/23 to 2021/3/29.
* Dataset is provided at [政府資料開放平臺-台灣電力公司_過去電力供需資訊](https://data.gov.tw/dataset/19995).

## Requirement
* python == 3.8
* scikit-learn == 0.24.1
* numpy == 1.20.0
* pandas == 1.2.3

## Usage
python app.py --training "Your Training Data" --output submission.csv

## Main Idea
* Extract features:
  1. operating reserve of yesterday
  2. gain/reduce of operating reserve of yesterday
  3. operating reserve of last week (7 days ago)
  4. operating reserve of last month (30 days ago)
* Use sklearn.model_selection.TimeSeriesSplit to split dataset into different time based series.
* Use sklearn.neural_network.MLPRegressor as the model for training and predicting.
* The dataset has data from 2020/1/1 to 2021/1/31. All data is used for training, and then predictions are made to predict next one by one until 2021/3/29