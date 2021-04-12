# Course-DSAI2021 HW2

## Introduction
* The goal is to decide actions - buying/holding/selling stock, maximize the profit

## Requirement
* python == 3.8
* numpy == 1.20.0
* pandas == 1.2.3
* tensorflow == 2.4.1

## Usage
python trader.py --training "Training Data" --testing "Testing Data" --output "Output File"

## Main Idea
* Extract features:
  1. `upper_shadow`: In stock market, long upper shadow usually means high pressure above.
  2. `lower_shadow`: In stock market, long lower shadow usually means strong support beneath.
  3. `day_change`: Price change in single day, to determine if open-high & close-low or open low & close high.
  4. `RSV`: (close price - lowest in 5 days) / (highest in 5 days - lowest in 5 days)
  5. `K`: RSV today * (1/3) + K yesterday * (2/3)
  6. `K_cmp`: K value change from yesterday
  7. `D`: K today * (1/3) + D yestereday * (2/3)
  8. `D_cmp`: D value change from yesterday
  9. `MA`: Moving average of stock price, here I choose 5MA
  10. `BIAS`: Close price's bias from 5MA

* From features above, I use `['upper_shadow', 'lower_shadow', 'day_change', 'yesterday_cmp', 'BIAS', 'K_cmp', 'D_cmp']` as X, input of model.

* Solution of prediction would be `[prob of rising / prob of falling]` of close price tomorrow:
  1. if tomorrow is going to rise, set [1.0, 0.0]
  2. if tomorrow is going to fall, set [0.0, 1.0]
  3. if tomorrow is not changing, set [0.5, 0.5]

* MLP model structure
  1. input layer - 7 nodes
  2. hidden layer - 19 nodes with leaky_relu
  3. hidden layer - 35 nodes with leaky_relu
  4. hidden layer - 19 nodes with leaky_relu
  5. hidden layer - 11 nodes with leaky_relu
  6. hidden layer - 7 nodes with leaky_relu
  7. output layer - 2 nodes with softmax

* Hyperparameters
  1. learning rate: 0.001, use Adam optimizer
  2. batch size: 128
  3. epochs: 2000

* Mentioned above, prediction would be `[prob of rising / prob of falling]`, here I set a threshold to make sure the chance of winning is high enough:
  1. when prob >= 0.75, buy it
  2. when prob < 0.75 & prob >= 0.5, hold
  3. otherwise, sell it

* According to training/testing dataset provided in class, this model have stable and well performance in predicting starts of uptrends, especially when the rising probability is larger than 75%.