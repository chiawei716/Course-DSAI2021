# Course-DSAI2021 HW3

## Introduction
* The goal is to decide actions - buy/sell electricity, minimize the bill

## Requirement
* python == 3.8
* numpy == 1.20.0
* pandas == 1.2.3
* tensorflow == 2.4.1

## Usage
python main.py --consumption "Consumption Data" --generation "Generation Data" --bidresult "Bidresult Data" --output "Output File"

## Main Idea
* Use two models to predict consumption and generation, and based on these two prediction, decide whether buy or sell electricity.

* MLP model structure
  1. input layer - 7 * 24 = 168 nodes
  2. hidden layer - 336 nodes with relu
  3. hidden layer - 48 nodes with relu
  4. output layer - 24 nodes with relu

* Hyperparameters
  1. Adam optimizer
  2. batch size: 64
  3. epochs: 1000

* Reset buying price to 2.4 & selling price to 2.6 everytime the program executed. Check the bidding result in past seven days:
  1. Choose highest buying price + 0.01 if not successfully trade, lowest buying price if partial & lowest buying price - 0.01 if successed
  1. Choose lowest selling price + 0.01 if not successfully trade, highest selling price if partial & highest selling price + 0.01 if successed

* If predicted generation > consumption, sell

* If predicted consumption < generation, buy