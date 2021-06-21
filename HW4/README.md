# Course-DSAI2021 HW4

## Introduction
* The goal is to [predict future sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales)

## Requirement
* python == 3.8
* pandas==1.1.3
* numpy==1.19.2
* lightgbm==3.2.1
* scikit_learn==0.24.2
* gdown==3.13.0

## Usage

### Installation
```bash
$ pip install -r requirements.txt
```

### Preprocessing (Option 1)
```bash
$ unzip data.zip
$ python preprocess.py
# Will generate train.csv at current directory
```

### Download preprocessed data (Option 2)
```bash
$ gdown --id 1xlgfYJGMzavxo5NXAId0JiweP2pxXyT5
$ unzip train.csv.zip
```

### Training & Predicting
```bash
$ python train2predict.py
# Will generate model.txt & submission.csv at current directory
```

## Report
[link](https://docs.google.com/document/d/1GtrJatMWT9Mp5T2jy6WzfHB_ZKDxhECcLYDB6Fhvl5A/edit?usp=sharing)