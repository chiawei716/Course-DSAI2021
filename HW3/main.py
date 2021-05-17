from tensorflow.keras.models import load_model
import pandas as pd
from datetime import datetime, timedelta

# You should not modify this part.
def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()


def output(path, data):

    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)

    return


if __name__ == "__main__":
    args = config()

    buy_price = 2.4
    sell_price = 2.6

    df_bid = pd.read_csv(args.bidresult)
    df_bid = df_bid.loc[:, ['action', 'trade_price', 'status']]
    for i in range(df_bid.shape[0]):
        if df_bid.loc[i, 'status'] == '完全成交':
            if df_bid.loc[i, 'action'] == 'buy' and buy_price > df_bid.loc[i, 'trade_price']:
                buy_price = df_bid.loc[i, 'trade_price'] - 0.01
            elif df_bid.loc[i, 'action'] == 'sell' and sell_price < df_bid.loc[i, 'trade_price']:
                sell_price = df_bid.loc[i, 'trade_price'] + 0.01

        elif df_bid.loc[i, 'status'] == '部分成交':
            if df_bid.loc[i, 'action'] == 'buy' and buy_price > df_bid.loc[i, 'trade_price']:
                buy_price = df_bid.loc[i, 'trade_price']
            elif df_bid.loc[i, 'action'] == 'sell' and sell_price < df_bid.loc[i, 'trade_price']:
                sell_price = df_bid.loc[i, 'trade_price']
        else:
            if df_bid.loc[i, 'action'] == 'buy' and buy_price > df_bid.loc[i, 'trade_price']:
                buy_price = df_bid.loc[i, 'trade_price'] + 0.01
            elif df_bid.loc[i, 'action'] == 'sell' and sell_price < df_bid.loc[i, 'trade_price']:
                sell_price = df_bid.loc[i, 'trade_price'] - 0.01

    df_con = pd.read_csv(args.consumption)
    df_gen = pd.read_csv(args.generation)

    X_con = df_con.loc[:, 'consumption'].tolist()
    X_gen = df_gen.loc[:, 'generation'].tolist()

    model_con = load_model('con.h5')
    model_gen = load_model('gen.h5')

    y_con = model_con.predict([X_con])[0]
    y_gen = model_con.predict([X_gen])[0]

    y = y_con - y_gen

    current_time = datetime.strptime(df_con.loc[167, 'time'], "%Y-%m-%d %H:%M:%S") + timedelta(hours=1)

    data = []

    for need in y:
        if round(need) > 0:
            data.append([datetime.strftime(current_time, '%Y-%m-%d %H:%M:%S'), 'buy', 2.4, round(need)])
        elif round(need) < 0:
            data.append([datetime.strftime(current_time, '%Y-%m-%d %H:%M:%S'), 'sell', 2.6, round(need)])
        current_time = current_time + timedelta(hours=1)

    output(args.output, data)
