# Modules
import tensorflow as tf

# Model definition
class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=19, activation=tf.nn.leaky_relu)
        self.dense2 = tf.keras.layers.Dense(units=35, activation=tf.nn.leaky_relu)
        self.dense3 = tf.keras.layers.Dense(units=19, activation=tf.nn.leaky_relu)
        self.dense4 = tf.keras.layers.Dense(units=11, activation=tf.nn.leaky_relu)
        self.dense5 = tf.keras.layers.Dense(units=7, activation=tf.nn.leaky_relu)
        self.dense6 = tf.keras.layers.Dense(units=2)

    def call(self, inputs):         # [batch_size, 8]
        x = self.dense1(inputs)     # [batch_size, 20]
        x = self.dense2(x)     # [batch_size, 20]
        x = self.dense3(x)     # [batch_size, 20]
        x = self.dense4(x)     # [batch_size, 20]
        x = self.dense5(x)     # [batch_size, 20]
        x = self.dense6(x)     # [batch_size, 20]
        output = tf.nn.softmax(x)
        return output

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()

    # Modules
    import pandas as pd
    import numpy as np
    
    # Load training data
    training_data = pd.read_csv(args.training, names=['open', 'high', 'low', 'close'])

    for i in range(training_data.shape[0]):
        # Upper shadow
        if training_data.loc[i, 'open'] > training_data.loc[i, 'close']:
            training_data.loc[i, 'upper_shadow'] = (training_data.loc[i, 'high'] - training_data.loc[i, 'open']) / training_data.loc[i, 'open'] 
        else:
            training_data.loc[i, 'upper_shadow'] = (training_data.loc[i, 'high'] - training_data.loc[i, 'close']) / training_data.loc[i, 'open'] 
        # Lower shadow
        if training_data.loc[i, 'open'] > training_data.loc[i, 'close']:
            training_data.loc[i, 'lower_shadow'] = (training_data.loc[i, 'close'] - training_data.loc[i, 'low']) / training_data.loc[i, 'open'] 
        else:
            training_data.loc[i, 'lower_shadow'] = (training_data.loc[i, 'open'] - training_data.loc[i, 'low']) / training_data.loc[i, 'open'] 
        # Day change
        training_data.loc[i, 'day_change'] = (training_data.loc[i, 'close'] - training_data.loc[i, 'open']) / training_data.loc[i, 'open'] 
        # Find index of 5 days starter
        five_days_starter = i - 4
        if five_days_starter < 0:
            five_days_starter = 0
        # RSV
        training_data.loc[i, 'RSV'] = (training_data.loc[i, 'close'] - training_data.loc[five_days_starter:i, 'low'].min()) / (training_data.loc[five_days_starter:i, 'high'].max() - training_data.loc[five_days_starter:i, 'low'].min()) 
        # 5-MA
        training_data.loc[i, 'MA'] = training_data.loc[five_days_starter:i, 'close'].sum() / (i - five_days_starter + 1)
        # BIAS
        training_data.loc[i, 'BIAS'] = (training_data.loc[i, 'close'] - training_data.loc[i, 'MA']) / training_data.loc[i, 'close'] 

    training_data.loc[0, 'yesterday_cmp'] = 0.0
    training_data.loc[0, 'K'] = training_data.loc[0, 'RSV']
    training_data.loc[0, 'D'] = training_data.loc[0, 'K']
    training_data.loc[0, 'K_cmp'] = 0.0
    training_data.loc[0, 'D_cmp'] = 0.0
    for i in range(1, training_data.shape[0]):
        # Compare with yesterday
        training_data.loc[i, 'yesterday_cmp'] = (training_data.loc[i, 'close'] - training_data.loc[i - 1, 'close']) / training_data.loc[i - 1, 'close'] 
        # K
        training_data.loc[i, 'K'] = training_data.loc[i, 'RSV'] / 3 + training_data.loc[i - 1, 'K'] * 2 / 3
        training_data.loc[i, 'K_cmp'] = training_data.loc[i, 'K'] - training_data.loc[i - 1, 'K']
        # D
        training_data.loc[i, 'D'] = training_data.loc[i, 'K'] / 3 + training_data.loc[i - 1, 'D'] * 2 / 3
        training_data.loc[i, 'D_cmp'] = training_data.loc[i, 'D'] - training_data.loc[i - 1, 'D']
        
    col_X = ['upper_shadow', 'lower_shadow', 'day_change', 'yesterday_cmp', 'BIAS', 'K_cmp', 'D_cmp']
    
    train_X = training_data.loc[:training_data.shape[0] - 2, col_X]
    train_Y = []
    for i in range(1, training_data.shape[0]):
        if training_data.loc[i, 'yesterday_cmp'] < 0:
            train_Y.append([0.0, 1.0])
        elif training_data.loc[i, 'yesterday_cmp'] == 0:
            train_Y.append([0.5, 0.5])
        else:
            train_Y.append([1.0, 0.0])
    train_Y = np.array(train_Y)


    # Parameters
    lr = 0.001
    batch_size = 128
    num_epochs = 2000
    
    # Train
    model = Model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    num_batches = int(train_X.shape[0] // batch_size * num_epochs)
    for batch_index in range(num_batches):
        index = np.random.randint(0, train_X.shape[0], batch_size)
        X = np.array(train_X.loc[index, :])
        y = train_Y[index]
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
            if (batch_index + 1) % 5000 == 0:
                print("batch %d: loss %f" % (batch_index + 1, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    

    testing_data = pd.read_csv(args.testing, names=['open', 'high', 'low', 'close'])
    hold = 0
    with open(args.output, 'w') as output_file:
        for i in range(testing_data.shape[0] - 1):

            # Upper shadow
            if testing_data.loc[i, 'open'] > testing_data.loc[i, 'close']:
                testing_data.loc[i, 'upper_shadow'] = (testing_data.loc[i, 'high'] - testing_data.loc[i, 'open']) / testing_data.loc[i, 'open'] 
            else:
                testing_data.loc[i, 'upper_shadow'] = (testing_data.loc[i, 'high'] - testing_data.loc[i, 'close']) / testing_data.loc[i, 'open'] 
            # Lower shadow
            if testing_data.loc[i, 'open'] > testing_data.loc[i, 'close']:
                testing_data.loc[i, 'lower_shadow'] = (testing_data.loc[i, 'close'] - testing_data.loc[i, 'low']) / testing_data.loc[i, 'open'] 
            else:
                testing_data.loc[i, 'lower_shadow'] = (testing_data.loc[i, 'open'] - testing_data.loc[i, 'low']) / testing_data.loc[i, 'open'] 
            # Day change
            testing_data.loc[i, 'day_change'] = (testing_data.loc[i, 'close'] - testing_data.loc[i, 'open']) / testing_data.loc[i, 'open'] 
            # Find index of 5 days starter
            five_days_starter = i - 4
            if five_days_starter < 0:
                five_days_starter = 0
            # RSV
            testing_data.loc[i, 'RSV'] = (testing_data.loc[i, 'close'] - testing_data.loc[five_days_starter:i, 'low'].min()) / (testing_data.loc[five_days_starter:i, 'high'].max() - training_data.loc[five_days_starter:i, 'low'].min()) 
            # 5-MA
            testing_data.loc[i, 'MA'] = testing_data.loc[five_days_starter:i, 'close'].sum() / (i - five_days_starter + 1)
            # BIAS
            testing_data.loc[i, 'BIAS'] = (testing_data.loc[i, 'close'] - testing_data.loc[i, 'MA']) / testing_data.loc[i, 'close'] 
            
            if i == 0:
                testing_data.loc[0, 'yesterday_cmp'] = 0.0
                testing_data.loc[0, 'K'] = testing_data.loc[0, 'RSV']
                testing_data.loc[0, 'D'] = testing_data.loc[0, 'K']
                testing_data.loc[0, 'K_cmp'] = 0.0
                testing_data.loc[0, 'D_cmp'] = 0.0
            else:
                # Compare with yesterday
                testing_data.loc[i, 'yesterday_cmp'] = (testing_data.loc[i, 'close'] - testing_data.loc[i - 1, 'close']) / training_data.loc[i - 1, 'close'] 
                # K
                testing_data.loc[i, 'K'] = testing_data.loc[i, 'RSV'] / 3 + testing_data.loc[i - 1, 'K'] * 2 / 3
                testing_data.loc[i, 'K_cmp'] = testing_data.loc[i, 'K'] - testing_data.loc[i - 1, 'K']
                # D
                testing_data.loc[i, 'D'] = testing_data.loc[i, 'K'] / 3 + testing_data.loc[i - 1, 'D'] * 2 / 3
                testing_data.loc[i, 'D_cmp'] = testing_data.loc[i, 'D'] - testing_data.loc[i - 1, 'D']

            # Predict
            pred_y = model.predict(np.array([testing_data.loc[i, col_X]]))

            # If probability of rising > 75%, buy
            if pred_y[0][0] >= 0.75:
                if hold < 1:
                    output_file.write('1\n')
                    hold = hold + 1
                else:
                    output_file.write('0\n')
            # If between 50% ~ 75%, hold
            elif pred_y[0][0] < 0.75 and pred_y[0][0] >= 0.5:
                output_file.write('0\n')
            # If smaller than 50%, sold
            else:
                if hold > -1:
                    output_file.write('-1\n')
                    hold = hold - 1
                else:
                    output_file.write('0\n')
