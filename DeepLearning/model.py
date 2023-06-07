import tensorflow as tf
from tensorflow import keras

'''
reference: 
https://www.tensorflow.org/tutorials/structured_data/time_series
https://kknews.cc/zh-tw/code/z5vg5oq.html
'''

def oneD_CNN(input_shape):
    conv_model = tf.keras.Sequential(layers=[
        tf.keras.layers.Conv1D(filters=32, kernel_size=(7,), activation='relu', input_shape=input_shape[1:]),
        tf.keras.layers.Conv1D(filters=32, kernel_size=(7,), activation='relu', input_shape=input_shape[1:]),
        tf.keras.layers.MaxPool1D(pool_size=2),

        tf.keras.layers.Conv1D(filters=64, kernel_size=(5,), strides=2, activation='relu', input_shape=input_shape[1:]),
        tf.keras.layers.Conv1D(filters=64, kernel_size=(5,), activation='relu', input_shape=input_shape[1:]),
        tf.keras.layers.MaxPool1D(pool_size=2),

        tf.keras.layers.Conv1D(filters=128, kernel_size=(5,), strides=2, activation='relu', input_shape=input_shape[1:]),
        tf.keras.layers.Conv1D(filters=128, kernel_size=(5,), activation='relu', input_shape=input_shape[1:]),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.GlobalAveragePooling1D(),

        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid'),
    ], name="1DCNN_v2")
    return conv_model

def LSTM(input_shape):
    multi_lstm_model = tf.keras.Sequential([
        tf.keras.layers.MaxPool1D(pool_size=4, input_shape=input_shape[1:]),
        tf.keras.layers.LSTM(units=32, return_sequences=True),
        tf.keras.layers.MaxPool1D(pool_size=4),
        tf.keras.layers.LSTM(units=64, return_sequences=True),
        tf.keras.layers.MaxPool1D(pool_size=4),
        tf.keras.layers.LSTM(units=128, return_sequences=True),
        tf.keras.layers.MaxPool1D(pool_size=4),
        tf.keras.layers.LSTM(units=256, return_sequences=False),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid'),
     ], name="LSTM_v2")
    return multi_lstm_model

def model_info(s):
    fname="model_summary.txt"
    with open(fname,'a') as f:
        print(s, file=f)

if __name__ == "__main__":
    input_shape = (-1, 9216, 65)
    model = oneD_CNN(input_shape=input_shape)
    # model = LSTM(input_shape=input_shape)
    model.build(input_shape=input_shape)
    model.summary(print_fn=model_info)