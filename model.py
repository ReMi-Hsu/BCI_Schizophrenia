import tensorflow as tf
from tensorflow import keras

'''
reference: 
https://www.tensorflow.org/tutorials/structured_data/time_series
https://kknews.cc/zh-tw/code/z5vg5oq.html
'''

# def oneD_CNN(input_shape):
#     conv_model = tf.keras.Sequential(layers=[
#         tf.keras.layers.Conv1D(filters=32, kernel_size=(3,), activation='relu', input_shape=input_shape[1:]),
#         tf.keras.layers.Dropout(rate=0.2),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(units=512, activation='relu'),
#         tf.keras.layers.Dense(units=1),
#     ], name="1DCNN")
#     return conv_model

def oneD_CNN(input_shape):
    conv_model = tf.keras.Sequential(layers=[
        tf.keras.layers.Conv1D(filters=64, kernel_size=(7,), activation='relu', input_shape=input_shape[1:]),
        tf.keras.layers.Conv1D(filters=64, kernel_size=(7,), activation='relu', input_shape=input_shape[1:]),
        tf.keras.layers.Conv1D(filters=64, kernel_size=(7,), activation='relu', input_shape=input_shape[1:]),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=128, kernel_size=(7,), activation='relu', input_shape=input_shape[1:]),
        tf.keras.layers.Conv1D(filters=128, kernel_size=(7,), activation='relu', input_shape=input_shape[1:]),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
    ], name="1DCNN_v1")
    return conv_model

def LSTM():
    multi_lstm_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, lstm_units].
        # Adding more `lstm_units` just overfits more quickly.
        tf.keras.layers.LSTM(32, return_sequences=False),
        # Shape => [batch, out_steps*features].
        tf.keras.layers.Dense(OUT_STEPS*num_features,
                            kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features].
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])
    return multi_lstm_model

def model_info(s):
    fname="model_summary.txt"
    with open(fname,'a') as f:
        print(s, file=f)

if __name__ == "__main__":
    input_shape = (-1, 9216, 65)
    model = oneD_CNN(input_shape=input_shape)
    model.build(input_shape=input_shape)
    model.summary(print_fn=model_info)
