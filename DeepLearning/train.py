import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import sklearn.metrics

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, History, CSVLogger

from loaddata import load_data
from model import oneD_CNN, LSTM

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

## dataset variable ##
dataset_root = fr"/local/SSD/bci_dataset_npy/"
npy_types = ["raw", "asr", "filter", "filter_asr"]
npy_type = npy_types[3]
## ##

## model variable ##
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001
TRAIN = True
TEST = True
load_weights = ''
model_types = ["oneD_CNN", "LSTM"]
model_type = model_types[0]
## ##

## result path ##
model_name = "oneD_CNN_v2"
result_root = "./result"
## ##

def train(model, result_record_path, x_train, x_val, y_train, y_val):
    print("="*80)
    print("Model Training")
    optimizer = keras.optimizers.Adam( learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-7, amsgrad=False)
    loss = keras.losses.BinaryCrossentropy(from_logits=True)

    callbacks = [
        ModelCheckpoint(filepath=result_record_path+'.h5', save_weights_only=True, verbose=1, save_best_only=True),
        History(),
        CSVLogger(result_record_path+"_acc.csv", append=load_weights != '')
    ]

    model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])
    history = model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          shuffle=True,
          validation_data=(x_val, y_val), callbacks=callbacks)
    print("Model Training Finish")
    print("="*80)
    return history

def plot_history(history, result_record_path):
    print("="*80)
    print("Plot Training History")
    fig_acc = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(result_record_path+'_acc.jpg')
    plt.close(fig_acc)

    fig_loss = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig(result_record_path+'_loss.jpg')
    plt.close(fig_loss)
    print("Plot Training History")
    print("="*80)

def test(model, model_name, date, result_record_path, x_test, y_test):
    print("="*80)
    print("Model Testing")
    optimizer = keras.optimizers.Adam( learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-7, amsgrad=False)
    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])

    loss, accuracy = model.evaluate(x_test, y_test)
    print("acc: {acc}, loss: {loss}".format(acc=accuracy, loss=loss))

    y_pred = model.predict(x_test)
    confusion_mat = sklearn.metrics.confusion_matrix(y_test, np.round(y_pred))

    print("True Positive for HC: {TP}".format(TP=confusion_mat[0, 0]))
    print("False Positive for HC: {FP}".format(FP=confusion_mat[0, 1]))
    print("False Negative for SZ: {FN}".format(FN=confusion_mat[1, 0]))
    print("True Negative for SZ: {TN}".format(TN=confusion_mat[1, 1]))

    with open(result_record_path+"_test_result.txt", 'w') as f:
        f.write("model_name: {model_name}, training_date={training_date}\n".format(model_name=model_name, training_date=date))
        f.write("acc: {acc}, loss: {loss}\n".format(acc=accuracy, loss=loss))
        f.write("True Positive for HC: {TP}\n".format(TP=confusion_mat[0, 0]))
        f.write("False Positive for HC: {FP}\n".format(FP=confusion_mat[0, 1]))
        f.write("False Negative for SZ: {FN}\n".format(FN=confusion_mat[1, 0]))
        f.write("True Negative for SZ: {TN}".format(TN=confusion_mat[1, 1]))
    print("Model Testing Finish")
    print("="*80)

if __name__ == "__main__":
    daytime = datetime.now()
    date = daytime.strftime("%y_%m%d_%H%M")

    # load data
    x_train, x_val, x_test, y_train, y_val, y_test = load_data(dataset_root=dataset_root, data_type=npy_types[1])
    
    # model
    if model_type == "oneD_CNN":
        model = oneD_CNN(input_shape=x_train.shape)
    elif model_type == "LSTM":
        model = LSTM(input_shape=x_train.shape)
    else:
        model = oneD_CNN(input_shape=x_train.shape)

    if load_weights != '':
        model.load_weights(filepath=load_weights,by_name=True, skip_mismatch=True)

    result_record_root = os.path.join(result_root, model_name)
    os.makedirs(result_record_root, exist_ok=True)
    result_record_path = os.path.join(result_record_root, date)

    if TRAIN:
        history = train(model=model, result_record_path=result_record_path, x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val)
        plot_history(history=history, result_record_path=result_record_path)

    if TEST:
        test(model=model, model_name=model_name, date=date, result_record_path=result_record_path, x_test=x_test, y_test=y_test)