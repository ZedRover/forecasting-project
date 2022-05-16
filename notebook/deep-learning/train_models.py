import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import missingno as msno
sys.path.append('../')
import config
from sklearn.metrics import r2_score
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
import time
SAVE_DIR='.'




def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig(SAVE_DIR+'/'+title+'.png')






def show_plot(plot_data, delta, title):
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    time_steps = list(range(-(plot_data[0].shape[0]), 0))
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, val in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel("Time-Step")
    plt.show()
    plt.savefig(SAVE_DIR+'/'+title+'.png')





def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std


if __name__=='__main__':
    DATA_DIR = '../../data/data_clean.csv'
    raw_df = pd.read_csv(DATA_DIR)
    raw_df.columns
    raw_df = raw_df.drop(['Source.Name','date','hour','tag'],axis=1)
    split_fraction = 0.725
    train_split = int(split_fraction * int(raw_df.shape[0]))
    step = 1

    past = 200
    future = 15
    learning_rate = 0.00001
    batch_size = 256
    epochs = 50
    sequence_length = 200

    targets = config.TARGET
    cities = config.CITY
    horizon = config.HORIZON
    for target in targets:
        for city in cities:
            df_city = raw_df[raw_df['type']==target+'_24h'].drop('type',axis=1)
            df_city['label'] = df_city[target].shift(-horizon)
            df = df_city.dropna()
            





def train_sub_model(df,target,city):
    features = df.drop(['label'],axis=1)
    features = normalize(features.values, train_split)
    features = pd.DataFrame(features)
    features.head()

    train_data = features.loc[0 : train_split - 1]
    val_data = features.loc[train_split:]


    start = past + future
    end = start + train_split

    x_train = train_data.values
    y_train = features.iloc[start:end][[1]]

    
    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        x_train,
        y_train,
        sequence_length=sequence_length,
        # sampling_rate=step,
        batch_size=batch_size,
    )


    x_end = len(val_data) - past - future

    label_start = train_split + past + future

    x_val = val_data.iloc[:x_end,:].values
    y_val = features.iloc[label_start:][[1]]

    dataset_val = keras.preprocessing.timeseries_dataset_from_array(
        x_val,
        y_val,
        sequence_length=sequence_length,
        # sampling_rate=step,
        batch_size=batch_size,
    )


    for batch in dataset_val.take(1):
        inputs, targets = batch

    # define model 
    inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
    lstm_out = keras.layers.LSTM(128)(inputs)
    outputs = keras.layers.Dense(1)(lstm_out)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    model.summary()
    # path_checkpoint = "model_checkpoint.h5"
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        # filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )

    history = model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_val,
        callbacks=[es_callback, 
                modelckpt_callback]
                #    tensorboard_callback],
    )

    visualize_loss(history, "Training and Validation Loss of "+target+"_"+city)

    for x, y in dataset_val.take(3):
        show_plot(
            [x[0][:, 1].numpy(), y[0].numpy(), model.predict(x)[0]],
            12,
            "Single Step Prediction of "+target+"_"+city,
        )

    pred = model.predict(dataset_val)



    y_pred = pred
    y_true = y_val.values.flatten()[-len(y_pred):]
    plt.figure(figsize=(20, 5))
    plt.plot(y_pred,label = 'pred')
    plt.plot(y_true,label='true')
    plt.legend()
    plt.title('r2_score:{:.4f}'.format(r2_score(y_true,y_pred)))
    plt.savefig(SAVE_DIR+'/'+target+'_'+city+'_'+str(round(time.time()))+'.png')


    # model.save('./models/lstm_128_dense_1.h5')