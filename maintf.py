# %%
from pickletools import optimize
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# %%
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy


# %%
import pandas as pd
import numpy as np

# %%
from modeltf import AttnCNNRNN, timeSince, select_XY

# %%
import datetime
# import matplotlib.pyplot as plt
# plt.switch_backend('agg')
# import matplotlib.ticker as ticker

# %%
data = np.load('data/seta.npy')

# %%
data_y = pd.read_csv('data/set_a.csv')
data_y.head()

# %%
# Drop unlabeled
# data_x = data[~data_y['label'].isna(), :]
# data_y = data_y[~data_y['label'].isna()]
# le = LabelEncoder().fit(data_y['label'])
# data_y = le.transform(data_y['label'])

data_y = pd.get_dummies(data_y[['label']], dummy_na=True)
data_x = data[data_y['label_nan']==0, :]
data_y = data_y[data_y['label_nan']==0]
data_y = data_y.drop('label_nan', axis=1)


# %%
data_x[:, 1:] = MinMaxScaler().fit_transform(data_x[:, 1:])
data_x = np.expand_dims(data_x, 2)
# data_x = np.moveaxis(data_x, 2, 0)
# leave zero padded
data_x = data_x[:, 1:, :]

print(data_x.shape, data_y.shape)

# %%
from tensorflow.keras.layers import Dense, Conv1D, GRU, Dropout, Attention
# %%
def trainIters(X, Y, model: tf.keras.Model, epochs: int, batch_size=10, learning_rate=0.01, callbacks=None):

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    # test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    optimizer = Adam(learning_rate=learning_rate)
    lossfn = CategoricalCrossentropy()
    accfn = Accuracy()

    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)

    n_filters=1
    kernel_size=3072
    strides=512
    dropout_rate=0.1
    enc_gru_units=1
    dec_gru_units=1

    conv1d = Conv1D(
        filters=n_filters,
        kernel_size=kernel_size,
        strides=strides,
        input_shape=X.shape[1:]
        )
    dropout = Dropout(dropout_rate)
    gruenc = GRU(enc_gru_units, return_sequences=True, return_state=True)
    # tdgru = TimeDistributed(gru)

    attn = Attention()
    grudec = GRU(dec_gru_units, return_sequences=False, return_state=False)

    out = Dense(Y.shape[1], activation='softmax')

    for e in range(epochs):
        for step, (xbatch, ybatch), in enumerate(dataset.batch(batch_size)):
            with tf.GradientTape() as tape:
                # output, _ = model(xbatch)
                output = conv1d(xbatch)
                output = dropout(output)
                rnn_output, state = gruenc(output)
                output, atten_scores = attn([rnn_output, state], return_attention_scores=True)
                output = grudec(output)
                output = out(output)

                loss = lossfn(ybatch, output)
                acc = accfn(ybatch, output)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss.numpy(), step=e)
            tf.summary.scalar('accuracy', acc.numpy(), step=e)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # model.build(X.shape)
    # model.summary()

    # history = model.fit(x=X, y=Y, batch_size=batch_size, epochs=epochs, callbacks=callbacks)

# %%
# def showPlot(points):
#     plt.figure()
#     fig, ax = plt.subplots()
#     # this locator puts ticks at regular intervals
#     loc = ticker.MultipleLocator(base=0.2)
#     ax.yaxis.set_major_locator(loc)
#     plt.plot(points)

# %%
nhidden = 100
model = AttnCNNRNN(data_x.shape[1:], data_y.shape[1])

# %%
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
trainIters(data_x, data_y, model, int(1e3), callbacks=[tensorboard_callback])
