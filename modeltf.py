import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv1D, GRU, Dropout, TimeDistributed, Attention

import numpy as np

import time
import math


MAX_LENGTH = 396901 // 512 # N_FRAMES in wave files / conv1d stride length

class AttnCNNRNN(Model):

    def __init__(self, input_shape, output_size, n_filters=1, kernel_size=3072, strides=512,
        dropout_rate=0.1, gru_units=1, **kwargs):

        super().__init__(**kwargs)

        self.conv1d = Conv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            strides=strides,
            input_shape=input_shape
            )
        self.dropout = Dropout(dropout_rate)
        self.gru = GRU(gru_units, return_sequences=True)
        # self.tdgru = TimeDistributed(self.gru)

        self.attn = Attention()
        self.out = Dense(output_size, activation='softmax')

    @tf.function
    def call(self, inputs):
        output = self.conv1d(inputs)
        output = self.dropout(output)
        print(tf.shape(output))
        rnn_output, state = self.gru(output)
        output, atten_scores = self.attn(rnn_output, state, return_attention_scores=True)
        output = self.out(output)

        return output, atten_scores



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def select_XY(X, Y, i):
    nframes = int(X[0, i, 0])
    # print(i, nframes)
    xi = X[i, :, 1:nframes+1]
    xi = np.expand_dims(xi, 1)
    yi = Y[i]
    # print(nframes, xi.shape, yi.shape)
    return xi, yi