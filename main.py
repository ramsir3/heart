# %%
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# %%
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

# %%
import pandas as pd
import numpy as np

# %%
from model import EncoderRNN, AttnDecoder, train, timeSince, tensorFromArr, device

# %%
import time
import random
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

# %%
data = np.load('data/seta.npy')

# %%
data_y = pd.read_csv('data/set_a.csv')
data_y.head()

# %%
# Drop unlabeled
data_x = data[~data_y['label'].isna(), :]
data_y = data_y[~data_y['label'].isna()]
le = LabelEncoder().fit(data_y['label'])
data_y = le.transform(data_y['label'])


# %%
data_x[:, 1:] = MinMaxScaler().fit_transform(data_x[:, 1:])
data_x = np.expand_dims(data_x, 1)
data_x = np.moveaxis(data_x, 2, 0)

print(data_x.shape, data_y.shape)

# %%
def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for iter in range(1, n_iters + 1):
        input_tensor, target_tensor = tensorFromArr(data_x, data_y, random.choice(range(data_x.shape[1])))

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    return plot_losses

# %%
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

# %%
nhidden = 100
enc = EncoderRNN(1, nhidden).to(device)
dec = AttnDecoder(nhidden, data_y.max()+1).to(device)

# %%
pl = trainIters(enc, dec, int(1e0))

# %%
showPlot(pl)
