import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import time
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 396900

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)

    def forward(self, input, hidden):
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.attn = nn.Linear(self.hidden_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, hidden, encoder_outputs):

        attn_weights = F.softmax(
            self.attn(hidden), dim=1)

        try:
            attn_applied = torch.bmm(attn_weights,
                                 encoder_outputs.unsqueeze(0))
        except RuntimeError as e:
            print(4, attn_weights.shape, encoder_outputs.shape)
            raise e

        output = attn_applied
        output = self.attn_combine(output)
        output = self.out(output)
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

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

def train(input_tensor, target, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    # encoder_outputs, encoder_hidden = encoder(
    #     input_tensor, encoder_hidden)


    decoder_hidden = encoder_hidden

    decoder_output, decoder_hidden, decoder_attention = decoder(
        decoder_hidden, encoder_outputs)

    loss = criterion(decoder_output.squeeze(0), target)

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()

def tensorFromArr(X, Y, i):
    nframes = int(X[0, i, 0])
    # print(i, nframes)
    xi = X[1:nframes+1, i, :]
    xi = np.expand_dims(xi, 1)
    yi = Y[i]
    xo = torch.tensor(xi,
        dtype=torch.float,
        device=device)
    yo = torch.tensor(yi,
        dtype=torch.long,
        device=device).view(1)
    # print(nframes, xo.shape, yo.shape)
    return xo, yo