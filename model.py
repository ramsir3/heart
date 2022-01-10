import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import time
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 770

class Encoder(nn.Module):
    def __init__(self, input_size, conv_output, hidden_size, dropout_p=0.1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.conv1_output = conv_output
        self.dropout_p = dropout_p

        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=conv_output,
            kernel_size=3072,
            stride=512,
            padding=0,
            )
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(input_size=self.conv1_output, hidden_size=hidden_size)

    def forward(self, input, hidden):
        output = self.conv1(input)
        output = self.dropout(output)
        output = torch.movedim(output, 2, 0)
        outputs = []
        for ei in output:
            output_i, hidden = self.gru(ei.unsqueeze(0), hidden)
            outputs.append(output_i)
        return torch.cat(outputs, dim=1), hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, max_length=MAX_LENGTH):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        self.attn = nn.Linear(self.hidden_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, hidden, encoder_outputs):

        attn_weights = F.softmax(
            self.attn(hidden), dim=1)

        attn_applied = torch.bmm(attn_weights, encoder_outputs)

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

    # for ei in range(input_length):
    #     print('ei:', ei)
    #     encoder_output, encoder_hidden = encoder(
    #         input_tensor[ei], encoder_hidden)
    #     encoder_outputs[ei] = encoder_output[0, 0]
    encoder_outputs, encoder_hidden = encoder(
        input_tensor, encoder_hidden)


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
    xi = X[i, :, 1:nframes+1]
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