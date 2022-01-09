import pandas as pd
import numpy as np
import wave
import struct

set_a_path = 'data/set_a.csv'
data_path = 'data/'

seta = pd.read_csv(set_a_path)
# seta = seta.head()

data = np.zeros((len(seta), 396901))

def getlen(x, i):
    fn = data_path + x['fname']
    wf = wave.open(fn)
    frames = wf.readframes(-1)
    nframes = wf.getnframes()
    d = struct.unpack('h'*nframes, frames)
    data[i, 0] = nframes
    data[i, 1:nframes+1] = d
    x['nframes'] = nframes
    return x

class getlenfoo:
    def __init__(self):
        self.i = 0

    def __call__(self, x):
        o = getlen(x, self.i)
        self.i += 1
        return o


seta = seta.apply(getlenfoo(), axis=1)

np.save('data/seta.npy', data, allow_pickle=False)