import numpy as np
import haarlib
import os
from task3 import plot_transform
import scipy.io.wavfile

def short_term_haar(x, window_len=4096, window_shift=2048):
    length = (x.shape[0]/window_shift) * window_shift
    truncx = x[:length]
    X = scipy.array([haarlib.haar(truncx[i:i+window_len].copy())
        for i in range(0, length-window_len,window_shift)])
    return scipy.absolute(X)
    
if __name__ == '__main__':
        # add some code to get correct path
        mypath = os.path.dirname(__file__)
        files = [('1.wav', 1.64, 3.15),
                 ('2.wav', 2.31, 3.55),
                 ('3.wav', 2.40, 3.80),
                 ('4.wav', 0.44, 2.33)]
                 
        for f in files:
            filepath = os.path.join(mypath, 'data', 'clips', f[0])
            rate, data = scipy.io.wavfile.read(filepath)
            # Strip out the stereo channel if present
            if (len(data.shape) > 1):
		data = data[:,0]

            # Get just the first 10 seconds as our audio signal
            x = data[0:10*rate]

            X = short_term_haar(x)
            plot_transform(X, title = f[0])