import scipy
import scipy.io.wavfile
import scipy.signal as signal
import pylab
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import numpy as np
import os

HALF_WINDOW = 10
# Computes the Short-Time Fourier Transform (STFT) of a signal, with a given
# window length, and shift between adjacent windows
def stft(x, window_len=4096, window_shift=2048):
	w = scipy.hamming(window_len)
	X = scipy.array([scipy.fft(w*x[i:i+window_len])
		for i in range(0, len(x)-window_len, window_shift)])
	return scipy.absolute(X[:,0:window_len/2])
    
def window(arr, i, j):
    """
    Given numpy array and indices of an element, return numpy array representing
    square window extending HALF_WINDOW elements in every direction. I'm sure
    this could be done more efficiently with slices.
    """
    return arr[max(0, i-HALF_WINDOW):min(arr.shape[0], i+HALF_WINDOW+1),max(0, j-10):min(arr.shape[1], j+HALF_WINDOW+1)]
    
def get_peaks(signal):
    """
    Given numpy array signal, return array of same shape with boolean entries:
        True = is a peak
        False = is not a peak
    Could be much cleaner if I could figure out how to pass an array element as
    an argument to window() instead of indices i,j.
    """
    is_peak = np.empty(shape=signal.shape, dtype = np.bool)
    for i in range(0,signal.shape[0]):
        for j in range(0, signal.shape[1]):
            is_peak[i,j] = (signal[i,j] == window(signal, i, j).max())
    return is_peak
    
def peaks_to_list(peak_array):
    """
    Given boolean array mapping peaks, return list of tuples of indices where
    the array is 'true'.  Used to feed data into the peak plotting routine.
    """
    peaks = np.where(peak_array)
    return [(peaks[0][i], peaks[1][i]) for i in range(0, len(peaks[0]))]

def plot_all(X, peak_list, start, end, title = 'title'):
    fig, axes = plt.subplots(2, 1, sharex = 'col', sharey = 'col')
    fig.subplots_adjust(wspace = .01, hspace = .05)
    fig.suptitle(title)

    ax = axes[1]
    s_list, f_list = zip(*peak_list)    
    ax.plot(s_list, f_list, 'bo')    
    ymin, ymax = ax.get_ylim()    
    ax.axvline(start, color = 'red')
    ax.axvline(end, color = 'red')
    ax.set_xlabel('Window index')
    ax.set_ylabel('Transform coefficient')
    
    ax = axes[0]
    ax.imshow(scipy.log(X.T), origin='lower', aspect='auto', interpolation='nearest', norm=clrs.Normalize())
    ax.set_ylabel('Transform coefficient')
    fig.set_size_inches(8,15.5)
    fig.show()
    
# Plot a transformed signal X, i.e., if X = stft(x), then
# plot_transform(X) plots the spectrogram of x
def plot_transform(X, title = None):
	pylab.ion()
	pylab.figure()
	pylab.imshow(scipy.log(X.T), origin='lower', aspect='auto', interpolation='nearest', norm=clrs.Normalize())
	pylab.xlabel('Window index')
	pylab.ylabel('Transform coefficient')
	if title is not None:
	    pylab.title(title)
	pylab.ioff()

# Plot a list of peaks in the form [(s1, f1), (s2, f2), ...]
def plot_peaks(peak_list,start,end):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)    
    s_list, f_list = zip(*peak_list)    
    plt.plot(s_list, f_list, 'bo')    
    ymin, ymax = ax.get_ylim()    
    ax.vlines((start,end),ymin,ymax,'red')
    plt.xlabel('Window index')
    plt.ylabel('Transform coefficient')
    fig.show()
  
def flatten_peaks(peak_array, max_coeff = 1375):
    """
    Given an array, returns 1d array of max value in each row.
    Columns > max_coeff are truncated. To get list of highest peaks, feed in
    array where non-peaks have been zero'd.
    """
    # ignore high frequencies; noise
    return np.amax(peak_array[:, :max_coeff], axis = 1)
    
if __name__ == '__main__':
        # add some code to get correct path
        mypath = os.path.dirname(__file__)
        files = [('1.wav', 1.64, 3.15),
                 ('2.wav', 2.31, 3.55),
                 ('3.wav', 2.40, 3.80),
                 ('4.wav', 0.44, 2.33)]
                 
        peaks_1d = []
        for f in files:
            filepath = os.path.join(mypath, 'data', 'clips', f[0])
            rate, data = scipy.io.wavfile.read(filepath)
            # Strip out the stereo channel if present
            if (len(data.shape) > 1):
		data = data[:,0]

            # Get just the first 10 seconds as our audio signal
            x = data[0:10*rate]

            X = stft(x)
            peaks = get_peaks(X)
            peak_list = peaks_to_list(peaks)
            plot_all(X, peak_list, f[1]*rate/2048., f[2]*rate/2048., title = f[0])
            X[peaks == False] = 0

            #save 'flattened' peaks for correlation
            peaks_1d.append(flatten_peaks(X))

        # correlate signals and plot cross-correlations
        shifts = []
        plots = []
        labels = []
        fig = plt.figure()
        for i in [0, 2, 3]:
            corr = signal.correlate(peaks_1d[i], peaks_1d[1])
            plots.append(plt.plot(corr)[0])
            labels.append(files[i][0])
            shifts.append(np.argmax(corr) - (peaks_1d[1].shape[0] - 1))
            
        plt.title('Cross-correlation with 2.wav')
        plt.legend(plots,labels)
        plt.show()
        
        print 'Offsets from 2.wav in seconds:'
        for i in range(0,3):
            print labels[i] + ': '+ str(float(shifts[i]*2048)/rate)
#	plot_transform(X)

# Save the figure we just plotted as a .png
#	pylab.savefig('spectrogram.png')

# Plot some dummy peaks
#	plot_peaks([(100, 50), (200, 87), (345, 20)],150,200)
#        peaks = get_peaks(X)
#        plot_peaks(peaks,150,200)
#	pylab.savefig('peaks.png')

# Wait for the user to continue (exiting the script closes all figures)
#	input('Press [Enter] to finish')