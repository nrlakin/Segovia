import os
from haarlib import *
from accel_example import graph_accel
import csv
from matplotlib import pyplot as plt

def load(in_file):
    f = csv.reader(open(in_file,'rb'))
    return np.array([row for row in f]).astype(np.float)

def find_extremes(interval):
    """
    Return tuple of two lists: one with the 5 most negative entries, and one
    with the 5 largest entries.  Use argpartition to avoid full sort, but call
    it twice, so may be a wash in terms of savings.
    Returned values are not sorted.
    """
    return np.argpartition(interval, 5)[:5], np.argpartition(interval, -5)[-5:]

    
def main():
    # Get path of this file.
    mypath = os.path.dirname(__file__)
    
    # Data is stored in 'data' subfolder.
    filepath = os.path.join(mypath, 'data', 'accel.csv')
    data = load(filepath)
    
    # Get largest power of 2 smaller than len(data)
    N = largest2(data.shape[0])
    
    # Truncate data set.  Will now have
    data = data[:N,:]
    plt.gcf()
#    graph_accel(haar(data[:,6].copy()))
    graph_filter(data, 4, col = 6, title = 'Column 6, Filtered Signal')
    graph_filter_stacked(data, 4, col = 6, title = 'Column 6, Filtered Signal')
    
    # Get lists of most positive and most negative values.
    lows, highs = find_extremes(data[data.shape[0]/2::,6])
    
    # Print point and associated values for lows and highs.    
    print 'Lows:'
    for i in lows:
        print str(i)

    print 'Highs:'
    for i in highs:
        print str(i) 
    
if __name__ == '__main__':
    main()