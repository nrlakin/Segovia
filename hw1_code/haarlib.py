import numpy as np
from matplotlib import pyplot as plt
import math

def trunc(data,lvl = 1):
    """
    Return truncated copy of data, truncated to power of 2 'lvl'. Defaults
    to 1 (returns first half). If col isn't given, will return all columns.
    """
    t = data[:].copy()
    n = t.shape[0]
    t[n>>lvl:] = 0
    return t

def graph_filter_stacked(data, k = 0, col = 0, title = 'filter_stacked'):
    """
    Plot signal in data plus the first k truncations of the inverse haar.
    k = 0 will plot the data and the inverse_haar(haar(data)), with no
    truncation. Will show k + 1 subplots, stacked vertically.
    """
    # Get a copy of relevant column, don't transform original data.
    x = data[:, col].copy()
    
    labels = ['signal']
    fig, axes = plt.subplots(k + 1, 1, sharex = 'col', sharey = 'col')
    fig.subplots_adjust(wspace = .01, hspace = .1)
    plts = [axes[0].plot(x)[0]]
    axes[0].set_ylabel('Accel')
    
    h = haar(x)
    for i in range(0, k + 1):
        ax = axes[i]
        plts.append(ax.plot(inverse_haar(trunc(h, i)))[0])
        labels.append('k = ' + str(i))
        ax.legend(plts, labels)
        plts = []
        labels = []
        ax.set_ylabel('Accel')

    # only show x label bottom row
    ax.set_xlabel('Time (no units given)')
    fig.suptitle(title)    
    fig.show()
        
def graph_filter(data, k = 0, col = 0, title = 'filtered'):
    # Get a copy of relevant column, don't transform original data.
    x = data[:, col].copy()
    
    labels = ['signal']
    plts = [plt.plot(x)[0]]
    
    h = haar(x)
    for i in range(0, k + 1):
        labels.append('k = ' + str(i))
        plts.append(plt.plot(inverse_haar(trunc(h, i)))[0])
    plt.xlabel('Time (no units given)')
    plt.ylabel('Acceleration')
    plt.legend(plts, labels)
    plt.title(title)
    plt.show()
    
def largest2(num):
    """
    Return largest power of 2 smaller than num.
    """
    result = 1
    while(result <= num):
        result*=2
    
    return result/2
    
def haar(arr):
    """
    Recursively calculate haar transform of a wave.  The transform is computed
    in place, so be careful. This uses more space making copies than if I'd just
    made a copy in the first place, plus the extra baggage of making log(len) 
    copies; may rewrite later.
    """
    length = arr.shape[0]
    
    if length == 1:
        return arr
    
    mid = length/2
    old = arr.copy()
    for i in range(0,mid):
        arr[i] = np.average(old[2*i:2*(i+1)])
        arr[mid + i] = old[2*i] - arr[i]
    
    arr[0:mid] = haar(arr[0:mid])
    return arr

def inverse_haar(arr):
    """
    Return inverse_haar of arr. In place, be careful. This uses more space
    making copies than if I'd just made a copy in the first place, plus the
    extra baggage of making log(len) copies; may rewrite later.
    """
    length = arr.shape[0]
    
    if length == 1:
        return arr

    mid = length/2

    arr[0:mid] = inverse_haar(arr[0:mid])
    
    old = arr.copy()

    for i in range(0,mid):
        arr[2*i] = old[i] + old[mid + i]
        arr[2*i+1] = old[i] - old[mid + i]
    
    return arr


def main():
    """
    Test function.  Plots sin, takes haar transform, and plots original,
    inverse_haar(haar(original)), and first 3 truncations.
    """
    x = [math.sin(x * math.pi * 0.05) for x in range(0, 100)]
    X = np.array(x).reshape(len(x), 1)
    graph_filter(X, 4, 0)
    graph_filter_stacked(X, 4, 0)
    
if __name__ == '__main__':
    main()