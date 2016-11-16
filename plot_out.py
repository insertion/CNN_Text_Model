import numpy as np
import matplotlib.pyplot as plt

# A figure in matplotlib means the whole window in the user interface. Within this figure there can be subplots
# x = np.linspace(0, 10)
# numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
# x = np.arange(0., 10., 0.2)
# numpy.arange([start, ]stop, [step, ]dtype=None)
with open("log\model_out.txt") as f:
    epoch = []
    val   = []
    train  = []
    cost  = []
    for line in f:
#   1,50.40,50.26,63.02
#   2,49.60,50.81,74.56
#   3,51.20,50.57,75.53
#   4,49.60,51.36,78.30
#   5,49.60,50.73,76.42
#   6,49.60,51.58,64.84
#   7,55.90,51.53,77.13
        line = line[:-1]
        epoch.append(int(line.split(',')[0]))
        val.append(float(line.split(',')[1]))
        train.append(float(line.split(',')[2]))
        cost.append(float(line.split(',')[3]))
plt.xlabel("epoch")
plt.ylabel("rate")
plt.title("PyPlot First Example")
#plt.axis([0, 100, 0, 100])
# The axis() command in the example above takes a list of [xmin, xmax, ymin, ymax] and specifies the viewport of the axes
with plt.style.context('fivethirtyeight'):
    plt.plot(epoch,   val, label = 'val')
    plt.plot(epoch, train, label = 'train')
    #plt.plot(epoch,  cost, label = 'cost')

plt.legend(loc = 'upper left')
#plt.legend(loc = 'lower left')
plt.show()
