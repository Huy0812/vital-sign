import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')

def animate(i):
    data = pd.read_csv('data.txt', sep="\t", header = None)
    data = np.array(data)
    x_vals = data[:,0:1]
    y_vals = data[:,1:2]

    plt.cla()

    plt.plot(x_vals, y_vals, label='Channel 1')

    plt.legend(loc='upper left')
    plt.tight_layout()


ani = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.tight_layout()
plt.show()