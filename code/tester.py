import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

fig = plt.figure()
ax = Axes3D(fig)

pts = [np.arange(1,100, i) for i in reversed(range(1,50))]


ax.scatter(pts[0], np.zeros(len(pts[0])), np.zeros(len(pts[0])))

def animate(i):
    ax.scatter(pts[i+1], np.zeros(len(pts[i+1])), np.zeros(len(pts[i+1])))


anim = FuncAnimation(
    fig, animate, interval=200, frames=len(pts) - 1)

plt.draw()
plt.show()
