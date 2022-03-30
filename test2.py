import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

# Initial plot
x = np.arange(0., 10., 0.2)
y = np.arange(0., 10., 0.2)
line, = ax.plot(x, y)

plt.rcParams["figure.figsize"] = (10,8)
plt.ylabel("Price")
plt.xlabel("Size (sq.ft)")
plt.plot([1, 1.2, 3], [3, 3.5, 4.7], 'go', label='Training data')
#ax.plot(test_house_size, test_house_price, 'mo', label='Testing data')

def animate(i):
    x = np.arange(0., 6, 0.05)
    line.set_xdata(x)  # update the data
    line.set_ydata( x ** (1 + (i/10.0)))  # update the data

    return line,


# Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(y)
    return line,

ani = animation.FuncAnimation(fig, animate, frames=np.arange(1, 10), init_func=init, interval=1000)
plt.show()