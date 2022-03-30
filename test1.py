import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def generate_fake_data(length: int, start: int, end: int, f: Callable) -> np.array:
    x = np.linspace(start, end, length)
    y = f(x)
    noise = 0.5 * np.random.normal(0, 1, length)
    y = y + noise
    return np.array((x, y)).T

def calculate_loss(data: np.array, w: float, b: float) -> float:
    x = data[:, 0]
    y = data[:, 1]
    loss_arr = (w * x + b - y) ** 2
    return loss_arr.sum() / len(data)

def step_gradient(data: np.array, w: float, b: float, learning_rate: float)-> tuple[float, float]:
    x = data[:, 0]
    y = data[:, 1]
    grad_w = ((w * x + b - y) * x).sum() * 2 / len(data)
    grad_b = ((w * x + b - y)).sum() * 2 / len(data)
    new_b = b - grad_b * learning_rate
    new_w = w - grad_w * learning_rate
    return new_b, new_w

def plot_result(data, w_arr, b_arr):
    x = data[:, 0]
    y = data[:, 1]
    # y_pred = w * x + b
    # plt.scatter(x, y, s=5)
    # plt.plot(x, y_pred, color='r')
    # Set chart title.
    plt.title("Linear regression test")
    # Set x, y label text.
    plt.xlabel("X")
    plt.ylabel("Y")

    # initial plot
    plt.plot(x, np.zeros(len(x)))

    def _animation_frame(i):
        y_pred = w_arr[i] * x + b_arr[i]
        plt.cla()
        plt.scatter(x, y, s=5)
        plt.plot(x, y_pred, color='red')

    anim = FuncAnimation(plt.gcf(), _animation_frame, range(len(w_arr)), interval=1)
    anim.save('myAnimation.gif', writer='imagemagick', fps=30)


if __name__ == '__main__':
    data = generate_fake_data(1000, 0, 10, lambda x: 1.5*x + 6.3)
    loss = calculate_loss(data, 0.5, 1)
    print(f'The loss at the beginning is {loss}')
    w_arr = [0.5]
    b_arr = [1]
    w = 0.5
    b = 1
    learning_rate = 0.008
    for i in range(10000):
        b, w = step_gradient(data, w, b, learning_rate)
        w_arr.append(w)
        b_arr.append(b)
    print(f'The loss after 2000 iterations is {calculate_loss(data, w, b)}')
    print(f'The final parameter is: w = {w}, b = {b}')

    plot_result(data, w_arr, b_arr)
    # plt.scatter(data[0], data[1], s=5)
    # # Set chart title.
    # plt.title("Linear regression test")
    # # Set x, y label text.
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.show()
    # pass
