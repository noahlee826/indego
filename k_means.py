# Adapted from http://web.mit.edu/6.S097/www/resources/kmeans.py

import random
import numpy as np
import matplotlib.pyplot as plt

# data: list of (x, y) points
def k_means(data, k):
    zd = list(zip(*data))
    low = int(min(min(zd[0]), min(zd[1])))
    high = int(max(max(zd[0]), max(zd[1]))) + 1
    means = []
    for i in range(k):
        means.append((random.randint(low, high), random.randint(low, high)))
    labels = [-1] * len(data)
    dist = [[0.] * k] * len(data)
    diff = True
    while diff:
        num_points = [0.] * k
        new_means = [(0, 0)] * k
        old_labels = list(labels)
        for i in range(len(data)):
            (x, y) = data[i]
            min_d = float('inf')
            ind = -1
            for j in range(k):
                (a, b) = means[j]
                d = (a - x) ** 2 + (b - y) ** 2
                if d < min_d:
                    min_d = d
                    ind = j
                dist[i][j] = d
            labels[i] = ind
            num_points[ind] += 1
            new_means[ind] = (x + new_means[ind][0], y + new_means[ind][1])
        for j in range(k):
            if num_points[j] == 0:
                new_means[j] = (float('inf'), float('inf'))
            else:
                new_means[j] = (new_means[j][0] / num_points[j],
                                new_means[j][1] / num_points[j])
        means = new_means
        print("means: " + str(means))
        if old_labels == labels:
            diff = False


    print("data:")
    print(data)
    basic_colors = ['r', 'g', 'b', 'c', 'm', 'y']
    colors = basic_colors
    if k > len(basic_colors):
        colors = np.random.rand(k, 3)
    data_xs = list(zip(*data))[0]
    data_ys = list(zip(*data))[1]
    data_color = [colors[label] for label in labels]
    plt.scatter(data_xs, data_ys, c=data_color, marker='.')

    print("means:")
    print(means)
    means_xs = list(zip(*means))[0]
    means_ys = list(zip(*means))[1]
    mean_color = ['k'] * len(means_xs)
    plt.scatter(means_xs, means_ys, c=mean_color, marker='x')

    print("labels:")
    print(labels)
    return labels


def generate_data(n, low, high):
    points = []
    for i in range(n):
        x = random.randint(low, high)
        y = random.randint(low, high)
        points.append((x, y))
    return points


# Plots k_means(data, k) for each k such that max_k >= k >= 1
def k_means_multiplot(data, max_k, xlabel='', ylabel=''):
    if max_k < 1:
        return
    max_fig_width = 15  # magic number
    subplot_width = max_fig_width / (max_k)
    subplot_height = subplot_width
    plt.figure(figsize=(max_fig_width, subplot_height))
    for k in range(1, max_k + 1):
        plt.subplot(1, max_k, k)
        plt.axis('equal')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        k_means(data, k)
    plt.show()


# p = generate_data(60, -15, 15)
# k_means_multiplot(p, 8)
