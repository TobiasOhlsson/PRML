import csv
import numpy as np


def read_file():
    data = np.zeros((1000, 2))
    with open('Dataset3.csv', 'rt')as f:
        data_txt = csv.reader(f)
        for i, row in enumerate(data_txt):
            data[i][0] = row[0]
            data[i][1] = row[1]

        return data


def question3I():
    data = read_file()
    pca(data)


def center_data(data):
    means = np.mean(data, axis=0)
    for i in range(len(data)):
        data[i][0] = data[i][0] - means[0]
        data[i][1] = data[i][1] - means[1]
    return data


def pca(data):
    data = center_data(data)
    c = np.zeros((2, 2))
    for d in data:
        a = np.zeros((2, 2))
        a[0][0] = d[0]*d[0]
        a[0][1] = d[1]*d[0]
        a[1][0] = d[0]*d[1]
        a[1][1] = d[1]*d[1]
        c = np.add(a, c)
    print(c)
    c = np.divide(c, len(data))
    print(c)
    k = len(data)
    d = 1 / k * np.eye(k, k)



question3I()
