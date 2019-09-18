import csv

import math


def calculate_mean_and_variance(data):
    mean = [0, 0]
    for d in data:
        mean[0] += d[0]
        mean[1] += d[1]
    mean[0] = mean[0] / len(data)
    mean[1] = mean[1] / len(data)
    variance = [0, 0, 0]
    for d in data:
        variance[0] += pow((d[0] - mean[0]), 2)
        variance[1] += pow((d[1] - mean[1]), 2)
        variance[2] += (d[0] - mean[0]) * (d[1] - mean[1])
    variance[0] = variance[0] / len(data)
    variance[1] = variance[1] / len(data)
    variance[2] = variance[2] / len(data)
    return mean, variance


def calculate_log_likelihood_value(data, m, v):
    log_likelihood = 0
    for d in data:
        p1 = 1 / (2 * math.pi * v[2])
        p2 = math.exp(-1 / 2 * (pow(d[0] - m[0], 2) / v[0] + pow(d[1] - m[1], 2) / v[1]))
        log_likelihood = log_likelihood + math.log(p1*p2)
    return log_likelihood


data = []
with open('Dataset1.csv', 'rt')as f:
    data_txt = csv.reader(f)
    for row in data_txt:
        point = (float(row[0]), float(row[1]))
        data.append(point)

mean, variance = calculate_mean_and_variance(data)
print(calculate_log_likelihood_value(data, mean, variance))
