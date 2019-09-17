import csv
import random
import numpy as np


def e_step(data, k, means, variances, phis):
    r = np.zeros((len(data), k))
    for i in range(len(data)):
        normal_dist_sum = 0
        for j in range(k):
            normal_dist_sum += phis[j]/(np.sqrt(2 * np.pi * variances[j]))*np.exp(-pow(data[i]-means[j], 2)/(2*variances[j]))
        for j in range(k):
            normal_dist_j = phis[j]/(np.sqrt(2 * np.pi * variances[j]))*np.exp(-pow(data[i]-means[j], 2)/(2*variances[j]))
            r[i][j] = normal_dist_j/normal_dist_sum
    return r


def m_step(data, k, r):
    m_c = np.zeros(k)
    means = np.zeros(k)
    var = np.zeros(k)
    phi = np.zeros(k)
    for i in range(len(data)):
        for j in range(k):
            m_c[j] += data[i]
    for j in range(k):
        phi[j] = m_c[j]/sum(m_c)
        for i, x in enumerate(data):
            means[j] += r[i][j]*x
        means[j] = means[j]/sum(m_c)
        for i, x in enumerate(data):
            var[j] += r[i][j]*pow(x-means[j], 2)
        var[j] = var[j]/sum(m_c)
    return means, var, phi


def parameter_estimation(data, k):
    means = np.zeros(k)
    var = np.zeros(k)
    phi = np.zeros(k)
    r = np.zeros((len(data), k))
    for i in range(k):
        means[i] = (random.random()*16-8)
        var[i] = 1
        phi[i] = (1/k)
    while True:
        r = e_step(data, k, means, var, phi)
        #print(r)
        means, var, phi = m_step(data, k, r)
        print(means)
        print(var)
        print(phi)
    return means, var, phi


data = []
with open('Dataset2.csv', 'rt')as f:
    data_txt = csv.reader(f)
    for row in data_txt:
        point = (float(row[0]))
        data.append(point)

parameter_estimation(data, 5)

#for k in range(10):
#    parameter_estimation(data, k)


