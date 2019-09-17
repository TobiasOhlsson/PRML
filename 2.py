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
            m_c[j] += r[i][j]
    for j in range(k):
        phi[j] = m_c[j]/sum(m_c)
        if phi[j] == 0:
            print(r)
            print("SAD")
            exit(0)
        for i, x in enumerate(data):
            means[j] += r[i][j]*x
        means[j] = means[j]/m_c[j]
        for i, x in enumerate(data):
            var[j] += r[i][j]*pow(x-means[j], 2)
        var[j] = var[j]/m_c[j]
    return means, var, phi


def parameter_estimation(data, k):
    means = np.zeros(k)
    var = np.zeros(k)
    phi = np.zeros(k)
    r = np.zeros((len(data), k))
    for i in range(k):
        means[i] = (random.uniform(-8, 8))
        var[i] = 1
        phi[i] = (1/k)
    while True:
        r = e_step(data, k, means, var, phi)
        new_means, new_var, new_phi = m_step(data, k, r)
        if np.array_equal(means, new_means) & np.array_equal(var, new_var) & np.array_equal(phi, new_phi):
            return means, var, phi
        means, var, phi = new_means, new_var, new_phi


data = []
with open('Dataset2.csv', 'rt')as f:
    data_txt = csv.reader(f)
    for row in data_txt:
        point = (float(row[0]))
        data.append(point)

k = 4
means, var, phi = parameter_estimation(data, k)
print("For k = " + str(k))
print("means = " + str(means))
print("variance = " + str(var))
print("phi = " + str(phi))


