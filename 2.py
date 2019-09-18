import csv
import random
import numpy as np


# e step of the Algorithm: Given the current parameters calculate the membership probabilities for each data point.
def e_step(data, k, means, variances, phis):
    r = np.zeros((len(data), k))
    for i in range(len(data)):
        normal_dist_sum = 0
        for j in range(k):
            normal_dist_sum += phis[j] / (np.sqrt(2 * np.pi * variances[j])) * np.exp(
                -pow(data[i] - means[j], 2) / (2 * variances[j]))
        for j in range(k):
            normal_dist_j = phis[j] / (np.sqrt(2 * np.pi * variances[j])) * np.exp(
                -pow(data[i] - means[j], 2) / (2 * variances[j]))
            r[i][j] = normal_dist_j / normal_dist_sum
    return r


# m step of the Algorithm, Maximization of the models parameters given data and it's membership probabilities.
def m_step(data, k, r):
    m_c = np.zeros(k)
    means = np.zeros(k)
    var = np.zeros(k)
    phi = np.zeros(k)
    for i in range(len(data)):
        for j in range(k):
            m_c[j] += r[i][j]
    for j in range(k):
        phi[j] = m_c[j] / sum(m_c)
        for i, x in enumerate(data):
            means[j] += r[i][j] * x
        means[j] = means[j] / m_c[j]
        for i, x in enumerate(data):
            var[j] += r[i][j] * pow(x - means[j], 2)
        var[j] = var[j] / m_c[j]
    return means, var, phi


# Parameter Estimation using the EM - Algorithm until it converges or with at most 500 steps
def parameter_estimation(data, k):
    means = np.zeros(k)
    var = np.zeros(k)
    phi = np.zeros(k)
    r = np.zeros((len(data), k))
    for i in range(k):
        means[i] = (random.uniform(-8, 8))
        var[i] = 1
        phi[i] = (1 / k)
    iteration = 0
    while iteration < 500:
        iteration += 1
        r = e_step(data, k, means, var, phi)
        new_means, new_var, new_phi = m_step(data, k, r)
        if np.array_equal(means, new_means) & np.array_equal(var, new_var) & np.array_equal(phi, new_phi):
            return means, var, phi
        means, var, phi = new_means, new_var, new_phi
    return means, var, phi


def calculate_log_likelihood(data, means, var, phis):
    log_likelihood = 0
    for d in data:
        normal_dist_j = 0
        for j in range(len(means)):
            normal_dist_j += phis[j] / (np.sqrt(2 * np.pi * var[j])) * np.exp(
                -pow(d - means[j], 2) / (2 * var[j]))
        log_likelihood = log_likelihood + np.log(normal_dist_j)
    return log_likelihood


# Reading the Data
data = []
with open('Dataset2.csv', 'rt')as f:
    data_txt = csv.reader(f)
    for row in data_txt:
        point = (float(row[0]))
        data.append(point)

#k = 1
#means, var, phi = parameter_estimation(data, k)
#print("For k = " + str(k))
#print("means = " + str(means))
#print("variance = " + str(var))
#print("phi = " + str(phi))


# In the following the estimated parameters of each model are saved to calculate the log likelihood for them.
means1 = [-1.03289019]
variance1 = [17.01069807]
phi1 = [1.]

means2 = [-5.08542394, 1.38271053]
variance2 = [0.94845591, 10.96051591]
phi2 = [0.37346174, 0.62653826]

means3 = [5.12205904, -5.00275328, -0.19701913]
variance3 = [1.22006034, 1.06545761, 2.32286406]
phi3 = [0.22562717, 0.42366047, 0.35071236]

means4 = [-5.01008231, 0.11854755, -0.3261539, 5.22185873]
variance4 = [1.06444403, 4.87522532, 1.32670881, 1.09953406]
phi4 = [0.41725557, 0.2063634,  0.16803818, 0.20834286]

means5 = [-0.55779256, 5.38404017, 3.56953029, -5.00116813, 0.07574854]
variance5 = [2.27086098, 0.94824524, 1.55206124, 1.07004419, 1.30912868]
phi5 = [0.21946836, 0.18018584, 0.0643325, 0.42365069, 0.11236261]

means6 = [0.33607459, -1.0070071, 5.08869831, -3.4262316, -5.69016439, -4.38306218]
variance6 = [1.49728676, 0.09530925, 1.26699145, 1.02187234, 0.49051324, 0.10420304]
phi6 = [0.24925302, 0.05266948, 0.22986737, 0.13430495, 0.23690409, 0.0970011]

means7 = [0.41174193, -4.36988216, 4.23038672, -1.00762562, -3.23910578, 5.79290669, -5.68397121]
variance7 = [1.20838596, 0.11932984, 0.8373796, 0.10675433, 1.02524511, 0.6293783, 0.49262695]
phi7 = [0.22823475, 0.10668984, 0.11213038, 0.06141953, 0.12913618, 0.12241894, 0.23997038]

means8 = [-2.73287528, 0.15321581, 6.29980485, -0.94077722, 4.49803862, 5.71622582,
          -5.12977441, 1.16908436]
variance8 = [1.81299456, 0.17267051, 0.01156247, 0.12769576, 1.21932602, 0.77551002,
             0.94293642, 0.53797029]
phi8 = [0.12103749, 0.06707379, 0.01859245, 0.0901054, 0.14987284, 0.07077591,
        0.37905019, 0.10349193]

means9 = [-4.65048963, -1.03535186, -7.00653335, -2.70411547, 0.36990768, -4.15785368,
          4.93567915, 6.29950501, -5.64969264]
variance9 = [9.91604006e-04, 1.16164840e-01, 2.94666054e-02, 2.77492187e-01,
             1.31716702e+00, 1.74659392e-01, 1.29582052e+00, 1.21660969e-02,
             3.35312714e-01]
phi9 = [0.02807053, 0.06354868, 0.01461812, 0.06182402, 0.23670556, 0.14061354,
        0.21315378, 0.02027655, 0.22118923]

means10 = [-5.72811014, -4.21106207, 6.12846005, 3.55511769, -1.01524184, 4.78291212,
           -4.41932708, -2.63317255, 1.75634531, 0.22048614]
variance10 = [0.46260678, 0.36442604, 0.35036068, 0.05792674, 0.11758633, 0.17743831,
              0.08158962, 0.56597013, 0.94432481, 0.62343707]
phi10 = [0.22957603, 0.10636219, 0.09537896, 0.03239077, 0.07600889, 0.09278107,
         0.06478738, 0.07972351, 0.07217125, 0.15081995]

#print("Log Likelihood for k = 10")
#print(calculate_log_likelihood(data, means10, variance10, phi10))
