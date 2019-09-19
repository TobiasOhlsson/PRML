import csv
import numpy as np
import matplotlib.pyplot as plt

# reads and plots the unchanged data
# reads the file different then the read file method for easier access to the data while plotting.
def plot_original_data():
    data = np.zeros((2, 1000))
    with open('Dataset3.csv', 'rt')as f:
        data_txt = csv.reader(f)
        for i, row in enumerate(data_txt):
            data[0][i] = row[0]
            data[1][i] = row[1]

    plt.scatter(data[0], data[1])
    plt.savefig('Plain_Data.png')
    plt.show()


# Reads the the file Dataset3.csv and returns the data as a numpy matrix
def read_file():
    data = np.zeros((1000, 2))
    with open('Dataset3.csv', 'rt')as f:
        data_txt = csv.reader(f)
        for i, row in enumerate(data_txt):
            data[i][0] = row[0]
            data[i][1] = row[1]

        return data


# calls methods needed to solve question 3) i
def question3i():
    data = read_file()
    pca(data)


# centers a given dataset
def center_data(data):
    means = np.mean(data, axis=0)
    for i in range(len(data)):
        data[i][0] = data[i][0] - means[0]
        data[i][1] = data[i][1] - means[1]
    return data


# Prints the eigenvectores of the co-variance matrix for a given dataset and the share of each eigenvector of the total Variance.
def pca(data):
    data = center_data(data)
    c = np.zeros((2, 2))
    # calculation of co-variance matrix
    for d in data:
        a = np.zeros((2, 2))
        a[0][0] = d[0]*d[0]
        a[0][1] = d[1]*d[0]
        a[1][0] = d[0]*d[1]
        a[1][1] = d[1]*d[1]
        c = np.add(a, c)
    c = np.divide(c, len(data))
    eigenvalues, eigenvectors = np.linalg.eig(c)
    perc_of_variance = np.divide(eigenvalues, sum(eigenvalues))
    print("The Eigenvector " + str(eigenvectors[0]) + " contributes " + str(perc_of_variance[0]) + " percent of the total variance")
    print("The Eigenvector " + str(eigenvectors[1]) + " contributes " + str(perc_of_variance[1]) + " percent of the total variance")


# the first kernel defined in exercise 3) ii
def kernel1(x, y, param):
    return pow((1+np.dot(x, y)), param)


# the second kernel defined in exercise 3) ii
def kernel2(x, y, param):
    return np.exp(-np.dot(np.subtract(x, y), np.subtract(x, y))/(2*param*param))


# calls methods needed to solve question 3) ii and saves resulting plot as a png
def question3ii():
    data = read_file()
    kernel_param = 0.5
    transformed_data = kernel_pca(data, kernel2, kernel_param)
    plt.scatter(transformed_data[0], transformed_data[1])
    plt.savefig('Kernel_B_5.png')
    plt.show()


# Applies kernel PCA with the given kernel method on the given data.
def kernel_pca(data, kernel_method, kernel_param):
    k = np.zeros((len(data), len(data)))
    # Calculating Kernel Matrix K
    for i, x in enumerate(data):
        for j, y in enumerate(data):
            k[i][j] = kernel_method(x, y, kernel_param)
    eigenvalues, eigenvectors = np.linalg.eig(k)
    # Pick the two biggest eigenvalues to select which eigenvectores are the principal components
    indices_of_top_eigenvectores = np.argsort(eigenvalues)[-2:]
    top_eigenvectores = np.zeros((2, len(data)))
    top_eigenvectores[0] = eigenvectors[indices_of_top_eigenvectores[0]]
    top_eigenvectores[1] = eigenvectors[indices_of_top_eigenvectores[1]]
    data_pca = np.zeros((2, len(data)))
    # Calculates transformation in the principal components space for each datapoint
    for i in range(len(data)):
        data_pca[0][i] = np.dot(top_eigenvectores[0], k[i])
        data_pca[1][i] = np.dot(top_eigenvectores[1], k[i])
    return data_pca


#question3i()
question3ii()



