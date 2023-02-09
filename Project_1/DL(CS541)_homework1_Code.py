#Problem 1

import numpy as np

#Using Python operator overloading
def problem_1a (A, B):
    return A + B 

#Used array multiplication operator
def problem_1b (A, B, C):
    return A@B - C

#Hadamard Product using *
def problem_1c (A, B, C):
    return A*B + C.T

#Calculating inner product for column vectors
def problem_1d (x, y):
    return np.inner(x.T, y.T)

#Solving the equation without matrix inversion
def problem_1e (A, x):
    return np.linalg.solve(A, x)

#Computing using Lin ALg Solver
def problem_1f (A, x):
    return np.transpose(np.linalg.solve(np.transpose(A), np.transpose(x)))

#Using np.sum instead of loops
def problem_1g (A, i):
    cols = list(np.arange(0, len(A), 2)
    return np.sum(A[i-1][cols])

#Finding valid interval and their mean
def problem_1h (A, c, d):
    above = A[np.nonzero(A >= c)]
    interval = above[np.nonzero(above <= d)]
    return np.mean(interval)

#Matrix of right eigen vectors
def problem_1i(A, k):
    eval, evect = np.linalg.eig(A)
    desc_indi = np.argsort(eval)[::-1]
    indices = desc_indi[:k]
    n_k_matrix = evect[:,indices]
    return n_k_matrix

#Matrix of Multi-dimensional Gaussian distribution
def problem_1j (x, k, m, s):
    mean = x + (m * z)
    musq = s * np.identity(n)
    return (musq + (mean * np.random.randn(n, k)))

#Permuting the rows in matrix
def problem_1k (A):
    ans = np.take(A, np.random.permutation(A.shape[0]), axis=0, out=A);
    return ans

#Z-Scoring
def problem_1l (x):
    mean_A = np.mean(A)
    sigma_A = np.std(A)
    rows, cols = len(A), len(A[0])
    y = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            y[i][j] = ((A[i][j]) - mean_A) / sigma_A
            print(y[i][j])

    return y

#Matrix of k copies of x
def problem_1m (x, k):
    x = x[:, np.newaxis]
    x = np.atleast_2d(x)
    ans = (np.repeat(x, k, axis=1))
    return ans

#Matrix of pairwise distances 
def problem_1n (X):
    row,cols = np.shape(X)
    a = np.transpose([X]*c)
    swap_a = np.swapaxes(a, 0, 2)
    sq_diff = np.square(a-swap_a)
    sum = np.sum(sq_diff, axis = 1)
    l2n = np.sqrt(sum)
    return l2n

#Problem 2
#Train age regressor

def linear_regression(X_tr, y_tr):
    X_tr = X_tr.T
    X_X_T = np.matmul(X_tr, X_tr.T)
    X_y = np.matmul(X_tr, y_tr)
    return np.linalg.solve(X_X_T, X_y)

#Created a method for calculating the mean square error
def mean_square_error(y_hat, y):
        sq_error = np.square(y_hat-y)
        mse = (np.mean(sq_error))/2
        return mse

def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("age_regression_yte.npy")

    w = linear_regression(X_tr, ytr)

    # Report fMSE cost on the training and testing data (separately)

    y_hat_tr = np.matmul(X_tr, w)
    y_hat_te = np.matmul(X_te, w)

    mse_tr = mean_square_error(y_hat_tr,ytr)
    mse_te = mean_square_error(y_hat_te,yte)
    print("The MSE for Training Set and Test Set respectively")
    return mse_tr, mse_te

#Problem 3
from scipy.stats import poisson
import matplotlib.pyplot as plt

data = np.load("PoissonX.npy")

#Plotting the Histogram for given data
def plotempprobdist():
    plt.hist(data, density=True)
    plt.xlabel('x')
    plt.ylabel('P(y)')
    plt.title("Empirical Probability Distribution Histogram of data in Poisson.")
    plt.show()

# plotting the Probability Distributions
def plotPDpoirv():
    x = np.arange(-10, 20, 1)

    y1 = poisson.pmf(x, mu=2.5)
    plt.plot(x, y1)
    plt.xlabel('x')
    plt.ylabel('P(y)')
    plt.title("Probability Distribution for Posisson RV for Mean = 2.5")
    plt.show()
    y2 = poisson.pmf(x, mu=3.1)
    plt.plot(x, y2)
    plt.xlabel('x')
    plt.ylabel('P(y)')
    plt.title("Probability Distribution for Posisson RV for Mean = 3.1")
    plt.show()
    y3 = poisson.pmf(x, mu=3.7)
    plt.plot(x, y3)
    plt.xlabel('x')
    plt.ylabel('P(y)')
    plt.title("Probability Distribution for Posisson RV for Mean = 3.7")
    plt.show()
    y4 = poisson.pmf(x, mu=4.3)
    plt.plot(x, y4)
    plt.xlabel('x')
    plt.ylabel('P(y)')
    plt.title("Probability Distribution for Posisson RV for Mean = 4.3")
    plt.show()
