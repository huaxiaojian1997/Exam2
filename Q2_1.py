import numpy as np
import matplotlib.pyplot as plt
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import KFold


def Initial(Ntrain, Ntest):
    # Initial parameter
    n = 2
    alpha = np.array([0.33, 0.34, 0.33])  # must add to 1.0
    meanVectors = np.array([[-18, 0, 18], [-8, 0, 8]])
    covEvalues = np.array([[10.24, 0], [0, 0.36]])
    covEvectors = np.zeros([3, 2, 2])
    covEvectors[0, :, :] = np.array([[1, -1], [1, 1]]) / math.sqrt(2)
    covEvectors[1, :, :] = np.array([[1, 0], [0, 1]])
    covEvectors[2, :, :] = np.array([[1, -1], [1, 1]]) / math.sqrt(2)

    # Generate training set
    t = np.random.rand(1, Ntrain)
    ind1 = np.mat(np.where((0 <= t) & (t <= alpha[0])))
    ind2 = np.mat(np.where((alpha[0] < t) & (t <= alpha[0] + alpha[1])))
    ind3 = np.mat(np.where((alpha[0] + alpha[1] <= t) & (t <= 1)))
    Xtrain = np.zeros([n, Ntrain])
    z1 = np.mat(covEvectors[0, :, :]) * np.mat(covEvalues ** 0.5) * np.mat(
        np.random.randn(n, int(ind1.size / 2))) + np.mat(meanVectors[:, 0]).T
    Xtrain[0, ind1] = z1[0, :]
    Xtrain[1, ind1] = z1[1, :]
    z2 = np.mat(covEvectors[1, :, :]) * np.mat(covEvalues ** 0.5) * np.mat(
        np.random.randn(n, int(ind2.size / 2))) + np.mat(meanVectors[:, 1]).T
    Xtrain[0, ind2] = z2[0, :]
    Xtrain[1, ind2] = z2[1, :]
    z3 = np.mat(covEvectors[2, :, :]) * np.mat(covEvalues ** 0.5) * np.mat(
        np.random.randn(n, int(ind3.size / 2))) + np.mat(meanVectors[:, 2]).T
    Xtrain[0, ind3] = z3[0, :]
    Xtrain[1, ind3] = z3[1, :]

    # Generate test set
    t = np.random.rand(1, Ntest)
    ind1 = np.mat(np.where((0 <= t) & (t <= alpha[0])))
    ind2 = np.mat(np.where((alpha[0] < t) & (t <= alpha[0] + alpha[1])))
    ind3 = np.mat(np.where((alpha[0] + alpha[1] <= t) & (t <= 1)))
    Xtest = np.zeros([n, Ntest])
    z1 = np.mat(covEvectors[0, :, :]) * np.mat(covEvalues ** 0.5) * np.mat(
        np.random.randn(n, int(ind1.size / 2))) + np.mat(meanVectors[:, 0]).T
    Xtest[0, ind1] = z1[0, :]
    Xtest[1, ind1] = z1[1, :]
    z2 = np.mat(covEvectors[1, :, :]) * np.mat(covEvalues ** 0.5) * np.mat(
        np.random.randn(n, int(ind2.size / 2))) + np.mat(meanVectors[:, 1]).T
    Xtest[0, ind2] = z2[0, :]
    Xtest[1, ind2] = z2[1, :]
    z3 = np.mat(covEvectors[2, :, :]) * np.mat(covEvalues ** 0.5) * np.mat(
        np.random.randn(n, int(ind3.size / 2))) + np.mat(meanVectors[:, 2]).T
    Xtest[0, ind3] = z3[0, :]
    Xtest[1, ind3] = z3[1, :]
    return Xtrain, Xtest


def K_Fold(bs, ep):
    # K-Fold
    # Prepare cross validation
    score1 = np.zeros((10, 10))
    score2 = np.zeros((10, 10))
    Xdata = Xtrain[0, :]
    Ydata = Xtrain[1, :]
    kfold = KFold(n_splits=10)

    # sigmoid
    for i in range(1, 11):
        model = Sequential()
        model.add(Dense(i, activation='sigmoid', input_dim=1))
        model.add(Dense(1, activation=None))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        j = 0
        for train_index, test_index in kfold.split(Xdata):
            print('train: %s, test: %s' % (train_index, test_index))
            x_train, x_test = Xdata[train_index], Xdata[test_index]
            y_train, y_test = Ydata[train_index], Ydata[test_index]
            model.fit(x_train, y_train, batch_size=bs, epochs=ep)
            score, acc = model.evaluate(x_test, y_test)
            score1[j, i - 1] = score
            j += 1

    # softplus
    for i in range(1, 11):
        model = Sequential()
        model.add(Dense(i, activation='softplus', input_dim=1))
        model.add(Dense(1, activation=None))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        j = 0
        for train_index, test_index in kfold.split(Xdata):
            print('train: %s, test: %s' % (train_index, test_index))
            x_train, x_test = Xdata[train_index], Xdata[test_index]
            y_train, y_test = Ydata[train_index], Ydata[test_index]
            model.fit(x_train, y_train, batch_size=bs, epochs=ep)
            score, acc = model.evaluate(x_test, y_test)
            score2[j, i - 1] = score
            j += 1

    meansSig = np.mean(score1[:, :], axis=0)
    meansSoft = np.mean(score2[:, :], axis=0)
    return meansSig, meansSoft


def SelectActivation(meansSig, meansSoft):
    # Select activation and perceptrons
    orderSig = (np.argmin(meansSig) + 1)
    orderSoft = (np.argmin(meansSoft) + 1)

    if meansSoft[orderSoft - 1] < meansSig[orderSig - 1]:
        ActivationChoice = 1
        order = orderSoft
    else:
        ActivationChoice = 0
        order = orderSig
    nonlin = ['sigmoid', 'softplus']
    return order, nonlin[ActivationChoice]


def Model(order, AC, bs, ep):
    # Apply trained net to test set
    global score
    model = Sequential()
    model.add(Dense(order, activation=AC, input_dim=1))
    model.add(Dense(1, activation=None))
    model.compile(loss='mean_squared_error', optimizer='adam')
    converged = 0
    tmp = 0
    epsilon = 0.01
    while not converged:
        model.fit(Xtrain[0, :], Xtrain[1, :], batch_size=bs, epochs=ep)
        score = model.evaluate(Xtest[0, :], Xtest[1, :])
        converged = np.abs(score - tmp) < epsilon
        tmp = score

    Ypredict = model.predict(Xtest[0, :])
    return Ypredict, score


def Plot(Xtrain, Xtest, meansSig, meansSoft, Y_pred):
    # Plot training set and test set
    plt.figure(1)
    plt.subplot(121)
    plt.plot(Xtrain[0, :], Xtrain[1, :], '.')
    plt.title('Training set of 1000 samples')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.subplot(122)
    plt.plot(Xtest[0, :], Xtest[1, :], '.')
    plt.title('Test set of 10000 samples')
    plt.xlabel('x1')
    plt.ylabel('x2')

    plt.figure(2)
    plt.plot(np.arange(1, 11), meansSig, 'ro')
    plt.plot(np.arange(1, 11), meansSoft, 'bo')
    plt.title('Using 10-fold cross-validation to select activation function and the number of perceptrons')
    plt.xlabel('Perceptrons')
    plt.ylabel('MSE')
    plt.legend(['Sigmoid', 'Softplus'])

    plt.figure(3)
    plt.subplot(121)
    plt.plot(Xtest[0, :], Xtest[1, :], 'b.')
    plt.title('Test set data')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.subplot(122)
    plt.plot(Xtest[0, :], Y_pred, 'b.')
    plt.title('The output results of neural network after applying trained MLP to test set')
    plt.xlabel('x1')
    plt.ylabel('x2')

    plt.figure(4)
    plt.plot(Xtest[0, :], Xtest[1, :], 'b.')
    plt.plot(Xtest[0, :], Y_pred, 'r.')
    plt.title('Comparison of the test set data and the output results of neural network')
    plt.xlabel('x1')
    plt.ylabel('x2')

    plt.show()


if __name__ == '__main__':
    Ntrain = 1000
    Ntest = 10000
    batch_size = 10
    epochs = 50
    Xtrain, Xtest = Initial(Ntrain, Ntest)
    sigmoid, softplus = K_Fold(batch_size, epochs)
    per_order, AC = SelectActivation(sigmoid, softplus)
    Y, Estimate = Model(per_order, AC, batch_size, epochs)
    print('the selected model is', AC)
    print('the number of perceptrons is', per_order)
    print('Test Performace MSE is', Estimate)
    Plot(Xtrain, Xtest, sigmoid, softplus, Y)
