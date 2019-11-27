import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import KFold


def Initial():
    # Initialization parameters
    mu = np.array([[2.5, 2.5, 2.5],
                   [-2.5, 2.5, 2.5],
                   [2.5, -2.5, 2.5],
                   [-2.5, -2.5, 2.5]]).T
    sigma = np.array([[[3, .3, 2],
                       [.3, 3, 0],
                       [2, 0, 3]],
                      [[4, 0, 0],
                       [0, 4, -3],
                       [0, -3, 4]],
                      [[3, 0.5, 1],
                       [0.5, 3, 0],
                       [1, 0, 3]],
                      [[1, 0, 0.2],
                       [0, 1, 0.2],
                       [0.2, 0.2, 1]]])
    prior = np.array((0.25, 0.25, 0.25, 0.25))
    thr = np.cumsum(prior)
    thr = np.concatenate(([0], thr), axis=0)
    return mu, sigma, thr, prior


def GenerateSamples(N, mu, sigma, thr, A):
    # Generate Samples of data and plot distribution
    x = np.zeros((N, 3))
    u = np.random.rand(1, N)
    Label = np.zeros(N)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in np.arange(0, 4):
        indices = np.where((u > thr[i]) & (u <= thr[i + 1]))
        Label[indices[1]] = i
        x[indices[1], :] = np.random.multivariate_normal(mu[:, i], sigma[i, :, :], np.size(Label[indices[1]]))
        ax.scatter(x[indices[1], 0], x[indices[1], 1], x[indices[1], 2], marker='.')
    print('the number of label=1 is', np.size(np.where(Label == 0)))
    print('the number of label=2 is', np.size(np.where(Label == 1)))
    print('the number of label=3 is', np.size(np.where(Label == 2)))
    print('the number of label=4 is', np.size(np.where(Label == 3)))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend(['Label 1', 'Label 2', 'Label 3', 'Label 4'])
    plt.title(A)
    plt.show()
    return x, Label


def Decision(N, mu, sigma, prior, x):
    #  MAP-classification rule
    Decision = np.zeros(N)
    dist0 = multivariate_normal(mean=mu[:, 0], cov=sigma[0, :, :])
    dist1 = multivariate_normal(mean=mu[:, 1], cov=sigma[1, :, :])
    dist2 = multivariate_normal(mean=mu[:, 2], cov=sigma[2, :, :])
    dist3 = multivariate_normal(mean=mu[:, 3], cov=sigma[3, :, :])
    eval0 = dist0.pdf(x) * prior[0]
    eval1 = dist1.pdf(x) * prior[1]
    eval2 = dist2.pdf(x) * prior[2]
    eval3 = dist3.pdf(x) * prior[3]
    indices0 = np.where((eval0 > eval1) & (eval0 > eval2) & (eval0 > eval3))
    indices1 = np.where((eval1 > eval0) & (eval1 > eval2) & (eval1 > eval3))
    indices2 = np.where((eval2 > eval0) & (eval2 > eval1) & (eval2 > eval3))
    indices3 = np.where((eval3 > eval0) & (eval3 > eval1) & (eval3 > eval2))
    Decision[indices0[0]] = 0
    Decision[indices1[0]] = 1
    Decision[indices2[0]] = 2
    Decision[indices3[0]] = 3

    return Decision


def Print(N, L, Decision):
    # Print MAP confusion matrix, the number of misclassified samples and the theoretical minimum probability of error
    print('MAP confusion matrix(rows as decision and columns as true labels) is \n',
          confusion_matrix(L, Decision, labels=[0, 1, 2, 3]))

    # Counting the number of misclassified samples and the theoretical minimum probability of error
    correct = np.trace(confusion_matrix(L, Decision, labels=[0, 1, 2, 3]))
    error = N - correct
    P_error = error / N
    P_correct = 1 - P_error
    print('the number of misclassified samples is', error)
    print('the probability of error is', P_error)
    print('the probability of correct is', P_correct)


def PlotMap(x, L, D):
    # Plot samples with MAP classifier
    ind11 = np.where((D == 0) & (L == 0))
    ind21 = np.where((D == 1) & (L == 0))
    ind31 = np.where((D == 2) & (L == 0))
    ind41 = np.where((D == 3) & (L == 0))

    ind12 = np.where((D == 0) & (L == 1))
    ind22 = np.where((D == 1) & (L == 1))
    ind32 = np.where((D == 2) & (L == 1))
    ind42 = np.where((D == 3) & (L == 1))

    ind13 = np.where((D == 0) & (L == 2))
    ind23 = np.where((D == 1) & (L == 2))
    ind33 = np.where((D == 2) & (L == 2))
    ind43 = np.where((D == 3) & (L == 2))

    ind14 = np.where((D == 0) & (L == 3))
    ind24 = np.where((D == 1) & (L == 3))
    ind34 = np.where((D == 2) & (L == 3))
    ind44 = np.where((D == 3) & (L == 3))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[ind11, 0], x[ind11, 1], x[ind11, 2], c='c', marker='o')
    ax.scatter(x[ind22, 0], x[ind22, 1], x[ind22, 2], c='c', marker='^')
    ax.scatter(x[ind33, 0], x[ind33, 1], x[ind33, 2], c='c', marker='*')
    ax.scatter(x[ind44, 0], x[ind44, 1], x[ind44, 2], c='c', marker='s')

    ax.scatter(x[ind21, 0], x[ind21, 1], x[ind21, 2], c='r', marker='o')
    ax.scatter(x[ind31, 0], x[ind31, 1], x[ind31, 2], c='g', marker='o')
    ax.scatter(x[ind41, 0], x[ind41, 1], x[ind41, 2], c='m', marker='o')

    ax.scatter(x[ind12, 0], x[ind12, 1], x[ind12, 2], c='b', marker='^')
    ax.scatter(x[ind32, 0], x[ind32, 1], x[ind32, 2], c='g', marker='^')
    ax.scatter(x[ind42, 0], x[ind42, 1], x[ind42, 2], c='m', marker='^')

    ax.scatter(x[ind13, 0], x[ind13, 1], x[ind13, 2], c='b', marker='*')
    ax.scatter(x[ind23, 0], x[ind23, 1], x[ind23, 2], c='r', marker='*')
    ax.scatter(x[ind43, 0], x[ind43, 1], x[ind43, 2], c='m', marker='*')

    ax.scatter(x[ind14, 0], x[ind14, 1], x[ind14, 2], c='b', marker='s')
    ax.scatter(x[ind24, 0], x[ind24, 1], x[ind24, 2], c='r', marker='s')
    ax.scatter(x[ind34, 0], x[ind34, 1], x[ind34, 2], c='g', marker='s')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend(['Label = 1 with Decision = 1', 'Label = 2 with Decision = 2', 'Label = 3 with Decision = 3',
                'Label = 4 with Decision = 4', 'Label = 1 with Decision = 2', 'Label = 1 with Decision = 3',
                'Label = 1 with Decision = 4', 'Label = 2 with Decision = 1', 'Label = 2 with Decision = 3',
                'Label = 2 with Decision = 4', 'Label = 3 with Decision = 1', 'Label = 3 with Decision = 2',
                'Label = 3 with Decision = 4', 'Label = 4 with Decision = 1', 'Label = 4 with Decision = 2',
                'Label = 4 with Decision = 3'])
    plt.title('The test set of 10,000 samples with MAP classifier')
    plt.show()


def K_Fold(X_train, Label_train, bs, ep):
    # Determine the most appropriate number of perceptions in the hidden layer, using 10-fold cross-validation
    global score
    score1 = np.zeros((10, 10))
    Xdata = X_train
    Ydata = keras.utils.to_categorical(Label_train, num_classes=4)
    kfold = KFold(n_splits=10)

    # sigmoid
    for i in range(1, 11):
        model = Sequential()
        model.add(Dense(i, activation='sigmoid', input_dim=3))
        model.add(Dense(4, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        j = 0
        for train_index, test_index in kfold.split(Xdata):
            print('train: %s, test: %s' % (train_index, test_index))
            x_train, x_test = Xdata[train_index], Xdata[test_index]
            y_train, y_test = Ydata[train_index], Ydata[test_index]
            model.fit(x_train, y_train, batch_size=bs, epochs=ep)
            score, acc = model.evaluate(x_test, y_test)
            score1[j, i - 1] = acc
            j += 1

    meansSig = np.mean(score1[:, :], axis=0)
    orderSig = (np.argmax(meansSig) + 1)
    return meansSig, orderSig, Ydata


def Model(X_test, label_test, x_train, y_train, order, bs, ep):
    # Apply  trained neural network classifiers to the test set
    global score
    Y_test = keras.utils.to_categorical(label_test, num_classes=4)
    model = Sequential()
    model.add(Dense(order, activation='sigmoid', input_dim=3))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    converged = 0
    tmp = 0
    epsilon = 0.01
    while not converged:
        model.fit(x_train, y_train, batch_size=bs, epochs=ep)
        score = model.evaluate(X_test, Y_test)
        converged = np.abs(score - tmp) < epsilon
        tmp = score

    Ypredict = model.predict(X_test)
    decision = np.argmax(Ypredict, axis=1)

    return Ypredict, score, decision


def Plot(meansSig1, meansSig2, meansSig3):
    # Plot the process of using cross-validation to determine the model sequence selection
    plt.figure(2)
    plt.plot(np.arange(1, 11), meansSig1, 'r')
    plt.plot(np.arange(1, 11), meansSig2, 'b')
    plt.plot(np.arange(1, 11), meansSig3, 'g')
    plt.title('Using cross-validation to determine the most appropriate number of perceptrons')
    plt.xlabel('Perceptrons')
    plt.ylabel('Probability of correct decisions')
    plt.legend(['100 samples', '1000 samples', '10000 samples'])

    plt.show()


if __name__ == '__main__':
    batch_size = None
    epochs = 50

    # Initial parameter
    Mu, Sigma, Thr, Prior = Initial()

    # Draw 1000 samples and plot them in a 3-dimensional scatter plot with class labels color coded
    GenerateSamples(1000, Mu, Sigma, Thr, '1000 samples in a 3-dimensional scatter plot with class labels color coded')

    # Test set with 10000 samples
    X_test, Label_test = GenerateSamples(10000, Mu, Sigma, Thr, 'The test set of 10,000 samples')
    Decision_test = Decision(10000, Mu, Sigma, Prior, X_test)
    Print(10000, Label_test, Decision_test)
    PlotMap(X_test, Label_test, Decision_test)

    # Training set with 100 samples
    X_train100, Label_train100 = GenerateSamples(100, Mu, Sigma, Thr, 'The training set with 100 samples')
    MeanSig_train100, OrderSig_train100, Y_train100 = K_Fold(X_train100, Label_train100, batch_size, epochs)
    Y_predict100, Estimate_train100, Decision_train100 = Model(X_test, Label_test, X_train100, Y_train100,
                                                               OrderSig_train100, batch_size, epochs)

    # Training set with 1000 samples
    X_train1000, Label_train1000 = GenerateSamples(1000, Mu, Sigma, Thr, 'The training set with 1000 samples')
    MeanSig_train1000, OrderSig_train1000, Y_train1000 = K_Fold(X_train1000, Label_train1000, batch_size, epochs)
    Y_predict1000, Estimate_train1000, Decision_train1000 = Model(X_test, Label_test, X_train1000, Y_train1000,
                                                                  OrderSig_train1000, batch_size, epochs)

    # Training set with 10000 samples
    X_train10000, Label_train10000 = GenerateSamples(10000, Mu, Sigma, Thr, 'The training set with 10000 samples')
    MeanSig_train10000, OrderSig_train10000, Y_train10000 = K_Fold(X_train10000, Label_train10000, batch_size, epochs)
    Y_predict10000, Estimate_train10000, Decision_train10000 = Model(X_test, Label_test, X_train10000, Y_train10000,
                                                                     OrderSig_train10000, batch_size, epochs)

    Print(10000, Label_test, Decision_train100)
    print('the number of perceptrons for 100 samples is', OrderSig_train100)

    Print(10000, Label_test, Decision_train1000)
    print('the number of perceptrons for 1000 samples is', OrderSig_train1000)

    Print(10000, Label_test, Decision_train10000)
    print('the number of perceptrons for 10000 samples is', OrderSig_train10000)

    Plot(MeanSig_train100, MeanSig_train1000, MeanSig_train10000)
