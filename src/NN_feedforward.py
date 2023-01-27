from numpy import zeros, loadtxt, vstack, log10
from numba import njit, prange


@njit(parallel=True)
def nn3(b0, b1, b2, b3, b4, w0, w1, w2, w3, w4, features, predict):
    for i in prange(len(features[:, 0])):
        n1 = zeros(len(w0[0, :]))
        n2 = zeros(len(w1[0, :]))
        n3 = zeros(len(w2[0, :]))
        n4 = zeros(len(w3[0, :]))
        n5 = zeros(len(w4[0, :]))

        # layer 1
        for k in range(len(w0[:, 0])):
            n1[:] += features[i, k] * w0[k, :]
        n1[:] += b0[:]
        for j in range(len(n1)):
            if n1[j] < 0:
                n1[j] = 0
        # layer 2
        for k in range(len(w1[:, 0])):
            n2[:] += n1[k] * w1[k, :]
        n2[:] += b1[:]
        for j in range(len(n2)):
            if n2[j] < 0:
                n2[j] = 0
        # layer 3

        for k in range(len(w2[:, 0])):
            n3[:] += n2[k] * w2[k, :]
        n3[:] += b2[:]
        for j in range(len(n3)):
            if n3[j] < 0:
                n3[j] = 0
        # layer 4

        for k in range(len(w3[:, 0])):
            n4[:] += n3[k] * w3[k, :]
        n4[:] += b3[:]
        for j in range(len(n4)):
            if n4[j] < 0:
                n4[j] = 0
        # layer 5
        for k in range(len(w4[:, 0])):
            n5[:] += n4[k] * w4[k, :]
        n5[:] += b4[:]
        x = 0
        predict[i, :] = n5[:]
    return predict


def forward(prop):
    # import weights and biases


    w0 = loadtxt('model/w0.txt')
    w1 = loadtxt('model/w1.txt')
    w2 = loadtxt('model/w2.txt')
    w3 = loadtxt('model/w3.txt')
    w4 = loadtxt('model/w4.txt')

    b0 = loadtxt('model/b0.txt')
    b1 = loadtxt('model/b1.txt')
    b2 = loadtxt('model/b2.txt')
    b3 = loadtxt('model/b3.txt')
    b4 = loadtxt('model/b4.txt')

    # prepare and normalize input array
    features = zeros((0, 5))
    for i in range(len(prop[:, 0])):
        if prop[i, 4] != 0:
            features = vstack((features, prop[i, :]))
    # normalize each feature
    features[:, 0] = (features[:, 0]) / 10
    features[:, 1] = log10(features[:, 1] + 1) / 4.6990
    features[:, 2] = log10(features[:, 2] + 1) / 5.1761
    features[:, 4] = features[:, 4] / 0.05
    results = zeros((len(features[:, 0]), 4))
    predict = zeros((len(features[:, 0]), 3))
    print("Start NN")
    predict = nn3(b0, b1, b2, b3, b4, w0, w1, w2, w3, w4, features, predict)
    print("End NN")

    # specular refl
    results[:, 0] = ((10 * features[:, 0] - 1) / (10 * features[:, 0] + 1)) ** 2
    # normalize prediction
    results[:, 1] = (predict[:, 0] / (predict[:, 0] + predict[:, 1] + predict[:, 2])) * (1 - results[:, 0])
    results[:, 2] = (predict[:, 1] / (predict[:, 0] + predict[:, 1] + predict[:, 2])) * (1 - results[:, 0])
    results[:, 3] = (predict[:, 2] / (predict[:, 0] + predict[:, 1] + predict[:, 2])) * (1 - results[:, 0])
    return results