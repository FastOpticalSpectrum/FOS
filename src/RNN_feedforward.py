#  FOS: FOS, which means "light" in Greek, is used for Fast Optical Spectrum (FOS) calculations of nanoparticle media by combining Mie theory with either Monte Carlo simulations or a pre-trained deep neural network.
#  Copyright (C) 2023 Daniel Carne <dcarne@purdue.edu>
#  Copyright (C) 2023 Joseph Peoples <peoplesj@purdue.edu>
#  Copyright (C) 2023 Ziqi Guo <gziqi@purdue.edu>
#  Copyright (C) 2023 Dudong Feng <feng376@purdue.edu>
#  Copyright (C) 2023 Zherui Han <zrhan@purdue.edu>
#  Copyright (C) 2023 Xiulin Ruan <ruan@purdue.edu>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

from numpy import zeros, loadtxt, vstack, log10, concatenate, max, float32
from numba import njit, prange


@njit(parallel=True)
def rnn(b1, b2, b3, bo, bh, w1, w2, w3, wo, wh, features, predict):

    for sim in prange(len(features[0, 0, :])):
        n1 = zeros(len(w1[:, 0]))
        n2 = zeros(len(w2[:, 0]))
        n3 = zeros(len(w3[:, 0]))
        n4 = zeros(len(wo[:, 0]))
        input = zeros(20)
        num_layers = int(features[0, 4, sim])
        hidden_state = zeros(16)
        for layer in range(num_layers):
            n1[:] = 0
            n2[:] = 0
            n3[:] = 0
            n4[:] = 0
            input[:4] = features[layer, :4, sim]
            input[4:] = hidden_state[:]
            hidden_state[:] = 0
            # layer 1
            for k in range(len(w1[0, :])):
                n1[:] += input[k] * w1[:, k]
            n1[:] += b1[:]
            for j in range(len(n1)):
                if n1[j] < 0:
                    n1[j] = 0
            # layer 2
            for k in range(len(w2[0, :])):
                n2[:] += n1[k] * w2[:, k]
            n2[:] += b2[:]
            for j in range(len(n2)):
                if n2[j] < 0:
                    n2[j] = 0
            # layer 3
            for k in range(len(w3[0, :])):
                n3[:] += n2[k] * w3[:, k]
            n3[:] += b3[:]
            for j in range(len(n3)):
                if n3[j] < 0:
                    n3[j] = 0
            # output layer
            for k in range(len(wo[0, :])):
                n4[:] += n3[k] * wo[:, k]
            n4[:] += bo[:]
            # hidden state
            for k in range(len(wh[0, :])):
                hidden_state[:] += n3[k] * wh[:, k]
            hidden_state[:] += bh[:]
        predict[sim, :] = n4[:]

    return predict


@njit()
def remove_ones_zeros(results):
    # sets any negative values to 0
    for i in range(len(results[:, 0])):
        if results[i, 0] < 0:
            results[i, 0] = 0
        if results[i, 1] < 0:
            results[i, 1] = 0
        if results[i, 2] < 0:
            results[i, 2] = 0
        if results[i, 0] > 1:
            results[i, 0] = 1
        if results[i, 1] > 1:
            results[i, 1] = 1
        if results[i, 2] > 1:
            results[i, 2] = 1
    return results


def shrink(predict):
    results = zeros((0, 5))
    for i in range(len(predict[:, 0])):
        if predict[i, 0] + predict[i, 1] + predict[i, 2] > 0.1:
            results = vstack((results, [0, predict[i, 0], predict[i, 1], predict[i, 2], 1]))
    return results


def stack_features(prop, sims):
    max_layers = int(max(prop[:, 4]))
    features = zeros((max_layers, 5, sims))
    sim = 0
    for i in range(len(prop[:, 0])):
        if prop[i, 0] == 0:
            num_layers = int(prop[i-2, 4])
            features[:num_layers, :, sim] = prop[(i-1-num_layers):(i-1), :]
            sim += 1
    return features


def forward(prop, sims):
    # import weights and biases
    w1 = loadtxt('model/w1_rnn1.7.txt', dtype=float32)
    w2 = loadtxt('model/w2_rnn1.7.txt', dtype=float32)
    w3 = loadtxt('model/w3_rnn1.7.txt', dtype=float32)
    wo = loadtxt('model/wo_rnn1.7.txt', dtype=float32)
    wh = loadtxt('model/wh_rnn1.7.txt', dtype=float32)

    b1 = loadtxt('model/b1_rnn1.7.txt', dtype=float32)
    b2 = loadtxt('model/b2_rnn1.7.txt', dtype=float32)
    b3 = loadtxt('model/b3_rnn1.7.txt', dtype=float32)
    bo = loadtxt('model/bo_rnn1.7.txt', dtype=float32)
    bh = loadtxt('model/bh_rnn1.7.txt', dtype=float32)

    # non-dimensionalize
    prop2 = prop.copy()
    prop2[:, 1] *= prop[:, 4]
    prop2[:, 2] *= prop[:, 4]
    # get num layers
    rows = 0
    for i in range(len(prop2[:, 0])):
        if prop2[i, 0] == 0:
            num_layers = rows - 2
            prop2[(i-num_layers-1):(i-1), 4] = num_layers
            rows = 0
        else:
            rows += 1

    features = stack_features(prop2, sims)


    # normalize each feature
    features[:, 0, :] = (features[:, 0, :]) / 2.5
    features[:, 1, :] = (log10(features[:, 1, :] + 0.0001)+4)/8.69154
    features[:, 2, :] = (log10(features[:, 2, :] + 0.0001)+4)/8.69549
    features[:, 3, :] = -((-features[:, 3, :]+1)**(1/3)) + 1

    predict = zeros((sims, 3))
    print("Compile and start Recurrent NN")
    predict = rnn(b1, b2, b3, bo, bh, w1, w2, w3, wo, wh, features, predict)
    print("End Recurrent NN")
    # normalize prediction
    predict = remove_ones_zeros(predict)
    results = shrink(predict)
    results[:, 1] = (results[:, 1] / (results[:, 1] + results[:, 2] + results[:, 3]))
    results[:, 2] = (results[:, 2] / (results[:, 1] + results[:, 2] + results[:, 3]))
    results[:, 3] = (results[:, 3] / (results[:, 1] + results[:, 2] + results[:, 3]))
    results[:, 4] = features[0, 4, :]
    return results