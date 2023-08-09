#  FOS: FOS, which means "light" in Greek, is used for Fast Optical Spectrum (FOS) calculations of nanoparticle media by combining Mie theory with either Monte Carlo simulations or a pre-trained deep neural network.
#  Copyright (C) 2023 Daniel Carne <dcarne@purdue.edu>
#  Copyright (C) 2023 Joseph Peoples <@gmail.com>
#  Copyright (C) 2023 Ziqi Guo <wu.li.phys2011@gmail.com>
#  Copyright (C) 2023 Dudong Feng <Tianli.Feng2011@gmail.com>
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

from numpy import zeros, loadtxt, vstack, log10
from numba import njit, prange


def setup():
    b0 = zeros(100)
    b1 = zeros(100)
    b2 = zeros(100)
    b3 = zeros(100)
    b4 = zeros(100)
    w0 = zeros((5, 100))
    w1 = zeros((100, 100))
    w2 = zeros((100, 100))
    w3 = zeros((100, 100))
    w4 = zeros((100, 3))



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
                n1[j] = n1[j]*0.01
        # layer 2
        for k in range(len(w1[:, 0])):
            n2[:] += n1[k] * w1[k, :]
        n2[:] += b1[:]
        for j in range(len(n2)):
            if n2[j] < 0:
                n2[j] = n2[j]*0.01
        # layer 3
        for k in range(len(w2[:, 0])):
            n3[:] += n2[k] * w2[k, :]
        n3[:] += b2[:]
        for j in range(len(n3)):
            if n3[j] < 0:
                n3[j] = n3[j]*0.01
        # layer 4
        for k in range(len(w3[:, 0])):
            n4[:] += n3[k] * w3[k, :]
        n4[:] += b3[:]
        for j in range(len(n4)):
            if n4[j] < 0:
                n4[j] = n4[j]*0.01
        # layer 5
        for k in range(len(w4[:, 0])):
            n5[:] += n4[k] * w4[k, :]
        n5[:] += b4[:]

        predict[i, :] = n5[:]
    return predict


@njit()
def remove_negative_values(results):
    # sets any negative values to 0
    for i in range(len(results[:, 0])):
        if results[i, 1] < 0:
            results[i, 1] = 0
        if results[i, 2] < 0:
            results[i, 2] = 0
        if results[i, 3] < 0:
            results[i, 4] = 0
    return results

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
    features[:, 0] = (features[:, 0]) / 7
    features[:, 1] = log10(features[:, 1] + 1) / 6.01
    features[:, 2] = log10(features[:, 2] + 1) / 5.31
    features[:, 3] = -((-features[:, 3]+1)**(1/5)) + 1
    features[:, 4] = log10(features[:, 4] + 1) / 0.0212
    results = zeros((len(features[:, 0]), 4))
    predict = zeros((len(features[:, 0]), 3))
    print("Start NN")
    predict = nn3(b0, b1, b2, b3, b4, w0, w1, w2, w3, w4, features, predict)
    print("End NN")

    # specular refl
    results[:, 0] = ((7 * features[:, 0] - 1) / (7 * features[:, 0] + 1)) ** 2
    # normalize prediction
    results[:, 1] = (predict[:, 0] / (predict[:, 0] + predict[:, 1] + predict[:, 2])) * (1 - results[:, 0])
    results[:, 2] = (predict[:, 1] / (predict[:, 0] + predict[:, 1] + predict[:, 2])) * (1 - results[:, 0])
    results[:, 3] = (predict[:, 2] / (predict[:, 0] + predict[:, 1] + predict[:, 2])) * (1 - results[:, 0])
    results = remove_negative_values(results)
    return results