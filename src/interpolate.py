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

from numpy import amax, zeros, append, interp, ceil, min, max
from numba import njit


# calculates length of each input set of particles and mediums

def calc_length(particle, medium, plength, mlength, length):
    for j in range(len(plength)):
        plength[j] = length
    for j in range(len(mlength)):
        mlength[j] = length
    for i in range(length-1, 0, -1):
        for j in range(len(plength)):
            if particle[j, i, 0] == 0:
                plength[j] = i
        for j in range(len(mlength)):
            if medium[j, i, 0] == 0:
                mlength[j] = i
    return plength, mlength


# interpolate properties into new mesh
@njit()
def new_properties(particle2, medium2, plength, mlength, particle, medium, mesh):
    for i in range(len(particle2[:, 0, 0])):
        particle2[i, :, 0] = mesh[:]
        for j in range(len(mesh)):
            particle2[i, j, 1] = interp(particle2[i, j, 0], particle[i, :plength[i], 0], particle[i, :plength[i], 1])
            particle2[i, j, 2] = interp(particle2[i, j, 0], particle[i, :plength[i], 0], particle[i, :plength[i], 2])
            if len(particle2[i, j, :]) == 4:
                particle2[i, j, 3] = interp(particle2[i, j, 0], particle[i, :plength[i], 0], particle[i, :plength[i], 3])
    for i in range(len(medium[:, 0, 0])):
        medium2[i, :, 0] = mesh[:]
        for j in range(len(mesh)):
            medium2[i, j, 1] = interp(medium2[i, j, 0], medium[i, :mlength[i], 0], medium[i, :mlength[i], 1])
            medium2[i, j, 2] = interp(medium2[i, j, 0], medium[i, :mlength[i], 0], medium[i, :mlength[i], 2])
            if len(medium[i, j, :]) == 4:
                medium2[i, j, 3] = interp(medium2[i, j, 0], medium[i, :mlength[i], 0], medium[i, :mlength[i], 3])
    return particle2, medium2


def calc_start_end(particle, medium):
    start = max(particle[:, 0, 0])
    end = min(particle[:, -1, 0])
    if max(medium[:, 0, 0]) > start:
        start = max(medium[:, 0, 0])
    if min(medium[:, -1, 0]) < end:
        start = min(medium[:, -1, 0])
    return start, end


def interpolate(particle, medium, length, interval, start, end):
    # if start and end are not specified
    if start == 0 and end == 0:
        start, end = calc_start_end(particle, medium)
    # find length for each particle and medium
    plength = zeros(len(particle[:, 0, 0]), dtype=int)
    mlength = zeros(len(medium[:, 0, 0]), dtype=int)
    plength, mlength = calc_length(particle, medium, plength, mlength, length)
    # find number of wavelengths required
    num_wl = int(ceil(1+((end-start)/interval)))
    # new particle and matrix arrays
    particle2 = zeros((len(particle[:, 0, 0]), num_wl, 4))
    medium2 = zeros((len(medium[:, 0, 0]), num_wl, 4))
    # create mesh
    mesh = zeros(num_wl)
    for i in range(num_wl-1):
        mesh[i] = start + i*interval
    mesh[-1] = end
    # interpolate each particle and matrix
    particle2, medium2 = new_properties(particle2, medium2, plength, mlength, particle, medium, mesh)

    return particle2, medium2, start, end