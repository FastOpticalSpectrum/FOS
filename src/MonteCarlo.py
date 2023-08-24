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

from numpy import zeros, random, pi, cos, log, vstack
from numba import njit, prange, get_num_threads, set_num_threads


# specular reflectance at the top layer
@njit()
def r_specular(layer):
    n_amb = layer[0, 0]
    n_medium = layer[1, 0]
    return (n_amb-n_medium)**2/((n_amb+n_medium)**2)


# checks if photon step size hits a boundary
@njit()
def check_bounds(layer_depths, z, uz, step, current_layer, ua, us):
    hit = False
    step_in_z = abs(step * uz)
    step_remaining = 0
    # check if it hits the upper boundary
    if uz < 0:
        distance_to_bound = (z-layer_depths[current_layer])
        if step_in_z > distance_to_bound:
            hit = True
            step_remaining = (step - (-distance_to_bound/uz)) *(ua + us)
            step = -distance_to_bound/uz

    # otherwise, check if it hits the lower boundary
    else:
        distance_to_bound = (layer_depths[current_layer+1]-z)
        if step_in_z > distance_to_bound:
            hit = True
            step_remaining = (step - (distance_to_bound / uz))*(ua + us)
            step = distance_to_bound / uz
    return hit, step, step_remaining


@njit()
def fresnel_reflectance(n_medium, n_outer, uz):
    if n_medium == n_outer:
        return 0, uz
    elif uz > (1.0 - 1.0e-6):
        probability = ((n_outer-n_medium)/(n_outer+n_medium))**2
        return probability, uz
    elif uz < (1.0e-6):
        return 1, 0
    else:
        temp_1 = (1.0 - uz * uz) ** 0.5
        temp_2 = n_medium * temp_1 / n_outer
        if temp_2 >= 1:
            return 1, 0
        temp_3 = (1-temp_2*temp_2)**0.5
        temp_4 = uz*temp_3 - temp_1*temp_2
        temp_5 = uz*temp_3 + temp_1*temp_2
        temp_6 = temp_1*temp_3 + uz*temp_2
        temp_7 = temp_1*temp_3 - uz*temp_2
        probability = 0.5*temp_7*temp_7*(temp_4*temp_4 + temp_5*temp_5) / (temp_6*temp_6*temp_5*temp_5)
        return probability, temp_3


# checks if photon passes through boundary
@njit()
def hit_bound(current_layer, layer, uz, z, crit_cos, r, t, active, w):
    if uz < 0:
        # check if within critical cosine
        if -uz > crit_cos[0, current_layer]:
            probability, uz_new = fresnel_reflectance(layer[current_layer+1, 0], layer[current_layer, 0], -uz)
            if random.random_sample() > probability:
                uz = -uz_new
                if current_layer == 0:
                    active = False
                    r = w
                else:
                    current_layer -= 1
            else:
                uz = -uz
        else:
            uz = -uz

    else:
        if uz > crit_cos[1, current_layer]:
            probability, uz_new = fresnel_reflectance(layer[current_layer+1, 0], layer[current_layer+2, 0], uz)
            if random.random_sample() > probability:
                uz = uz_new
                if current_layer == (len(layer[:, 0])-3):
                    active = False
                    t = w
                else:
                    current_layer += 1
            else:
                uz = -uz
        else:
            uz = -uz

    return current_layer, uz, r, t, active


@njit()
def new_angle(uz, g):
    if g == 0:
        cos_theta = 2*random.random_sample()-1
    else:
        temp = (1 - g * g) / (1 - g + 2 * g * random.random_sample())
        cos_theta = (1 + g * g - temp * temp) / (2 * g)

    sin_theta = (1.0 - cos_theta * cos_theta) ** 0.5

    psi = 2.0 * pi * random.random_sample()  # spin psi 0-2pi
    cos_psi = cos(psi)

    uz = -sin_theta*cos_psi*((1-uz*uz)**0.5)+uz*cos_theta

    return uz


@njit()
def initialize_photon(layer_depths, crit_cos, layer, rsp):
    # local r, a, t values.
    r = 0
    a = 0
    t = 0
    # the layer the photon is currently in
    current_layer = 0
    z = 0
    uz = 1
    w = 1-rsp
    # is the photon still active in the medium?
    active = True
    step_remaining = 0
    while active is True:
        # absorption coefficient of layer
        ua = layer[current_layer+1, 1]
        # scattering coefficient of layer
        us = layer[current_layer + 1, 2]
        # asymmetry parameter of layer
        g = layer[current_layer + 1, 3]

        # set the step size
        if step_remaining == 0:
            rnd = random.random_sample()
            step = -log(rnd) / (ua + us)
        else:
            step = step_remaining / (ua + us)

        # does the step hit the boundary?
        hit, step, step_remaining = check_bounds(layer_depths, z, uz, step, current_layer, ua, us)

        # move photon
        z += step*uz
        # check if it crosses the boundary
        if hit is True:
            current_layer, uz, r, t, active= hit_bound(current_layer, layer, uz, z, crit_cos, r, t, active, w)
        else:
            # partially absorb photon packet
            change_in_w = w*ua/(ua+us)
            w -= change_in_w
            a += change_in_w

            # find new angle
            uz = new_angle(uz, layer[current_layer+1, 3])

        if w < 0.0001 and active is True:
            if random.random_sample() < 0.1:
                w = w/0.1
            else:
                active = False

    return r, a, t


# parellel loop through each photon
@njit(parallel=True)
def run_mc(layer_depths, crit_cos, layer, n, r, a, t, rsp):
    # RUN THIS LOOP IN PARALLEL
    for i in prange(n):
        r[i], a[i], t[i] = initialize_photon(layer_depths, crit_cos, layer, rsp)
    return r, a, t


@njit()
def setup_mc(layer, layer_depths, crit_cos, num_layers):
    z = 0
    for i in range(num_layers):
        thickness = layer[i+1,4]
        z += thickness
        layer_depths[i+1] = z

        # upper interface critical cosine
        n_up = layer[i, 0]
        n_medium = layer[i+1, 0]
        if n_medium > n_up:
            crit_cos[0, i] = (1.0 - n_up * n_up / (n_medium * n_medium)) ** 0.5
        else:
            crit_cos[0, i] = 0
        # lower interface critical cosine
        n_down = layer[i+2, 0]
        if n_medium > n_down:
            crit_cos[1, i] = (1.0 - n_down * n_down / (n_medium * n_medium)) ** 0.5
        else:
            crit_cos[1, i] = 0
    return layer_depths, crit_cos


@njit()
def monte_carlo(layer, n):
    num_layers = len(layer[:, 0]) - 2
    wth = 0.0001
    # reflection, absorption, and transmission
    r = zeros(n)
    a = zeros(n)
    t = zeros(n)

    layer_depths = zeros(num_layers+1)
    crit_cos = zeros((2, num_layers))

    # setup initial params
    layer_depths, crit_cos = setup_mc(layer, layer_depths, crit_cos, num_layers)

    # specular reflectance
    rsp = r_specular(layer)

    # run Monte Carlo sim

    r, a, t = run_mc(layer_depths, crit_cos, layer, n, r, a, t, rsp)
    # sum up results
    r_tot = 0
    a_tot = 0
    t_tot = 0
    for i in range(n):
        r_tot += r[i]
        a_tot += a[i]
        t_tot += t[i]
    r_tot = r_tot/n
    a_tot = a_tot / n
    t_tot = t_tot / n
    return rsp, r_tot, a_tot, t_tot


# counts number of sims
@njit()
def count_sims(prop):
    count = 0
    for i in range(len(prop[:, 0])):
        if prop[i, 0] == 0:
            count += 1
    return count


def main_mc(prop, photons):
    # array to send monte carlo
    layer = zeros((0, 5))
    results = zeros((0, 4))
    count = 0
    total_sims = count_sims(prop)
    #n = get_num_threads()
    #set_num_threads(n - 1)
    #1print("Running Monte Carlo using", n-1, "/", n, "available threads.")
    for i in range(len(prop[:, 0])):
        if prop[i, 0] == 0:
            count += 1
            print("Simulation number:", count, "/", total_sims)
            # send to monte carlo
            rsp, r, a, t = monte_carlo(layer, photons)
            # record results
            results = vstack((results, [rsp, r, a, t]))
            # resets layer
            layer = zeros((0, 5))
        else:
            layer = vstack((layer, prop[i, :]))

    return results


