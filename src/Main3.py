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

from numpy import loadtxt, zeros, append, vstack, asarray, concatenate, dstack, round, max
from MieTheory3 import mie_theory, effective_medium, mie_theory_coreshell
from MonteCarlo import main_mc
from SolarIntegration import solar_spectrum
import os.path
from RNN_feedforward import forward
from interpolate import interpolate
import matplotlib.pyplot as plt


# imports the provided txt file
def fname():
    filecheck = False
    infile = " "
    while filecheck is False:
        file = input('Input file: ')
        if os.path.exists(file):
            infile = loadtxt(file, comments="#", dtype=str, delimiter="/")
            filecheck = True
        else:
            print("File not found. Make sure it is in the same directory as this program.")
    return infile


def check_file(file, check):
    infile = " "
    if os.path.exists(file):
        infile = loadtxt(file, comments="#", dtype=str, delimiter="/")
    else:
        print("File not found. Make sure it is in the same directory as this program.")
        check = False
    return infile, check


# makes sure a file exists within the directory
def check_file_exists(name, check):
    if not os.path.exists(name):
        check = True
        print("File " + name + " does not exist in this directory.")
        print("Please re-enter input file once corrected.")
        print("\n")
    return check


# returns a number before a colon, to allow for multi-digit numbers
def get_number(string):
    number = 0
    for i in range(len(string)):
        if string[i] == ':':
            number = string[:i]
            break
    return number, len(number)


# checks to makes sure all material files are within the wavelength range given
def check_material_wavelength_range(particle, medium, check, start, end):
    if start == 0 and end == 0:
        return check
    # initialize the min and max values
    min_wave = particle[0, 0, 0]
    max_wave = max(particle[0, :, 0])
    for i in range(1, len(particle[:, 0, 0])):
        if particle[i, 0, 0] > min_wave:
            min_wave = particle[i, 0, 0]
        if max(particle[i, :, 0]) < max_wave:
            max_wave = max(particle[i, :, 0])

    for i in range(len(medium[:, 0, 0])):
        if medium[i, 0, 0] > min_wave:
            min_wave = medium[i, 0, 0]
        if max(medium[i, :, 0]) < max_wave:
            max_wave = max(medium[i, :, 0])
    if start < min_wave or end > max_wave:
        check = True
        print('Wavelength range is not covered by all input materials')
        print("Please re-enter input file once corrected.")
        print("\n")
    return check


# makes sure the particle or matrix number exists in the header
def check_num_exists(num, p_num, check, type):
    local_check = 1
    for i in range(len(p_num)):
        if num == int(p_num[i]):
            local_check = 0
    if local_check == 1:
        print(type + ' ' + str(num) + ' ' + 'does not exist in the header.')
        print("Please re-enter input file once corrected.")
        print("\n")
        check = True
    return check

# imports information from header of input file and loads in necessary files
def import_header(infile, check):
    # p is list of particle input file names, m for matrix input files.
    # each of these files consists of three columns, wavelength, refractive index, and extinction coefficient
    p = zeros(0, dtype=str)
    m = zeros(0, dtype=str)
    # contains the number tied to each particle and matrix
    p_num = zeros(0, dtype=str)
    m_num = zeros(0, dtype=str)
    # solar spectrum file to import. If there is not one, this variable remains blank
    solar = ""
    # output file name
    output_name = ""
    # mesh percentage, interval spacing between wavelengths.
    mesh_percentage = 0

    #initialize for later
    start = end = 0
    photons = 0
    line = 0

    # loops through header. Breaks loop when it hits the first "sim"
    for i in range(len(infile)):
        if infile[i][0:8] == "particle":
            num, length = get_number(infile[i][8:])
            p_num = append(p_num, num)
            p = append(p, infile[i][(9+length):])
        if infile[i][0:6] == "matrix":
            num, length = get_number(infile[i][6:])
            m_num = append(m_num, num)
            m = append(m, infile[i][7+length:])
        if infile[i][0:6] == "output":
            output_name = infile[i][7:]
        if infile[i][0:5] == "solar":
            solar = infile[i][6:]
        if infile[i][0:8] == "interval":
            mesh_percentage = float(infile[i][9:])
        if infile[i][0:7] == "photons":
            photons = int(infile[i][8:])
        if infile[i][0:5] == "start":
            start = float(infile[i][6:])
        if infile[i][0:3] == "end":
            end = float(infile[i][4:])
        if infile[i][0:3] == "sim":
            line = i-1
            break

    # counts number of simulations and makes sure each particle and matrix exists
    sims = 0
    for i in range(line, len(infile)):
        if infile[i][0:3] == "sim":
            sims += 1
        if infile[i][0:8] == "particle":
            particle_num = int(infile[i][8:])
            check = check_num_exists(particle_num, p_num, check, 'Particle')
        if infile[i][0:6] == "matrix":
            matrix_num = int(infile[i][6:])
            check = check_num_exists(matrix_num, m_num, check, 'Matrix')

    # length is the length (number of wavelength datapoints) of the longest input file
    length = 0
    for i in range(len(p)):
        check = check_file_exists(p[i], check)
        if check == False:
            temp = loadtxt(p[i])
            if len(temp) > length:
                length = len(temp)
    for i in range(len(m)):
        check = check_file_exists(m[i], check)
        if check == False:
            temp = loadtxt(m[i])
            if len(temp) > length:
                length = len(temp)

    # arrays for material properties [number of materials, wavelengths, properties]
    particle = zeros((len(p), length, 4))
    medium = zeros((len(m), length, 4))
    # particle_type is 0 if n and k are provided, 1 if scattering properties are provided
    particle_type = zeros(len(p))
    # need_interp is True if wavelengths of input files don't match, or the mesh_percentage is not 1
    need_interp = False


    if check == False:
        # interp if mesh percentage =/= 1
        if mesh_percentage != 1:
            need_interp = True
        # imports particle files and checks if they need to be interpolated
        for i in range(len(p)):
            temp = loadtxt(p[i])
            # set particle type
            if len(temp[0, :]) == 4:
                particle_type[i] = 1
            particle[i, :len(temp), :len(temp[0, :])] = temp
            if len(temp) != length:
                need_interp = True
            for j in range(length):
                if particle[i, j, 0] < (particle[0, j, 0] - 0.001) or particle[i, j, 0] > (particle[0, j, 0] + 0.001):
                    need_interp = True

        # imports medium files and checks if they need to be interpolated
        for i in range(len(m)):
            temp = loadtxt(m[i])
            medium[i, :len(temp), :len(temp[0, :])] = temp
            if len(temp) != length:
                need_interp = True
            # check wavelengths match
            for j in range(length):
                if medium[i, j, 0] < (particle[0, j, 0] - 0.001) or medium[i, j, 0] > (particle[0, j, 0] + 0.001):
                    need_interp = True

        check = check_material_wavelength_range(particle, medium, check, start, end)
        if check == False:
            # send to interpolation method if the wavelengths don't match or the mesh_percentage is not 1
            if need_interp is True:
                print('Interpolating properties')
                particle, medium, start, end = interpolate(particle, medium, length, mesh_percentage, start, end)

    return particle, medium, output_name, solar, sims, photons, line, check, start, end, p_num, m_num, particle_type


# checks to make sure the number of diameters matches the number of volume fraction inputs
def check_diameters(current_sim, fv, sizes, check, ptype_in, dist):
    # check to make sure same number of diameters and volume fraction
    if ptype_in == 0:
        if len(fv) != len(sizes):
            print('Number of diameters does not match number of volume fractions provided in sim:', current_sim)
            print("Please re-enter input file once corrected.")
            check = True
        if dist[0] != 0 or len(dist) > 1:
            if len(fv) != len(dist):
                print('Number of diameters does not match number of standard deviations provided in sim:', current_sim)
                print("Please re-enter input file once corrected.")
                check = True
    return check


# checks to make sure dist = 0 for core shell
def check_dist(current_sim, dist, check):
    # check to make sure same number of diameters and volume fraction
    if all(dist) != 0:
        print('Std must be 0 for core shell sim', current_sim)
        print("Please re-enter input file once corrected.")
        check = True
    return check


# gets index referring to each material number (index 1 may refer to particle 2 if they are not sequentially numbered)
def get_index(number, type, p_num, m_num):
    if type == 'p':
        for i in range(len(p_num)):
            if int(p_num[i]) == number:
                index = i
                break
    elif type == 'm':
        for i in range(len(m_num)):
            if int(m_num[i]) == number:
                index = i
                break
    return index


def create_prop_array(prop, particle, upper, lower, upper_type, lower_type, layers, optical_per_layer, medium):

    if upper == -1 and lower == -1:
        for j in range(len(particle[0, :, 0])):
            prop = vstack((prop, [1, 0, 0, 0, 0]))
            for k in range(layers):
                prop = vstack((prop, optical_per_layer[:, j, k]))
            prop = vstack((prop, [1, 0, 0, 0, 0]))
            prop = vstack((prop, [0, 0, 0, 0, 0]))
    elif upper != -1 and lower == -1:
        for j in range(len(particle[0, :, 0])):
            prop = vstack((prop, [medium[int(upper_type), j, 1], medium[int(upper_type), j, 2], 0, 0, 0]))
            for k in range(layers):
                prop = vstack((prop, optical_per_layer[:, j, k]))
            prop = vstack((prop, [1, 0, 0, 0, 0]))
            prop = vstack((prop, [0, 0, 0, 0, 0]))
    elif upper == -1 and lower != -1:
        for j in range(len(particle[0, :, 0])):
            prop = vstack((prop, [1, 0, 0, 0, 0]))
            for k in range(layers):
                prop = vstack((prop, optical_per_layer[:, j, k]))
            prop = vstack((prop, [medium[int(lower_type), j, 1], medium[int(lower_type), j, 2], 0, 0, 0]))
            prop = vstack((prop, [0, 0, 0, 0, 0]))
    elif upper != -1 and lower != -1:
        for j in range(len(particle[0, :, 0])):
            prop = vstack((prop, [medium[int(upper_type), j, 1], medium[int(upper_type), j, 2], 0, 0, 0]))
            for k in range(layers):
                prop = vstack((prop, optical_per_layer[:, j, k]))
            prop = vstack((prop, [medium[int(lower_type), j, 1], medium[int(lower_type), j, 2], 0, 0, 0]))
            prop = vstack((prop, [0, 0, 0, 0, 0]))
    return prop


# finds info from input and sends to Mie Theory to calculate optical properties
def optical(line, infile, particle, medium, check, p_num, m_num, particle_type):
    print("Running Mie theory")
    # array of optical properties to send to Monte Carlo
    prop = zeros((0, 5))
    optics_sum = zeros((5, len(particle[0, :, 0])))
    optical_per_layer = zeros((5, len(particle[0, :, 0]), 0))
    vol_frac_sum = 0
    layers = 1
    count = 0
    sizes = 0
    fv = None
    # dist defaults to 0
    dist = zeros(1)
    # if coreshell is true then use that version of Mie theory, else use standard Mie theory
    coreshell = False
    # upper and lower default to negative number
    upper = -1
    lower = -1
    upper_type = 0
    lower_type = 0
    # if wait is true, the shell particle must be read in as well
    wait = False

    current_sim = 1
    for i in range(line+1, len(infile)):
        if infile[i][0:8] == "particle":
            if count > 0:
                if coreshell is False:
                    ptype_in = get_index(ptype, 'p', p_num, m_num)
                    check = check_diameters(current_sim, fv, sizes, check, particle_type[ptype_in], dist)
                if coreshell is True:
                    if wait is False:
                        check = check_dist(current_sim, dist, check)
                        # get the index for each material number (index 1 may refer to particle 2 if they are not sequentially numbered)
                        ptypeCore_in = get_index(ptypeCore, 'p', p_num, m_num)
                        ptype_in = get_index(ptype, 'p', p_num, m_num)
                        mtype_in = get_index(mtype, 'm', p_num, m_num)
                        optics = mie_theory_coreshell(core, shell, fv, particle[int(ptypeCore_in), :, :], particle[int(ptype_in), :, :], medium[int(mtype_in), :, :], thickness, dist)
                        optics_sum += optics
                else:
                    # get the index for each material number (index 1 may refer to particle 2 if they are not sequentially numbered)
                    ptype_in = get_index(ptype, 'p', p_num, m_num)
                    mtype_in = get_index(mtype, 'm', p_num, m_num)
                    optics = mie_theory(sizes, fv, particle[int(ptype_in), :, :], medium[int(mtype_in), :, :], thickness, dist, particle_type[int(ptype_in)])
                    optics_sum += optics
                # dist defaults to 0
                dist = zeros(1)
            count += 1
            # holds previous ptype for core shell
            if count > 1:
                ptypeCore = ptype
            ptype = int(infile[i][8:])
        if infile[i][0:5] == "upper":
            upper = int(infile[i][12:])
            upper_type = get_index(upper, 'm', p_num, m_num)
        elif infile[i][0:5] == "lower":
            lower = int(infile[i][12:])
            lower_type = get_index(lower, 'm', p_num, m_num)
        elif infile[i][0:6] == "matrix":
            mtype = int(infile[i][6:])
        elif infile[i][0:2] == "t:" or infile[i][0:2] == "t=":
            thickness = float(infile[i][2:])
            thickness = thickness / 10000
        elif infile[i][0:2] == "d:" or infile[i][0:2] == "d=":
            sizes = (infile[i][2:]).split(",")
            sizes = asarray(sizes, dtype=float)
            sizes = sizes/2
            coreshell = False
        elif infile[i][0:3] == "vf:" or infile[i][0:3] == "vf=":
            fv = (infile[i][3:]).split(",")
            fv = asarray(fv, dtype=float)
            fv = fv/100
            vol_frac_sum += sum(fv)
        elif infile[i][0:4] == "std:" or infile[i][0:4] == "std=":
            dist = (infile[i][4:]).split(",")
            dist = asarray(dist, dtype=float)
        elif infile[i][0:2] == "c:" or infile[i][0:2] == "c=":
            core = (infile[i][2:]).split(",")
            core = asarray(core, dtype=float)
            core = core/2
            coreshell = True
            wait = True
        elif infile[i][0:2] == "s:" or infile[i][0:2] == "s=":
            shell = (infile[i][2:]).split(",")
            shell = asarray(shell, dtype=float)
            wait = False
        elif infile[i][0:5] == "layer" and infile[i][5:] != "1":
            # check to make sure same number of diameters as volume fractions
            if coreshell is False:
                ptype_in = get_index(ptype, 'p', p_num, m_num)
                check = check_diameters(current_sim, fv, sizes, check, particle_type[ptype_in], dist)
            layers += 1
            if coreshell is True:
                check = check_dist(current_sim, dist, check)
                # get the index for each material number (index 1 may refer to particle 2 if they are not sequentially numbered)
                ptypeCore_in = get_index(ptypeCore, 'p', p_num, m_num)
                ptype_in = get_index(ptype, 'p', p_num, m_num)
                mtype_in = get_index(mtype, 'm', p_num, m_num)
                optics = mie_theory_coreshell(core, shell, fv, particle[int(ptypeCore_in), :, :], particle[int(ptype_in), :, :], medium[int(mtype_in), :, :], thickness, dist)
            else:
                # get the index for each material number (index 1 may refer to particle 2 if they are not sequentially numbered)
                ptype_in = get_index(ptype, 'p', p_num, m_num)
                mtype_in = get_index(mtype, 'm', p_num, m_num)
                optics = mie_theory(sizes, fv, particle[int(ptype_in), :, :], medium[int(mtype_in), :, :], thickness, dist, particle_type[int(ptype_in)])
            optics_sum += optics
            optics_sum[0, :] = optics[0, :]
            optics_sum[4, :] = optics[4, :]
            # get the index for each material number (index 1 may refer to particle 2 if they are not sequentially numbered)
            optics_sum = effective_medium(optics_sum, vol_frac_sum, medium[int(mtype_in), :, :], particle_type[int(ptype_in)])
            optical_per_layer = dstack((optical_per_layer, optics_sum))
            vol_frac_sum = 0
            optics_sum = zeros((5, len(particle[0, :, 0])))
            count = 0
            # reset default dist to 0
            dist = zeros(1)
        elif infile[i][0:3] == "sim" and infile[i][3:] != "1":
            current_sim = int(infile[i][3:])
            # check to make sure same number of diameters as volume fractions
            if coreshell is False:
                ptype_in = get_index(ptype, 'p', p_num, m_num)
                check = check_diameters(current_sim, fv, sizes, check, particle_type[ptype_in], dist)
            if coreshell is True:
                check = check_dist(current_sim, dist, check)
                # get the index for each material number (index 1 may refer to particle 2 if they are not sequentially numbered)
                ptypeCore_in = get_index(ptypeCore, 'p', p_num, m_num)
                ptype_in = get_index(ptype, 'p', p_num, m_num)
                mtype_in = get_index(mtype, 'm', p_num, m_num)
                optics = mie_theory_coreshell(core, shell, fv, particle[int(ptypeCore_in), :, :], particle[int(ptype_in), :, :], medium[int(mtype_in), :, :], thickness, dist)
            else:
                # get the index for each material number (index 1 may refer to particle 2 if they are not sequentially numbered)
                ptype_in = get_index(ptype, 'p', p_num, m_num)
                mtype_in = get_index(mtype, 'm', p_num, m_num)
                optics = mie_theory(sizes, fv, particle[int(ptype_in), :, :], medium[int(mtype_in), :, :], thickness, dist, particle_type[int(ptype_in)])
            optics_sum += optics
            optics_sum[0, :] = optics[0, :]
            optics_sum[4, :] = optics[4, :]
            optics_sum = effective_medium(optics_sum, vol_frac_sum, medium[int(mtype_in), :, :], particle_type[int(ptype_in)])
            optical_per_layer = dstack((optical_per_layer, optics_sum))
            prop = create_prop_array(prop, particle, upper, lower, upper_type, lower_type, layers, optical_per_layer, medium)
            vol_frac_sum = 0
            optics_sum = zeros((5, len(particle[0, :, 0])))
            optical_per_layer = zeros((5, len(particle[0, :, 0]), 0))
            layers = 1
            count = 0
            # reset default dist to 0
            dist = zeros(1)
            # reset upper and lower default to negative number
            upper = -1
            lower = -1
    # check to make sure same number of diameters as volume fractions
    if coreshell is False:
        ptype_in = get_index(ptype, 'p', p_num, m_num)
        check = check_diameters(current_sim, fv, sizes, check, particle_type[ptype_in], dist)
    if coreshell is True:
        check = check_dist(current_sim, dist, check)
        # get the index for each material number (index 1 may refer to particle 2 if they are not sequentially numbered)
        ptypeCore_in = get_index(ptypeCore, 'p', p_num, m_num)
        ptype_in = get_index(ptype, 'p', p_num, m_num)
        mtype_in = get_index(mtype, 'm', p_num, m_num)
        optics = mie_theory_coreshell(core, shell, fv, particle[int(ptypeCore_in), :, :], particle[int(ptype_in), :, :], medium[int(mtype_in), :, :], thickness, dist)
    else:
        # get the index for each material number (index 1 may refer to particle 2 if they are not sequentially numbered)
        ptype_in = get_index(ptype, 'p', p_num, m_num)
        mtype_in = get_index(mtype, 'm', p_num, m_num)
        optics = mie_theory(sizes, fv, particle[int(ptype_in), :, :], medium[int(mtype_in), :, :], thickness, dist, particle_type[int(ptype_in)])
    optics_sum += optics
    optics_sum[0, :] = optics[0, :]
    optics_sum[4, :] = optics[4, :]
    optics_sum = effective_medium(optics_sum, vol_frac_sum, medium[int(mtype_in), :, :], particle_type[int(ptype_in)])
    optical_per_layer = dstack((optical_per_layer, optics_sum))
    prop = create_prop_array(prop, particle, upper, lower, upper_type, lower_type, layers, optical_per_layer, medium)
    return prop, check


# breaks down import file, runs Mie Theory calculations
def nanoparticle(infile, check, line, particle, medium, p_num, m_num, particle_type):
    # gets optical properties of each simulation
    prop, check = optical(line, infile, particle, medium, check, p_num, m_num, particle_type)
    sims_per_medium = len(particle[0, :, 0])
    wavelengths = particle[0, :, 0]

    return prop, sims_per_medium, wavelengths, check


# Makes sure the input data is within the range the NN is trained on
def check_NN_range(prop, check):
    # check for model folder
    model_folder_exists = os.path.exists('model/bh_rnn1.7.txt')
    if model_folder_exists == False:
        print("Cannot fine 'model' folder for neural network weights, make sure model folder is in the same directory as FOS.")
        check = True
    # If check is True, NN does not run and the user must change the input file
    for i in range(len(prop[:, 0])):
        # checks refractive indices
        if prop[i, 0] != 0:
            if prop[i, 4] == 0 and prop[i, 0] != 1:
                print("Neural network is only trained on boundary indices of 1")
                check = True
            if prop[i, 0] < 1 or prop[i, 0] > 2.5:
                print("A refractive index of " + str(prop[i, 0]) + " is out of bounds [1, 2.5]")
                check = True
        # Checks the non-dimensional absorption coefficient, non-dimensional scattering coefficient, asymmetry parameter
        if prop[i, 4] != 0:
            # absorption coefficient
            if (prop[i, 1]*prop[i,4]) < 0 or (prop[i, 1]*prop[i,4]) > 50000:
                print("An absorption coefficient * thickness of " + str((prop[i, 1]*prop[i,4])) + " is out of bounds [0, 50,000]")
                check = True
            # scattering coefficient
            if (prop[i, 2]*prop[i,4]) < 0 or (prop[i, 2]*prop[i,4]) > 50000:
                print("A scattering coefficient * thickness of " + str((prop[i, 2]*prop[i,4])) + " is out of bounds [0, 50000]")
                check = True
            # asymmetry parameter
            if prop[i, 3] < 0 or prop[i, 3] > 1:
                print("An asymmetry parameter of " + str(prop[i, 3]) + " is out of bounds [0, 1]")
                check = True

        if check is True:
            print("Please re-enter input file once corrected, or change to a Monte-Carlo simulation")
            break
    return check


# checks to make sure each required item per simulation is there
def check_for_word_in_sim(infile, word, statement, check):
    # first, find first line of body to start with
    start_line = 0
    for i in range(len(infile)):
        if infile[i][:3] == "sim":
            start_line = i
            break

    word_present = 1
    sim_number = str(1)
    word_size = len(word)
    for i in range(start_line, len(infile)):
        if (infile[i][:3] == "sim" and infile[i][3:] != '1'):

            if word_present == 1:
                check = True
                print("Sim" + sim_number + " must have " + statement)
            sim_number = infile[i][4:]
            word_present = 1
        if infile[i][:word_size] == word:
            word_present = 0
    if word_present == 1:
        check = True
        if statement != '':
            print("Sim" + sim_number + " must have " + statement)
    return check


# check the input file for errors
def check_input_for_errors(infile):
    # check = True is there is an issue caught with the input file, otherwise false
    check = False

    # remove all spaces and tabs, and make lowercase to clean things up
    for i in range(len(infile)):
        infile[i] = infile[i].replace(' ', '')
        infile[i] = infile[i].replace('\t', '')
        infile[i] = infile[i].lower()

    # first line must specify "nn" or "mc"
    if infile[0] != "mc" and infile[0] != "nn":
        print("Either MC or NN must be specified in the first line of the input file.")
        check = True

    ### check for each item in header
    # initially these variables are set to 0, if they are changed to 1 then it is good. If it remains 0 then there is an error
    output = particle = medium = photon = interval = 0
    if infile[0] == "nn":
        photon = 1

    for i in range(len(infile)):
        # first it checks the required items, output name, particle, medium, and number of photons if running MC
        # output name
        if infile[i][:6] == 'output' and infile[i][7:] != '':
            output = 1
        # at least one particle
        if infile[i][:8] == 'particle' and infile[i][10:] != '':
            particle = 1
        # at least one medium
        if infile[i][:6] == 'matrix' and infile[i][8:] != '':
            medium = 1
        # number of photons if using Monte Carlo
        if infile[0] == "mc":
            if infile[i][:7] == "photons" and infile[i][9:] != '':
                photon = 1
        if infile[i][:8] == "interval":
            interval = 1
        # break loop once the first sim starts, this only checks the header
        if infile[i][:3] == "sim":
            break

    # print out error for each bug found in header
    if output == 0:
        check = True
        print("No output file name specified")
    if particle == 0:
        check = True
        print("No particle input specified")
    if medium == 0:
        check = True
        print("No matrix input specified")
    if photon == 0:
        check = True
        print("Number of photons must be specified")
    if interval == 0:
        check = True
        print("No interval specified")


    ### check for errors in the body of the file
    # make sure each sim is numbered correctly
    sim_number_should_be = 1
    # if sim_number_error is flipped to 0, there is an error
    sim_number_error = 1
    for i in range(len(infile)):
        if infile[i][:3] == "sim":
            if infile[i][3:] != str(sim_number_should_be):
                sim_number_error = 0

            sim_number_should_be += 1
    if sim_number_error == 0:
        check = True
        print("Error in simulation numbers. Make sure they are labeled Sim 1, Sim 2, etc.")

    # check for at least one layer
    check = check_for_word_in_sim(infile, 'layer', 'at least one defined layer', check)
    # check for at least one medium
    check = check_for_word_in_sim(infile, 'matrix', 'at least one matrix', check)
    # check for medium thickness
    check = check_for_word_in_sim(infile, 't:', 'a defined thickness', check)
    # check for at least one particle
    check = check_for_word_in_sim(infile, 'particle', 'at least one particle', check)
    # check for at least one volume fraction
    check = check_for_word_in_sim(infile, 'vf', 'at least volume fraction', check)



    if check:
        print("Please re-enter input file once corrected.")
        print("\n")
    return check, infile


def integrate_solar(solar, sims, wavelengths, start, end, results, sims_per_medium):
    if solar != "":
        # solar r, a, and t value for each simulation
        solar_r = zeros(sims)
        solar_a = zeros(sims)
        solar_t = zeros(sims)
        # reflectance at each wavelength to be integrated with the solar spectrum
        refl = zeros((len(wavelengths), 2))
        tra = zeros((len(wavelengths), 2))
        # move wavelengths over to refl
        refl[:, 0] = wavelengths
        tra[:, 0] = wavelengths
        solar_file = loadtxt(solar)
        if start > solar_file[0, 0] or end < solar_file[-1, 0]:
            print("\n")
            print("WARNING: Wavelength range does not cover the full spectrum for the provided solar file.")
            print("The integration only covers the wavelength range provided and not the entire solar file range\n")
        # loop through each simulation to calculate solar reflectance
        for i in range(sims):
            # add the specular and diffuse reflectance together for total reflectance
            refl[:, 1] = (results[i*sims_per_medium:(i+1)*sims_per_medium, 0]+results[i*sims_per_medium:(i+1)*sims_per_medium, 1])
            tra[:, 1] = (results[i * sims_per_medium:(i + 1) * sims_per_medium, 3])
            # send to function to integrate
            solar_r[i], solar_a[i], solar_t[i] = solar_spectrum(solar_file, refl, tra)

    else:
        solar = solar_r = solar_a = solar_t = ""
    return solar_r, solar_a, solar_t, solar


def write_output_script(results, sims, output_name, solar, solar_r, solar_a, solar_t, wavelengths, prop, infile):
    # save output scripts
    length = int(len(results[:, 0]) / sims)
    prop_line = 0
    for i in range(sims):
        output_sim = str(output_name) + str(i + 1) + ".txt"
        with open(output_sim, 'w') as f:
            f.write('Sim ' + str(i + 1) + '\n')
            f.write('\n')
            if solar != "":
                f.write('Solar R: ' + str(abs(round(solar_r[i], 4))) + '\n')
                f.write('Solar A: ' + str(abs(round(solar_a[i], 4))) + '\n')
                f.write('Solar T: ' + str(abs(round(solar_t[i], 4))) + '\n')
                f.write('\n')
            f.write('WL\tR\tA\tT')
            num_layers = int(results[i * length, 4])
            for layer in range(num_layers):
                f.write('\tn(' + str(layer + 1) + ')' + '\tmu_a(' + str(layer + 1) + ')' + '\tmu_s(' + str(
                    layer + 1) + ')' + '\tg(' + str(layer + 1) + ')' + '\tt(' + str(layer + 1) + ')')
            f.write('\n')
            for j in range(length):
                f.write(str(round(wavelengths[j], 4)) + '\t')
                f.write(str(round(results[j + length * i, 0] + results[j + length * i, 1], 4)) + '\t')
                f.write(str(round(results[j + length * i, 2], 4)) + '\t')
                f.write(str(round(results[j + length * i, 3], 4)) + '\t')
                # save properties
                while prop[prop_line, 0] != 0:
                    prop_line += 1
                for layer in range(num_layers):
                    f.write(str(round(prop[prop_line - (1 + num_layers) + layer, 0], 4)) + '\t')
                    f.write(str(round(prop[prop_line - (1 + num_layers) + layer, 1] / 10000, 4)) + '\t')
                    f.write(str(round(prop[prop_line - (1 + num_layers) + layer, 2] / 10000, 4)) + '\t')
                    f.write(str(round(prop[prop_line - (1 + num_layers) + layer, 3], 4)) + '\t')
                    f.write(str(round(prop[prop_line - (1 + num_layers) + layer, 4] * 10000, 4)) + '\t')
                prop_line += 1
                f.write('\n')
            f.write('\n')
            f.write('Input file:\n')
            for j in range(len(infile)):
                if infile[j][:3] == "sim":
                    break
                f.write(infile[j] + '\n')
            for j in range(len(infile)):
                if infile[j][:] == "sim" + str(i + 1):
                    f.write('\n')
                    f.write("sim" + str(i + 1) + '\n')
                    for k in range(j + 1, len(infile)):
                        if infile[k][:3] == "sim":
                            break
                        f.write(infile[k] + '\n')
                    break

        f.close()
    return length


def plot_results(sims, results, length, wavelengths, solar, start, end, output_name):
    # plot configuration
    plt.style.use('default')
    figsize = (10, 8)
    dpi = 600
    axis_font_size = 15  # font_size = 6
    label_font_size = 17  # font_size = 7
    linewidth = 1.5

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.size'] = axis_font_size
    plt.rcParams['font.sans-serif'] = ['Arial']

    # plot the spectral response
    for i in range(sims):
        # get the result of this run
        R_results = results[i * length:(i + 1) * length, 0] + results[i * length:(i + 1) * length, 1]
        A_results = results[i * length:(i + 1) * length, 2]
        T_results = results[i * length:(i + 1) * length, 3]

        # plot the R, A, T
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(wavelengths, R_results, label="R", color='black', linewidth=linewidth)
        plt.plot(wavelengths, A_results, label="A", color='red', linewidth=linewidth)
        plt.plot(wavelengths, T_results, label="T", color='blue', linewidth=linewidth)

        # plot the solar spectrum
        if solar != "":
            solar_arr = loadtxt(solar)
            plt.fill_between(solar_arr[:, 0],
                             solar_arr[:, 1] / max(solar_arr[:, 1]),
                             color='pink',
                             alpha=0.5)

        # plot configuration
        plt.xlim(start, end)
        plt.ylim(0, 1)
        plt.xlabel('Wavelength (\u03BCm)', fontsize=label_font_size)
        plt.ylabel('Spectral Responce', fontsize=label_font_size)
        plt.legend(loc='upper right', frameon=False, fontsize=axis_font_size)
        plt.savefig(str(output_name) + "_plot{}.png".format(i + 1))
    print("Results saved!")
    return


def main_func(file_name=None, return_array=False):

    # loop until input file has no identifiable errors
    # check is initially True, if the input file has no identifiable errors, check will be false
    check = True
    while check:
        if file_name:
            infile, check = check_file(file_name, check)
        else:
            # imports the provided txt file
            infile = fname()
        # check the input file for errors
        check, infile = check_input_for_errors(infile)

        # if the file looks good so far, go ahead and import / run Mie theory
        if not check:
            # imports information from header of input file and loads in necessary files
            particle, medium, output_name, solar, sims, photons, line, check, start, end, p_num, m_num, particle_type = import_header(infile, check)
            if not check:
                # calculates optical properties from inputs in infile
                prop, sims_per_medium, wavelengths, check = nanoparticle(infile, check, line, particle, medium, p_num, m_num, particle_type)
                # if nn, check the properties are within range
                if infile[0] == "nn":
                    check = check_NN_range(prop, check)


    # Send to either Monte Carlo or Neural Network
    if (infile[0][0:2]).lower() == "mc":
        # Monte Carlo
        results = main_mc(prop, photons)
        check = False
    elif (infile[0][0:2]).lower() == "nn":

        # Run NN prediction if properties are within NN range
        if check is False:
            results = forward(prop, sims*sims_per_medium)
    else:
        print("Please specify either MC or NN in the first line of the input file")
        print("Please re-enter input file once corrected")
        main_func()
        check = True


    # if solar spectrum is provided, integrate for solar reflectance
    solar_r, solar_a, solar_t, solar = integrate_solar(solar, sims, wavelengths, start, end, results, sims_per_medium)

    # write output script
    length = write_output_script(results, sims, output_name, solar, solar_r, solar_a, solar_t, wavelengths, prop, infile)

    # plot results
    plot_results(sims, results, length, wavelengths, solar, start, end, output_name)

    if return_array is True:
        wavelengths = wavelengths.reshape(-1, 1)
        R_results = (results[:, 0] + results[:, 1]).reshape(-1, 1)
        A_results = results[:, 2].reshape(-1, 1)
        T_results = results[:, 3].reshape(-1, 1)
        output_array = concatenate((wavelengths, R_results, A_results, T_results), axis=1)
        return output_array
    return


if __name__ == "__main__":
    print('\033[1m{: ^75s}\033[0m'.format("FOS"))
    print('{: ^75s}'.format("Fast Optical Spectrum calculations for nanoparticle media"))
    print('{: ^75s}'.format("Version: 0.6.6\n"))
    print('{: ^75s}'.format("Daniel Carne, Joseph Peoples, Ziqi Guo, Dudong Feng, Zherui Han, Xiaojie Liu, Xiulin Ruan"))
    print('{: ^75s}'.format("School of Mechanical Engineering, Purdue University"))
    print('{: ^75s}'.format("West Lafayette, IN 47907, USA\n"))

    main_func()
  