from numpy import loadtxt, zeros, append, vstack, asarray, savetxt, dstack, round
from MieTheory3 import mie_theory, effective_medium
from MonteCarlo import main_mc
from SolarIntegration import solar_spectrum
import os.path
from NN_feedforward import forward
from interpolate import interpolate



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


# imports information from header of input file and loads in necessary files
def import_header(infile):
    # p is list of particle input file names, m for medium input files.
    # each of these files consists of three columns, wavelength, refractive index, and extinction coefficient
    p = zeros(0, dtype=str)
    m = zeros(0, dtype=str)
    # solar spectrum file to import. If there is not one, this variable remains blank
    solar = ""
    # ooutput file name
    output_name = ""
    # mesh percentage, 1 keeps data as is. Above 1 increases the number of mesh points, below 1 decreases
    mesh_percentage = 1

    photons = 0
    line = 0
    # loops through header. Breaks loop when it hits the first "sim"
    for i in range(len(infile)):
        if infile[i][0:8] == "particle":
            p = append(p, infile[i][10:])
        if infile[i][0:6] == "medium":
            m = append(m, infile[i][8:])
        if infile[i][0:6] == "output":
            output_name = infile[i][7:]
        if infile[i][0:5] == "solar":
            solar = infile[i][6:]
        if infile[i][0:4] == "mesh":
            mesh_percentage = float(infile[i][5:])
        if infile[i][0:7] == "photons":
            photons = int(infile[i][8:])
        if infile[i][0:3] == "sim":
            line = i-1
            break

    # counts number of simulations
    sims = 0
    for i in range(len(infile)):
        if infile[i][0:3] == "sim":
            sims += 1

    # length is the length (number of wavelength datapoints) of the longest input file
    length = 0
    for i in range(len(p)):
        temp = loadtxt(p[i])
        if len(temp) > length:
            length = len(temp)
    for i in range(len(m)):
        temp = loadtxt(m[i])
        if len(temp) > length:
            length = len(temp)

    # arrays for material properties [number of materials, wavelengths, properties]
    particle = zeros((len(p), length, 3))
    medium = zeros((len(m), length, 3))

    # need_interp is True if wavelengths of input files don't match, or the mesh_percentage is not 1
    need_interp = False

    # imports particle files and checks if they need to be interpolated
    temp =loadtxt(p[0])
    particle[0, :len(temp), :] = temp
    if len(p) > 1:
        for i in range(1, len(p)):
            temp = loadtxt(p[i])
            particle[i, :len(temp), :] = temp
            if len(temp) != length:
                need_interp = True
            for j in range(length):
                if particle[i, j, 0] < (particle[0, j, 0] - 0.01) or particle[i, j, 0] > (particle[0, j, 0] + 0.01):
                    need_interp = True

    # imports medium files and checks if they need to be interpolated
    for i in range(len(m)):
        temp = loadtxt(m[i])
        medium[i, :len(temp), :] = temp
        if len(temp) != length:
            need_interp = True
        # check wavelengths match
        for j in range(length):
            if medium[i, j, 0] < (particle[0, j, 0] - 0.01) or medium[i, j, 0] > (particle[0, j, 0] + 0.01):
                need_interp = True

    # send to interpolation method if the wavelengths don't match or the mesh_percentage is not 1
    if need_interp is True:
        print('Interpolating properties to match wavelengths for each input')
        particle, medium = interpolate(particle, medium, length, mesh_percentage)
    elif mesh_percentage != 1:
        print('Interpolating properties for new mesh')
        particle, medium = interpolate(particle, medium, length, mesh_percentage)
    return particle, medium, output_name, solar, sims, photons, line


def check_diameters(current_sim, fv, sizes, check):
    # check to make sure same number of diameters and volume fraction
    if len(fv) != len(sizes):
        print('Number of diameters does not match number of volume fractions provided in sim:', current_sim)
        print("Please re-enter input file once corrected.")
        check = True
    return check


# finds info from input and sends to Mie Theory to calculate optical properties
def optical(line, infile, particle, medium, check):
    print("Running Mie theory")
    # array of optical properties to send to Monte Carlo
    prop = zeros((0, 5))
    optics_sum = zeros((5, len(particle[0, :, 0])))
    optical_per_layer = zeros((5, len(particle[0, :, 0]), 0))
    vol_frac_sum = 0
    layers = 1
    count = 0

    current_sim = 1
    for i in range(line+1, len(infile)):
        if infile[i][0:8] == "particle":
            if count > 0:
                optics = mie_theory(sizes, fv, particle[int(ptype - 1), :, :], medium[int(mtype - 1), :, :], thickness, dist)
                optics_sum += optics
            count += 1
            ptype = int(infile[i][8])
        if infile[i][0:5] == "upper":
            upper = float(infile[i][6:])
        elif infile[i][0:5] == "lower":
            lower = float(infile[i][6:])
        elif infile[i][0:6] == "medium":
            mtype = int(infile[i][6])
        elif infile[i][0:2] == "t:":
            thickness = float(infile[i][2:])
            thickness = thickness / 10000
        elif infile[i][0:2] == "d:":
            sizes = (infile[i][2:]).split(",")
            sizes = asarray(sizes, dtype=float)
            sizes = sizes/2
        elif infile[i][0:3] == "vf:":
            fv = (infile[i][3:]).split(",")
            fv = asarray(fv, dtype=float)
            fv = fv/100
            vol_frac_sum += sum(fv)
        elif infile[i][0:5] == "dist:":
            dist = float(infile[i][5:])
            dist = dist/100
        elif infile[i][0:5] == "layer" and infile[i][5:] != "1":
            # check to make sure same number of diameters as volume fractions
            check = check_diameters(current_sim, fv, sizes, check)
            layers += 1
            optics = mie_theory(sizes, fv, particle[int(ptype - 1), :, :], medium[int(mtype - 1), :, :], thickness, dist)
            optics_sum += optics
            optics_sum[0, :] = optics[0, :]
            optics_sum[4, :] = optics[4, :]
            optics_sum = effective_medium(optics_sum, vol_frac_sum, medium[int(mtype - 1), :, :])
            optical_per_layer = dstack((optical_per_layer, optics_sum))
            vol_frac_sum = 0
            optics_sum = zeros((5, len(particle[0, :, 0])))
            count = 0
        elif infile[i][0:3] == "sim" and infile[i][4:] != "1":
            current_sim = int(infile[i][4:])
            # check to make sure same number of diameters as volume fractions
            check = check_diameters(current_sim, fv, sizes, check)
            optics = mie_theory(sizes, fv, particle[int(ptype - 1), :, :], medium[int(mtype - 1), :, :], thickness, dist)
            optics_sum += optics
            optics_sum[0, :] = optics[0, :]
            optics_sum[4, :] = optics[4, :]
            optics_sum = effective_medium(optics_sum, vol_frac_sum, medium[int(mtype - 1), :, :])
            optical_per_layer = dstack((optical_per_layer, optics_sum))
            for j in range(len(particle[0, :, 0])):
                prop = vstack((prop, [upper, 0, 0, 0, 0]))
                for k in range(layers):
                    prop = vstack((prop, optical_per_layer[:, j, k]))
                prop = vstack((prop, [lower, 0, 0, 0, 0]))
                prop = vstack((prop, [0, 0, 0, 0, 0]))
            vol_frac_sum = 0
            optics_sum = zeros((5, len(particle[0, :, 0])))
            optical_per_layer = zeros((5, len(particle[0, :, 0]), 0))
            layers = 1
            count = 0
    # check to make sure same number of diameters as volume fractions
    check = check_diameters(current_sim, fv, sizes, check)
    optics = mie_theory(sizes, fv, particle[int(ptype - 1), :, :], medium[int(mtype - 1), :, :], thickness, dist)
    optics_sum += optics
    optics_sum[0, :] = optics[0, :]
    optics_sum[4, :] = optics[4, :]
    optics_sum = effective_medium(optics_sum, vol_frac_sum, medium[int(mtype - 1), :, :])
    optical_per_layer = dstack((optical_per_layer, optics_sum))
    for j in range(len(particle[0, :, 0])):
        prop = vstack((prop, [upper, 0, 0, 0, 0]))
        for k in range(layers):
            prop = vstack((prop, optical_per_layer[:, j, k]))
        prop = vstack((prop, [lower, 0, 0, 0, 0]))
        prop = vstack((prop, [0, 0, 0, 0, 0]))
    return prop, check


# breaks down import file, runs Mie Theory calculations
def nanoparticle(infile, check):
    # imports information from header of input file and loads in necessary files
    particle, medium, output_name, solar, sims, photons, line = import_header(infile)

    # gets optical properties of each simulation
    prop, check = optical(line, infile, particle, medium, check)
    sims_per_medium = len(particle[0, :, 0])
    wavelengths = particle[0, :, 0]

    return prop, photons, output_name, sims, sims_per_medium, solar, wavelengths, check



# Makes sure the input data is within the range the NN is trained on
def check_NN_range(prop, check):
    # If check is True, NN does not run and the user must change the input file
    for i in range(len(prop[:, 0])):
        # checks refractive indices
        if prop[i, 0] != 0:
            if prop[i, 4] == 0 and prop[i, 0] != 1:
                print("Neural network is only trained on boundary indices of 1")
                check = True
            if prop[i, 0] < 1 or prop[i, 0] > 10:
                print("A refractive index of" + str(prop[i, 0]) + "is out of bounds [1, 7]")
                check = True
        # Checks the absorption coefficient, scattering coefficient, asymmetry parameter
        if prop[i, 4] != 0:
            # absorption coefficient
            if prop[i, 1] < 0 or prop[i, 1] > 300000:
                print("An absorption coefficient of" + str(prop[i, 1]) + "is out of bounds [0, 1,000,000] (1/cm)")
                check = True
            # scattering coefficient
            if prop[i, 2] < 0 or prop[i, 2] > 150000:
                print("A scattering coefficient of" + str(prop[i, 2]) + "is out of bounds [0, 200,000] (1/cm)")
                check = True
            # asymmetry parameter
            if prop[i, 3] < 0 or prop[i, 3] > 1:
                print("An asymmetry parameter of" + str(prop[i, 3]) + "is out of bounds [0, 1]")
                check = True
        # check for multi layered sims
        if prop[i, 4] != 0 and prop[i+1, 4] != 0:
            print("The neural network cannot predict multi-layer media")
            check = True
        # check paint thickness
        if prop[i, 4] != 0:
            if prop[1, 4] < .0005 or prop[1, 4] > .05:
                print("A thickness of " + str(prop[1, 4]*10000) + " is out of bounds [5, 500] \u03BCm")
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
        if (infile[i][:3] == "sim" and infile[i][4:] != '1'):

            if word_present == 1:
                check = True
                print("Sim:" + sim_number + " must have " + statement)
            sim_number = infile[i][4:]
            word_present = 1
        if infile[i][:word_size] == word:
            word_present = 0
    if word_present == 1:
        check = True
        print("Sim:" + sim_number + " must have " + statement)
    return check

# check the input file for errors
def check_input_for_errors(infile):
    # check = True is there is an issue caught with the input file, otherwise false
    check = False

    # remove all spaces and make lowercase to clean things up
    for i in range(len(infile)):
        infile[i] = infile[i].replace(' ', '')
        infile[i] = infile[i].lower()

    # first line must specify "nn" or "mc"
    if infile[0] != "mc" and infile[0] != "nn":
        print("Either MC or NN must be specified in the first line of the input file.")
        check = True

    ### check for each item in header
    # initially these variables are set to 0, if they are changed to 1 then it is good. If it remains 0 then there is an error
    output = particle = medium = photon = 0
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
        if infile[i][:6] == 'medium' and infile[i][8:] != '':
            medium = 1
        # number of photons if using Monte Carlo
        if infile[0] == "mc":
            if infile[i][:7] == "photons" and infile[i][9:] != '':
                photon = 1
        if infile[i][:4] == "mesh":
            if float(infile[i][5:]) <= 0:
                check = True
                print('Mesh value must be > 0')
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
        print("No medium input specified")
    if photon == 0:
        check = True
        print("Number of photons must be specified")

    ### check for errors in the body of the file
    # make sure each sim is numbered correctly
    sim_number_should_be = 1
    # if sim_number_error is flipped to 0, there is an error
    sim_number_error = 1
    for i in range(len(infile)):
        if infile[i][:3] == "sim":
            if infile[i][4:] != str(sim_number_should_be):
                sim_number_error = 0

            sim_number_should_be += 1
    if sim_number_error == 0:
        check = True
        print("Error in simulation numbers. Make sure they are labeled Sim: 1, Sim: 2, etc.")

    # check for upper boundary condition
    check = check_for_word_in_sim(infile, 'upper', 'defined upper boundary condition', check)
    # check for lower boundary condition
    check = check_for_word_in_sim(infile, 'lower', 'defined lower boundary condition', check)
    # check for at least one layer
    check = check_for_word_in_sim(infile, 'layer', 'at least one defined layer', check)
    # check for at least one medium
    check = check_for_word_in_sim(infile, 'medium', 'at least one medium', check)
    # check for medium thickness
    check = check_for_word_in_sim(infile, 't:', 'a defined thickness', check)
    # check for at least one particle
    check = check_for_word_in_sim(infile, 'particle', 'at least one particle', check)
    # check for particle size
    check = check_for_word_in_sim(infile, 'd:', 'a defined particle diameter', check)
    # check for VF
    check = check_for_word_in_sim(infile, 'vf', 'a defined particle volume fraction', check)
    # check for distribution
    check = check_for_word_in_sim(infile, 'dist', 'a defined distribution. Put Dist: 0 for no distribution', check)


    if check:
        print("Please re-enter input file once corrected.")
        print("\n")
    return check, infile


def main_func():

    # loop until input file has no identifiable errors
    # check is initially True, if the input file has no identifiable errors, check will be false
    check = True
    while check:
        # imports the provided txt file
        infile = fname()
        # check the input file for errors
        check, infile = check_input_for_errors(infile)

        # if the file looks good so far, go ahead and import / run Mie theory
        if not check:
            # calculates optical properties from inputs in infile
            prop, photons, output_name, sims, sims_per_medium, solar, wavelengths, check = nanoparticle(infile, check)
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
            results = forward(prop)
    else:
        print("Please specify either MC or NN in the first line of the input file")
        print("Please re-enter input file once corrected")
        main_func()
        check = True


    # if solar spectrum is provided, integrate for solar reflectance
    if solar != "":
        # solar reflectance value for each simulation
        solar_r = zeros(sims)
        # reflectance at each wavelength to be integrated with the solar spectrum
        refl = zeros((len(wavelengths), 2))
        # move wavelengths over to refl
        refl[:, 0] = wavelengths
        # loop through each simulation to calculate solar reflectance
        for i in range(sims):
            # add the specular and diffuse reflectance together for total reflectance
            refl[:, 1] = (results[i*sims_per_medium:(i+1)*sims_per_medium, 0]+results[i*sims_per_medium:(i+1)*sims_per_medium, 1])
            # send to function to integrate
            solar_r[i] = solar_spectrum(solar, refl)
        # save solar reflectance of each sim to output file
        output_solar = str(output_name) + "_solar.txt"
        with open(output_solar, 'w') as f:
            for i in range(len(solar_r)):
                f.write('Sim ' + str(i+1) + ": ")
                f.write(str(round(solar_r[i], 4)))
                f.write('\n')
        f.close()

    # save output scripts
    length = int(len(results[:, 0])/sims)
    for i in range(sims):
        output_sim = str(output_name) + str(i+1) +".txt"
        with open(output_sim, 'w') as f:
            f.write('Wavelength\tSpecular R\tDiffuse R\tA\tT')
            f.write('\n')
            for j in range(length):
                f.write(str(round(wavelengths[j], 4)) + '\t')
                f.write(str(round(results[j+length*i, 0], 4)) + '\t')
                f.write(str(round(results[j + length * i, 1], 4)) + '\t')
                f.write(str(round(results[j + length * i, 2], 4)) + '\t')
                f.write(str(round(results[j + length * i, 3], 4)) + '\t')
                f.write('\n')
        f.close()
    print("Results saved!")
    return


if __name__ == "__main__":
    print('\033[1m{: ^75s}\033[0m'.format("FOS"))
    print('{: ^75s}'.format("Fast Optical Spectrum calculations for nanoparticle media"))
    print('{: ^75s}'.format("Version: 0.2.0\n"))
    print('{: ^75s}'.format("Daniel Carne, Joseph Peoples, Zherui Han, Dudong Feng, Xiulin Ruan"))
    print('{: ^75s}'.format("School of Mechanical Engineering, Purdue University"))
    print('{: ^75s}'.format("West Lafayette, IN 47909, USA\n"))

    main_func()
