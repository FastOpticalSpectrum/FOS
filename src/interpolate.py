from numpy import amax, zeros, append, interp
from numba import njit


# calculates length of each input set of particles and mediums
@njit()
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


# calculates spacing for each particle and medium
@njit()
def calc_spacing(particle, medium, particle_spacing, medium_spacing, minimum, max, length):
    # loops through each particle
    for i in range(len(particle[:, 0, 0])):
        # move wavelength over
        particle_spacing[:, 0, i] = particle[i, :, 0]
        # find min spacing on either side of a point
        particle_spacing[0, 1, i] = particle[i, 1, 0] - particle[i, 0, 0]
        for j in range(1, length - 1):

            particle_spacing[j, 1, i] = min(particle[i, j + 1, 0] - particle[i, j, 0],
                                            particle[i, j, 0] - particle[i, j - 1, 0])
            if particle[i, j + 1, 0] == 0:
                particle_spacing[j, 1, i] = particle[i, j, 0] - particle[i, j - 1, 0]
        particle_spacing[length - 1, 1, i] = particle[i, length - 1, 0] - particle[i, length - 2, 0]
        # check min max
        if particle[i, 0, 0] > minimum:
            minimum = particle[i, 0, 0]
        if amax(particle[i, :, 0]) < max:
            max = amax(particle[i, :, 0])
    # loops through each medium
    for i in range(len(medium[:, 0, 0])):
        # move wavelength over
        medium_spacing[:, 0, i] = medium[i, :, 0]
        # find min spacing on either side of a point
        medium_spacing[0, 1, i] = medium[i, 1, 0] - medium[i, 0, 0]
        for j in range(1, length - 1):

            medium_spacing[j, 1, i] = min(medium[i, j + 1, 0] - medium[i, j, 0], medium[i, j, 0] - medium[i, j - 1, 0])
            if medium[i, j + 1, 0] == 0:
                medium_spacing[j, 1, i] = medium[i, j, 0] - medium[i, j - 1, 0]
        medium_spacing[length - 1, 1, i] = medium[i, length - 1, 0] - medium[i, length - 2, 0]
        if medium[i, 0, 0] > minimum:
            minimum = medium[i, 0, 0]
        if amax(medium[i, :, 0]) < max:
            max = amax(medium[i, :, 0])
    return minimum, max, medium_spacing, particle_spacing


# combines each one to find the minimum total spacing between all particles and mediums
@njit()
def calc_minimum_spacing(min_spacing, minimum, max, particle, medium, particle_spacing, medium_spacing, plength, mlength, length, mesh_percentage):
    for i in range(10000):
        min_spacing[0, i] = minimum + i*(max-minimum)/9999
        # start with first particle
        # interpolate for spacing
        min_spacing[1, i] = interp(min_spacing[0, i], particle_spacing[:plength[0], 0, 0], particle_spacing[:plength[0], 1, 0])
        # check all other particles and mediums if spacing is lower
        if len(particle[:, 0, 0]) > 1:
            for j in range(1, len(particle[:, 0, 0])):
                temp = interp(min_spacing[0, i], particle_spacing[:plength[j], 0, j], particle_spacing[:plength[j], 1, j])
                if temp < min_spacing[1, i]:
                    min_spacing[1, i] = temp


        for j in range(len(medium[:, 0, 0])):
            for k in range(length):
                temp = interp(min_spacing[0, i],medium_spacing[:mlength[j], 0, j], medium_spacing[:mlength[j], 1, j])
                if temp < min_spacing[1, i]:
                    min_spacing[1, i] = temp
        min_spacing[1, i] *= 1/mesh_percentage
    return min_spacing


# create mesh of new wavelengths
@njit()
def create_mesh(mesh, minimum, max, min_spacing):
    mesh[0] = minimum
    wavelength = minimum
    spacing = min_spacing[1, 0]
    spot = 0
    while wavelength < max:
        # iterate to find spacing
        check = False
        spacing = interp(wavelength, min_spacing[0, :], min_spacing[1, :])
        while check is False:

            new_wavelength = wavelength + spacing
            if new_wavelength > max:
                break
            for j in range(spot, 10000):
                if new_wavelength > min_spacing[0, j]:
                    if min_spacing[1, j] < spacing:
                        spacing = min_spacing[1, j]
                        spot = j
                        break
                else:
                    check = True
        for j in range(spot, 10000):
            if (wavelength + spacing) < min_spacing[0, j]:
                spot = j - 1
                break
        wavelength += spacing
        mesh = append(mesh, wavelength)
    mesh[-1] = max
    return mesh


# interpolate properties into new mesh
@njit()
def new_properties(particle2, medium2, plength, mlength, particle, medium, mesh):
    for i in range(len(particle2[:, 0, 0])):
        particle2[i, :, 0] = mesh[:]
        for j in range(len(mesh)):
            particle2[i, j, 1] = interp(particle2[i, j, 0], particle[i, :plength[i], 0], particle[i, :plength[i], 1])
            particle2[i, j, 2] = interp(particle2[i, j, 0], particle[i, :plength[i], 0], particle[i, :plength[i], 2])
    for i in range(len(medium2[:, 0, 0])):
        medium2[i, :, 0] = mesh[:]
        for j in range(len(mesh)):
            medium2[i, j, 1] = interp(medium2[i, j, 0], medium[i, :mlength[i], 0], medium[i, :mlength[i], 1])
            medium2[i, j, 2] = interp(medium2[i, j, 0], medium[i, :mlength[i], 0], medium[i, :mlength[i], 2])
    return particle2, medium2


def interpolate(particle, medium, length, mesh_percentage):
    particle_spacing = zeros((length, 2, len(particle[:, 0, 0])))
    medium_spacing = zeros((length, 2, len(medium[:, 0, 0])))

    # min and max are min and max wavelength within the range of every input
    minimum = particle[0, 0, 0]
    max = amax(particle[0, :, 0])

    # find length for each particle and medium
    plength = zeros(len(particle[:, 0, 0]), dtype=int)
    mlength = zeros(len(medium[:, 0, 0]), dtype=int)
    plength, mlength = calc_length(particle, medium, plength, mlength, length)

    minimum, max, medium_spacing, particle_spacing = calc_spacing(particle, medium, particle_spacing, medium_spacing, minimum, max, length)


    # combine them all to one
    min_spacing = zeros((2, 10000))
    min_spacing = calc_minimum_spacing(min_spacing, minimum, max, particle, medium, particle_spacing, medium_spacing, plength,mlength, length, mesh_percentage)

    # build new mesh with wavelengths based on minimum spacing
    mesh = zeros(1)
    mesh = create_mesh(mesh, minimum, max, min_spacing)

    # Interpolate properties into new mesh
    particle2 = zeros((len(particle[:, 0, 0]), len(mesh), 3))
    medium2 = zeros((len(medium[:, 0, 0]), len(mesh), 3))
    particle2, medium2 = new_properties(particle2, medium2, plength, mlength, particle, medium, mesh)

    return particle2, medium2
