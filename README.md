# __`FOS`__
__`FOS`__, which means "light" in Greek, is used for Fast Optical Spectrum (`FOS`) calculations for nanoparticle media by combining Mie theory with either Monte Carlo simulations or a pre-trained deep neural network.
## Authors and References
+ Daniel Carne: dcarne@purdue.edu
+ Joseph Peoples: 
+ Ziqi Guo: gziqi@purdue.edu
+ Dudong Feng: feng376@purdue.edu
+ Zherui Han: zrhan@purdue.edu
+ Xiulin Ruan:ruan@purdue.edu

Cite: 

## How to download and use __`FOS`__
There are two ways to use this program. On a Windows operating system one can use the FOS.exe file. This .exe file does not have any dependencies so you do not need to have Python on your computer. This file can be downloaded through the latest release and may take a few seconds to load after opening.

Alternatively, for other operating systems or for security reasons, one can use the Python source code in the src folder by simply running Main3.py. This was built in Python 3.10 and may or may not work in other versions. 

If you have any questions, issues, or requests, please put them in the Discussions tab!

## Installing required packages

The `requirements.txt` file lists all Python libraries that this project depends on. If you are not using the executable file, they can be installed using:
```
pip install -r requirements.txt
```
The above command will install all the packages listed in the requirements.txt file using pip. If any of the packages fail to install, please make sure that you have the required dependencies installed on your system.

If you prefer to use conda, create a new environment for the project:
```
conda create --name FOS_env python=3.10
```
Replace project-env with the name of your environment. You can also use a different version of Python if needed.
Activate the environment:
```
conda activate FOS_env
```
Install the required packages using conda:
```
conda install --file requirements.txt
```

## Input file
Each input file contains a header and a body. The header consists of information that is not repeated per simulation. Upper / lower case text does not matter, and a # comments out a line for your personal comments. As shown in the example below, what is needed is:
+ Either "MC" or "NN" in the first line to specify Monte Carlo simulation or Neural Network prediction.
+ The output file prefix.
+ At least one particle .txt file with three columns, the wavelength (in microns), refractive index, and extinction coefficient.
+ At least one matrix .txt file with three columns, the wavelength (in microns), refractive index, and extinction coefficient.
+ The mesh setting (defaults to 1 if not included).
+ The solar spectrum to integrate with the reflectance (not required).
+ The number of photons if running a Monte Carlo simulation.
+ The start and end wavelength to simulate.

Example header:

```
MC
# MC for Monte-Carlo or NN for neural network, must be in the first line

# hashtags comment a line out
# output defines the prefix for output files
Output: test	# comments can also be inline next to input

# import any materials you will use as particles or matrixes
# materials files need three coloumns, wavelength, refractive index, and extinction coefficient
Particle 1: TiO2.txt
Particle 2: BaSO4.txt
Matrix 1: acr.txt
Matrix 2: air.txt

# mesh allows you to control the density of wavelengths simulated
# if mesh is not included, it will default to 1
Mesh: 0.5

# if solar is included, the output file will include the integrated solar response
Solar: am15.txt

# number of photons per wavelength for Monte Carlo simulations
Photons: 30000

# wavelength range tested in microns
# if this is left out, it will default to the broadest range all materials provide
Start: 0.25
End: 2.5
```

The body of the input file consists of information for each simulation. An example body is shown below with one simple simulation, and one with more complex features. In each simulation, the following is required:
+ Sequentially ordered simulation number (1, 2, 3, ..., n).
+ Upper and lower boundary condition refractive index, this number is applied to all wavelengths simulated.
+ Within each simulation each layer must be labelled. Sim 1 below only has one layer, Sim 2 has 2 layers.
+ Each layer must include at least one matrix and one particle.
+ The number next to medium or particle refer to the file imported from the header. For example, in Sim 1 when it refers to Particle 2, that uses the TiO2.txt file which was imported in the header as Particle 2.
+ Under matrix, the layer thickness must be specified. All units in the input file are in microns.
+ Under each particle, the diamater, volume fraction, and distribution must be specified. Multiple diameters can be provided under one particle such as in Sim 2 layer 1
+ Multiple particle types can be used within a layer such as in Sim 2 Layer 1 by specifying the particles sequentially before beginning layer 2

Example body:
```
Sim 1
Upper: 1
Lower: 1
Layer 1
Matrix 1
T: 100
Particle 2
D: 0.4
VF: 60
Dist: 0

Sim 2
Upper: 1
Lower: 1
Layer 1
Matrix 2
T: 50
Particle 1
D: 0.4, 0.8
VF: 10, 20
Dist: 10
Particle 2
D: 0.5, 0.2
VF: 20, 10
Dist: 20
Layer 2
Matrix 1
T: 50
Particle 1
D: 0.5
VF: 60
Dist: 45
```
An example input file is provided (main_example/input.txt) in the examples folder which shows examples of different simulation features.

## Output files
Two output files will be generated per simulation, one with data about the simulation and one with the plotted spectral response. In the file with data, it will include the solar reflectance, absorptance, and transmittance if the solar file is provided in the input, the spectral response at each wavelength simulated, and a copy of the input file information for that simulation at the end for reference.

## Dictionary

| Item | Location    | Units    | Description    |
| :---:   | :---: | :---: | :---: |
| MC/NN | Header   | -   | Either MC or NN must be specified in the first line for Monte Carlo or Neural Network   |
| D | Body   | \DeclareUnicodeCharacter{039C}{\Mu} m   | Particle diameter   |
