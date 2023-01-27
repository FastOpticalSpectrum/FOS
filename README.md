# __`FOS`__
__`FOS`__, which means "light" in Greek, is used for Fast Optical Spectrum (`FOS`) calculations of nanoparticle media by combining Mie theory with either Monte Carlo simulations or a pre-trained deep neural network.
## Authors and References
+ Daniel Carne: dcarne@purdue.edu
+ Joseph Peoples: 
+ Zherui Han: zrhan@purdue.edu
+ Dudong Feng: feng376@purdue.edu
+ Xiulin Ruan:ruan@purdue.edu


## How to download and use __`FOS`__
There are two main ways to use this program. On a Windows operating system one can use the FOS.exe file. This .exe file does not have any dependencies so you do not need to have Python on your computer. Alternatively, for other operating systems or for security reasons, one can use the Python source code in the src folder by simply running Main3.py. This was built in Python 3.10 and is not guranteed to work in other versions. 

## Input file
Each input file contains a header and a body. The header consists of information that is not repeated per simulation. Upper / lower case text does not matter, and a # comments out a line for your personal comments. As shown below, what is needed is:
+ Either "MC" or "NN" in the first line to specify Monte Carlo simulation or Neural Network prediction.
+ The output file prefix.
+ At least one particle .txt file with three columns, the wavelength (in microns), refractive index, and extinction coefficient.
+ At least one medium .txt file with three columns, the wavelength (in microns), refractive index, and extinction coefficient.
+ The mesh setting (defaults to 1 if not included).
+ The solar spectrum to integrate with the reflectance (not required).
+ The number of photons if running a Monte Carlo simulation.

More information about item in the input file is in the appendix at the bottom. 

```
mc

# a line starting with a hashtag comments it out

output: test

particle 1: BaSO4.txt
particle 2: TiO2.txt
medium 1: acrylic.txt

mesh: 0.5

solar: AM15.txt

photons: 100000
```

The body of the input file consists of information for each simulation. An example body is shown below with one simple simulation, and one with more complex features. In each simulation, the following is required:
+ Sequentially ordered simulation number (1, 2, 3, ..., n).
+ Upper and lower boundary condition refractive index, this number is applied to all wavelengths simulated.
+ Within each simulation each layer must be labelled. Sim: 1 below only has one layer, Sim: 2 has 2 layers.
+ Each layer must include at least one medium and one particle.
+ The number next to medium or particle refer to the file imported from the header. For example, in Sim: 1 when it refers to Particle 2, that uses the TiO2.txt file which was imported in the header as Particle 2.
+ Under medium, the layer thickness must be specified. All units in the input file are in microns.
+ Under each particle, the diamater, volume fraction, and distribution must be specified. Multiple diameters can be provided under one particle such as in Sim: 2 layer 1
+ Multiple particle types can be used within a layer such as in Sim: 2 Layer 1 by specifying the particles sequentially before beginning layer 2


```
Sim: 1
Upper: 1
Lower: 1
Layer 1
Medium 1
T: 100
Particle 2
D: 0.4
VF: 60
Dist: 0

Sim: 2
Upper: 1
Lower: 1
Layer 1
Medium 1
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
Medium 1
T: 50
Particle 1
D: 0.5
VF: 60
Dist: 45
```
An example input file is provided (input.txt) in the example folder which shows examples of different simulation features.

## Output files
There will be an output file generated for each simulation ran. Each file will be named the output file prefix followed by the simulation number. For example, using the above input file the output files would be test1.txt and test2.txt. Each of these files will have 5 columns, the wavelength, specular reflectance, diffuse reflectance, absortance, and transmittance.

If a solar spectrum file is included in the input header, then there will be an additional file named the output prefix followed by \_solar,  test\_solar.txt in this example. This file includes the solar reflectance of each simulation.

## Appendix
