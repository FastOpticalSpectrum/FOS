# __`FOS`__
__`FOS`__, which means "light" in Greek, is used for Fast Optical Spectrum (`FOS`) calculations for nanoparticle media by combining Mie theory with either Monte Carlo simulations or a pre-trained deep neural network.
## Authors and References
+ Daniel Carne: dcarne@purdue.edu
+ Joseph Peoples: peoplesj@purdue.edu
+ Ziqi Guo: gziqi@purdue.edu
+ Dudong Feng: feng376@purdue.edu
+ Zherui Han: zrhan@purdue.edu
+ Xiulin Ruan: ruan@purdue.edu

Cite: https://doi.org/10.1016/j.cpc.2024.109393

## How to download and use __`FOS`__
There are two ways to use this program. On a Windows operating system one can use the FOS.exe file. This .exe file does not have any dependencies so you do not need to have Python or install any libraries. This file can be downloaded through the latest release and may take a few seconds to load after opening.

Alternatively, for other operating systems or for security reasons, one can use the Python source code in the src folder by simply running Main3.py. This was built in Python 3.10 and may or may not work in other versions. Once all dependencies are installed, FOS can be run in the terminal with "python Main3.py". All python files, input files, material files, and the folder titled "Model" must be located in the same directory.

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
## Running FOS
Option 1: Run the file 'Main3.py' within a Python environment.

Option 2: Run the executable 'FOS.exe'. Does not require a local Python installation.

Option 3: From a seperate Python file, FOS can be called. First, import the function using:
```
from Main3 import main_func
```
Then, call the function passing two arguments. First, the name of the input file, and second, a boolean where True returns an array and False does not:
```
spectral_response = main_func('input.txt', True)
or
main_func('input.txt', False)
```

## Input file
Each input file contains a header and a body. The header consists of information that is not repeated per simulation. Upper / lower case text does not matter, and a # comments out a line for your personal comments. It is highly encouraged to view the example input files. Under the examples folder, the "main_example" input file contains simulations with many different examples including multiple layers, multiple particle types, inputting custom properties, core-shell particles, etc. As shown in the example below, what is needed is:
+ Either "MC" or "NN" in the first line to specify Monte Carlo simulation or Neural Network prediction.
+ The output file prefix.
+ At least one particle .txt file with either the complex refractive index or pre-calculated scattering properties.
+ At least one matrix .txt file with three columns, the wavelength (in microns), refractive index, and extinction coefficient.
+ The mesh setting (defaults to 1 if not included).
+ The solar spectrum to integrate with the reflectance (not required).
+ The number of photons if running a Monte Carlo simulation.
+ The start and end wavelength to simulate, as well as the wavelength interval.
+ Custom optical properties can be used (look at the example in main_example)

Example header:

```
MC
# MC for Monte-Carlo or NN for neural network, must be in the first line

# hashtags comment a line out
# output defines the prefix for output files
Output: test	# comments can also be inline next to input

# import any materials you will use as particles or matrixes
# materials files need three columns, wavelength [microns], refractive index, and extinction coefficient
Particle 1: TiO2.txt 
Particle 2: BaSO4.txt
Matrix 1: acr.txt
Matrix 2: air.txt

# if solar is included, the output file will include the integrated solar response
Solar: am15.txt

# number of photons per wavelength for Monte Carlo simulations
Photons: 30000

# wavelength range and interval in microns
# if this is left out, it will default to the broadest range all materials provide
Start: 0.25
End: 2.5
# must specify a wavelength interval
Interval: 0.005
```

The body of the input file consists of information for each simulation. An example body is shown below with one simple simulation, and one with more complex features. In each simulation, the following is required:
+ Sequentially ordered simulation number (1, 2, 3, ..., n).
+ Upper and lower refractive index boundaries. If not included, it will default to air (n=1)
+ Within each simulation each layer must be labelled. Sim 1 below only has one layer, Sim 2 has 2 layers.
+ Each layer must include at least one matrix and one particle.
+ The number next to medium or particle refer to the file imported from the header. For example, in Sim 1 when it refers to Particle 2, that uses the TiO2.txt file which was imported in the header as Particle 2.
+ Under matrix, the layer thickness must be specified. All units in the input file are in microns.
+ Under each particle, the diamater and volume fraction must be specified. Multiple diameters can be provided under one particle such as in Sim 2 layer 1
+ Additionally, below each particle the diameter standard deviation can be specified for one or multiple particles. Defaults to 0 if not included.
+ Multiple particle types can be used within a layer such as in Sim 2 Layer 1 by specifying the particles sequentially before beginning layer 2

Example body:
```
Sim 1
Upper: Matrix 2
Lower: Matrix 2
Layer 1
Matrix 1
T: 100
Particle 2
D: 0.4
VF: 60
Std: 0

Sim 2
Layer 1
Matrix 2
T: 50
Particle 1
D: 0.4, 0.8
VF: 10, 20
Std: 0.05, 0.1
Particle 2
D: 0.5, 0.2
VF: 20, 10
Layer 2
Matrix 1
T: 50
Particle 1
D: 0.5
VF: 60
Std: 0.2
```
An example input file is provided (main_example/input.txt) in the examples folder which shows examples of different simulation features.

## Output files
Two output files will be generated per simulation, one with data about the simulation and one with the plotted spectral response. In the file with data, it will include the solar reflectance, absorptance, and transmittance if the solar file is provided in the input, the spectral response and scattering properties (of each layer) at each wavelength simulated, and a copy of the input file information for that simulation at the end for reference. All units in the input and output files are in microns.

## Video tutorial (Version 0.5.x)
[![FOS video tutorial](https://i.imgur.com/FgmoT5N.png)](https://www.youtube.com/watch?v=fCwUsdP4lq8 "FOS tutorial")

## Dictionary

| Item  | Location    | Units    | Description    |
| :---:   | :---: | :---: | :---: |
| MC/NN | Header   | -   | Either MC or NN must be specified in the first line for Monte Carlo or Neural Network.   |
| Output: | Header   | -   | Output file prefix.   |
| Particle 3: | Header   | μm   | .txt materials file for a particulate material with n and k, each particle can be specified by any integer (3 in this case). The wavelength units in the material file must be in microns.   |
| Matrix 5: | Header   | μm   | .txt materials file for a matrix material with n and k, each matrix can be specified by any integer (5 in this case). The wavelength units in the material file must be in microns.   |
| Solar: | Header   | μm    | .txt file with solar power at each wavelength (or anything you want to integrate, does not need to just be solar). The wavelength units in the material file must be in microns.   |
| Photons | Header   | -    | Number of photons (energy bundles) to use per wavelength for Monte Carlo simulations, not required for NN.   |
| Start: | Header   | μm    | Shortest wavelength of wavelength range to simulate.   |
| End: | Header   | μm    | Longest wavelength of wavelength range to simulate.   |
| Interval: | Header   | μm    | Interval between wavelengths to simulate.   |
| Sim 7 | Body   | -    | Simulation number (7 in this case), must be sqeuentially numbered starting with 1. |
| Upper: | Body   | -    | Upper (top surface) refractive index. Defaults to air if left out, a matrix material is specified here otherwise. |
| Lower: | Body   | -    | Lower (bottom surface) refractive index value. Defaults to air if left out, a matrix material is specified here otherwise. |
| Layer 1 | Body   | -    | Layer number. If single layer simulation, the first layer must still be labelled 'Layer 1'. Layer numbers must be sequentially numbered starting with 1 where the first layer is the upper surface.  |
| Matrix 5 | Body   | -    | For each layer, one matrix must be chosen. Here, 'Matrix 5' would refer to the material file imported in the header.  |
| T: | Body   | μm    | Thickness of layer must be specified after the Matrix is set.  |
| Particle 3 | Body   | -    | At least one particle (or multiple) must be specified per layer. Here, 'Particle 3' would refer to the material file imported in the header.  |
| D: | Body   | μm    | Particle diameter, must have diameter set for each solid particle (not core-shell).    |
| C: | Body   | μm    | Particle core diameter, must have core diameter set for each core-shell particle (not solid).    |
| S: | Body   | μm    | Particle shell wall thickness, must have shell wall thickness set for each core-shell particle (not solid).    |
| VF: | Body   | -    | Particle volume fraction, must set for each particle.   |
| Std: | Body   | -    | Particle size standard deviation, defaults to 0 if not included. Only applicable for solid particles, does not work for core-shell particles. Particle sizes up to 3 times the standard deviation are modelled. |

