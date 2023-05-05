## Installing required packages

The `requirements.txt` file list all Python libraries that this project depend on, and they will be installed using:
```
pip install -r requirements.txt
```
The above command will install all the packages listed in the requirements.txt file using pip. If any of the packages fail to install, please make sure that you have the required dependencies installed on your system.

If you prefer to use conda, create a new environment for the project:
```
conda create --name FOS-env python=3.10
```
Replace project-env with the name of your environment. You can also use a different version of Python if needed.
Activate the environment:
```
conda activate FOS-env
```
Install the required packages using conda:
```
conda install --file requirements.txt
```