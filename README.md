# MCPM
Code for MCPM

# Efficient Inference in Multi-task Cox Process Models
An implementation of the model described in [Efficient Inference in Multi-task Cox Process Models](https://arxiv.org/abs/1805.09781).

The code was tested on Python 2.7, [TensorFlow 1.12](https://www.tensorflow.org/get_started/os_setup) and [TensorFlow Probability 0.5](https://www.tensorflow.org/probability)


# Installation
You can download and install MCPM in development mode using:
```
git clone git@github.com:VirgiAgl/MCPM.git
pip install -e MCPM
```
# Usage
The main code is in `MCPM/MCPM.py`. The folder `methods/` contains the functions to run LGCP, MCPM, ICM and Pooling. You will find all the experiments presented in the paper in the folder `Experiments/`. The results are saved in the folder `Data/` when running the experiments. The folder `Results_visualisation/` contains the notebooks to get the plots and the performance measures presented in the paper. 

# Contact
You can contact the first author of the paper [Virginia Aglietti](https://warwick.ac.uk/fac/sci/statistics/staff/research_students/aglietti/)

# Acknowledgements
The code to support triangular matrices operations under `autogp/util/tf_ops` was taken from the GPflow repository (Hensman, Matthews et al. GPflow, http://github.com/GPflow/GPflow, 2016).

