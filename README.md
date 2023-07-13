# Multi-Class Text Classification for Emotions

* Lab Course on Computational Empirical Research, Summer Term 2023
* University of Leipzig

## Getting started

### Conda environment

This project uses a local conda environment called ``emo``. It isolates the Python runtime and 
the required packages from the rest of the system where the code is run. To setup your local 
development environment, consider the following commands:

``` bash
# Create a conda environment based on the `environment.yml`.
(base) conda env create --file environment.yml

# To activate the new conda environment, use
(base) conda activate emo

# In case more packages are needed, install them using conda (or pip)
(emo) conda install scikit-learn

# Make sure to update the `environment.yml` file, so that others can update their conda env.
# If the `environment.yml` has been changed by others, you can update your own environment
# 'emo' using this command:
(base) conda env update --file environment.yml
```

