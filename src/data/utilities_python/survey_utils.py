#! /usr/bin/env python

# Victor Calderon
# February 14, 2018
# Vanderbilt University

"""
Compilation of useful functions and definitons for ECO, 
RESOLVE A and B galaxy redshift surveys
"""
__author__     =['Victor Calderon']
__copyright__  =["Copyright 2017 Victor Calderon, eco_utils"]
__email__      =['victor.calderon@vanderbilt.edu']
__maintainer__ =['Victor Calderon']
__all__        =["closest_val"]

## Importing modules
import numpy as num

## Functions

def closest_val(val, arr):
    """
    Finds the closest value in `arr` to `val`

    Parameters
    -------------
    val: int or float
        value to be looked at

    arr: numpy.ndarray
        numpy array used for finding closest value to `val`

    Returns
    -------------
    idx_choice: int
        index for the `best matched` index to `val`
    """
    val = float(val)
    arr = num.asarray(arr)
    idx = (num.abs(arr - val)).argmin()
    idx_arr = num.where(arr == arr[idx])[0]
    try:
        if len(idx_arr) == 1:
            return idx
        else:
            return num.random.choice(idx_arr)
    except:
        raise ValueError('>> Not matches found!')
