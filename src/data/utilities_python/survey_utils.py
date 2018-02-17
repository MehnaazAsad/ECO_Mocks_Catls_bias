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
__all__        =["closest_val", "survey_vol"]

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

def survey_vol(ra_arr, dec_arr, rho_arr):
    """
    Computes the volume of a "sphere" with given limits for 
    ra, dec, and distance

    Parameters
    ----------
    ra_arr: list or numpy.ndarray, shape (N,2)
        array with initial and final right ascension coordinates
        Unit: degrees

    dec_arr: list or numpy.ndarray, shape (N,2)
        array with initial and final declination coordinates
        Unit: degrees

    rho_arr: list or numpy.ndarray, shape (N,2)
        arrray with initial and final distance
        Unit: distance units
    
    Returns
    ----------
    survey_vol: float
        volume of the survey being analyzed
        Unit: distance**(3)
    """
    # Right ascension - Radians  theta coordinate
    theta_min_rad, theta_max_rad = num.radians(num.array(ra_arr))
    # Declination - Radians - phi coordinate
    phi_min_rad, phi_max_rad = num.radians(90.-num.array(dec_arr))[::-1]
    # Distance
    rho_min, rho_max = rho_arr
    # Calculating volume
    vol  = (1./3.)*(num.cos(phi_min_rad)-num.cos(phi_max_rad))
    vol *= (theta_max_rad) - (theta_min_rad)
    vol *= (rho_max**3) - rho_min**3
    vol  = num.abs(vol)

    return vol
