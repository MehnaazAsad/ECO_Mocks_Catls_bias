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
__all__        =["closest_val", "survey_vol", "cos_rule",
                    "distance_diff_catl", "geometry_calc", 
                    "mock_cart_to_spherical_coords"]

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

def cos_rule(a, b, gamma):
    """
    Computes the `cosine rule` for 2 distances and one angle

    Parameters
    -----------
    a: float
        one of the sides of the triangle

    b: float
        the other side of the triangle

    gamma: float
        angle facing the side of the triangle in questions
        Units: degrees

    Returns
    -----------
    c: float
        measure of the size `c` in question
    """
    # Degree in radians
    gamma_rad = num.radians(gamma)
    # Third side of the triangle
    c = (a**2 + b**2 - (2*a*b*num.cos(gamma_rad)))**(.5)

    return c

def distance_diff_catl(ra, dist, gap):
    """
    Computes the necessary distance between catalogues

    Parameters
    -----------
    ra: float
        1st distance

    dist: float
        2nd distance

    Returns
    -----------
    dist_diff: float
        amount of distance necessary between mocks
    """
    ra = float(ra)
    dist = float(dist)
    gap = float(gap)
    ## Calculation of distance between catalogues
    dist_diff = (((ra + gap)**2 - (dist/2.)**2.)**(0.5)) - ra

    return dist_diff

def geometry_calc(dist_1, dist_2, alpha):
    """
    Computes the geometrical components to construct the catalogues in 
    a simulation box

    Parameters
    -----------
    dist_1: float
        1st distance used to determine geometrical components

    dist_2: float
        2nd distance used to determine geometrical components

    alpha: float
        angle used to determine geometrical components
        Unit: degrees

    Returns
    -----------
    h_total: float

    h1: float

    s1: float

    s2: float
    """
    assert(dist_1 <= dist_2)
    ## Calculating distances for the triangles
    s1 = cos_rule(dist_1, dist_1, alpha)
    s2 = cos_rule(dist_2, dist_2, alpha)
    ## Height
    h1 = (dist_1**2 - (s1/2.)**2)**0.5
    assert(h1 <= dist_1)
    h2      = dist_1 - h1
    h_total = h2 + (dist_2 - dist_1)

    return h_total, h1, s1, s2

def mock_cart_to_spherical_coords(cart_arr, dist):
    """
    Computes the right ascension and declination for the given 
    point in (x,y,z) position

    Parameters
    -----------
    cart_arr: numpy.ndarray, shape (3,)
        array with (x,y,z) positions

    dist: float
        dist to the point from observer's position

    Returns
    -----------
    ra_val: float
        right ascension of the point on the sky

    dec_val: float
        declination of the point on the sky
    """
    ## Reformatting coordinates
    # Cartesian coordinates
    (   x_val,
        y_val,
        z_val) = cart_arr/float(dist)
    # Distance to object
    dist = float(dist)
    ## Declination
    dec_val = 90. - num.degrees(num.arccos(z_val))
    ## Right ascension
    if x_val == 0:
        if y_val > 0.:
            ra_val = 90.
        elif y_val < 0.:
            ra_val = -90.
    else:
        ra_val = num.degrees(num.arctan(y_val/x_val))
    ##
    ## Seeing on which quadrant the point is at
    if x_val < 0.:
        ra_val += 180.
    elif (x_val >= 0.) and (y_val < 0.):
        ra_val += 360.

    return ra_val, dec_val





















