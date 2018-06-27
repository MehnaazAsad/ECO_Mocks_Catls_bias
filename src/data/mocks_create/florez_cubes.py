#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 2018-06-27
# Last Modified: 2018-06-27
# Vanderbilt University
from __future__ import absolute_import, division, print_function 
__author__     = ['Victor Calderon']
__copyright__  = ["Copyright 2018 Victor Calderon, "]
__email__      = ['victor.calderon@vanderbilt.edu']
__maintainer__ = ['Victor Calderon']
"""
Script to turn specific simulation data into Pandas Dataframes
"""
# Importing Modules
from cosmo_utils       import mock_catalogues as cm
from cosmo_utils       import utils           as cu
from cosmo_utils.utils import file_utils      as cfutils
from cosmo_utils.utils import file_readers    as cfreaders
from cosmo_utils.utils import work_paths      as cwpaths
from cosmo_utils.utils import stats_funcs     as cstats
from cosmo_utils.utils import geometry        as cgeom
from cosmo_utils.mock_catalogues import catls_utils as cmcu

import numpy as num
import math
import os
import sys
import pandas as pd
import pickle
import matplotlib
matplotlib.use( 'Agg' )
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rc('text', usetex=True)
import seaborn as sns
#sns.set()
from progressbar import (Bar, ETA, FileTransferSpeed, Percentage, ProgressBar,
                        ReverseBar, RotatingMarker)
from tqdm import tqdm

# Extra-modules
import argparse
from argparse import ArgumentParser
from argparse import HelpFormatter
from operator import attrgetter
from tqdm import tqdm

## Functions
class SortingHelpFormatter(HelpFormatter):
    def add_arguments(self, actions):
        """
        Modifier for `argparse` help parameters, that sorts them alphabetically
        """
        actions = sorted(actions, key=attrgetter('option_strings'))
        super(SortingHelpFormatter, self).add_arguments(actions)

def _str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _check_pos_val(val, val_min=0):
    """
    Checks if value is larger than `val_min`

    Parameters
    ----------
    val: int or float
        value to be evaluated by `val_min`

    val_min: float or int, optional (default = 0)
        minimum value that `val` can be

    Returns
    -------
    ival: float
        value if `val` is larger than `val_min`

    Raises
    -------
    ArgumentTypeError: Raised if `val` is NOT larger than `val_min`
    """
    ival = float(val)
    if ival <= val_min:
        msg  = '`{0}` is an invalid input!'.format(ival)
        msg += '`val` must be larger than `{0}`!!'.format(val_min)
        raise argparse.ArgumentTypeError(msg)

    return ival

def get_parser():
    """
    Get parser object for `eco_mocks_create.py` script.

    Returns
    -------
    args: 
        input arguments to the script
    """
    ## Define parser object
    description_msg = 'Description of Script'
    parser = ArgumentParser(description=description_msg,
                            formatter_class=SortingHelpFormatter,)
    ## 
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')

    ## Program message
    parser.add_argument('-progmsg',
                        dest='Prog_msg',
                        help='Program message to use throught the script',
                        type=str,
                        default=cfutils.Program_Msg(__file__))
    ## Parsing Objects
    args = parser.parse_args()

    return args

def param_vals_test(param_dict):
    """
    Checks if values are consistent with each other.

    Parameters
    -----------
    param_dict: python dictionary
        dictionary with `project` variables

    Raises
    -----------
    ValueError: Error
        This function raises a `ValueError` error if one or more of the 
        required criteria are not met
    """
    ##
    ## This is where the tests for `param_dict` input parameters go.

def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""

    # from whichcraft import which
    from shutil import which

    return which(name) is not None

def add_to_dict(param_dict):
    """
    Aggregates extra variables to dictionary

    Parameters
    ----------
    param_dict: python dictionary
        dictionary with input parameters and values

    Returns
    ----------
    param_dict: python dictionary
        dictionary with old and new values added
    """
    # This is where you define `extra` parameters for adding to `param_dict`.

    return param_dict

def directory_skeleton(param_dict, proj_dict):
    """
    Creates the directory skeleton for the current project

    Parameters
    ----------
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    Returns
    ---------
    proj_dict: python dictionary
        Dictionary with current and new paths to project directories
    """
    file_msg = param_dict['Prog_msg']
    ## Path to the simulation cubes to analyze
    sim_dir = os.path.join('/fs1/caldervf/CODES/vandy_group_statistics2',
                            'RESOLVE_Mocks_Cube_Color_Age',
                            'With_Weights',
                            'New_Mocks')
    # Checking that folder exists
    if not (os.path.exists(sim_dir)):
        msg = '{0} `sim_dir` ({1}) does not exist!'.format(file_msg, sim_dir)
        raise FileNotFoundError(msg)
    # Finding combination of the directory
    simfile = os.path.join(sim_dir,
        'Mbaryon_9.0_Vpeak_0.2_u-r_Acc_Rate_100Myr_0.6_130.0.hdf5')
    # Check that file exists
    if not (os.path.exists(simfile)):
        msg = '{0} `simfile` ({1}) does not exist!'.format(file_msg, simfile)
        raise FileNotFoundError(msg)
    #
    # Output directory
    outdir = os.path.join(proj_dict['int_dir'],
                            'florez_age_matching_results')
    cfutils.Path_Folder(outdir)
    #
    # Saving to dictionary
    proj_dict['sim_dir'] = sim_dir
    proj_dict['simfile'] = simfile
    proj_dict['outdir' ] = outdir

    return proj_dict

## -------------------- Data Extraction -------------------- ##
def simfile_data_extraction(param_dict, proj_dict, save_file=True):
    """
    Extracts the data from the simulation file and converts it to
    a Pandas DataFrame.

    Parameters
    -----------
    param_dict : `dict`
        Dictionary with input parameters and values

    proj_dict : `dict`
        Dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    Returns
    -----------
    sim_pd : `pd.DataFrame`
        DataFrame containing the necessary information of `simfile`.
    """
    file_msg = param_dict['Prog_msg']
    # Output filepath
    filename = os.path.join(proj_dict['outdir'],
                    os.path.basename(proj_dict['simfile']))
    # Constatns
    cens = int(1)
    sats = int(0)
    failval = -1
    # Columns to be extracted
    sim_cols = ['Mbaryon', 'Vpeak', 'mvir', 'Acc_Rate_100Myr', 'u-r',
                'FSMGR', 'pid', 'id', 'upid']
    # Reading in filename and converting it to Pandas DataFrame
    sim_pd_tot = cfreaders.read_hdf5_file_to_pandas_DF(proj_dict['simfile'])
    # Only selecting certaing columns
    sim_pd = sim_pd_tot.loc[:, sim_cols]
    #
    # -- Figuring out host halo's mass
    mvir_arr     = sim_pd['mvir'].values
    galid_arr    = sim_pd['id']
    gal_upid_arr = sim_pd['upid']
    # Defining new array
    halo_m = [[] for x in range(len(sim_pd))]
    # Looping over galaxies
    for gal in tqdm(range(len(sim_pd))):
        if (gal_upid_arr[gal] == -1):
            mhalo_gal = float(mvir_arr[gal])
        else:
            try:
                upid_gal = gal_upid_arr[gal]
                mhalo_gal = float(mvir_arr[num.where(galid_arr == upid_gal)[0]])
            except:
                mhalo_gal = failval
        #
        # Saving into array
        halo_m[gal] = mhalo_gal
    #
    # Saving to the main DataFrame
    sim_pd.loc[:, 'mhalo_host'] = halo_m
    #
    # Deleting columns
    sim_pd.drop(['mvir', 'id', 'upid', 'pid'], inplace=True, axis=1)
    #
    # Saving file if necessary
    if save_file:
        cfreaders.pandas_df_to_hdf5_file(sim_pd, filename, key='sim_data')
        cfutils.File_Exists(filename)

    return sim_pd

# Saving file to output file

## -------------------- Main Function -------------------- ##

def main(args):
    """
    Script to turn specific simulation data into Pandas Dataframes
    """
    ## Reading all elements and converting to python dictionary
    param_dict = vars(args)
    ## Checking for correct input
    param_vals_test(param_dict)
    ## Adding extra variables
    param_dict = add_to_dict(param_dict)
    ## Program message
    Prog_msg = param_dict['Prog_msg']
    ##
    ## Creating Folder Structure
    # proj_dict  = directory_skeleton(param_dict, cwpaths.cookiecutter_paths(__file__))
    proj_dict  = directory_skeleton(param_dict, cwpaths.cookiecutter_paths('./'))
    ##
    ## Printing out project variables
    print('\n'+50*'='+'\n')
    for key, key_val in sorted(param_dict.items()):
        if key !='Prog_msg':
            print('{0} `{1}`: {2}'.format(Prog_msg, key, key_val))
    print('\n'+50*'='+'\n')
    #
    # Extracting info from `simfile`
    sim_pd = simfile_data_extraction(param_dict, proj_dict)

# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)
