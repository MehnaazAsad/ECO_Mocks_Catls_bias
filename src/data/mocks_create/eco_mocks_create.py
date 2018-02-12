#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : DATE
# Last Modified: DATE
# Vanderbilt University
from __future__ import print_function, division, absolute_import
__author__     =['Victor Calderon']
__copyright__  =["Copyright 2017 Victor Calderon, "]
__email__      =['victor.calderon@vanderbilt.edu']
__maintainer__ =['Victor Calderon']
"""

"""
# Path to Custom Utilities folder
import os
import sys
import git
from path_variables import git_root_dir
sys.path.insert(0, os.path.realpath(git_root_dir(__file__)))

# Importing Modules
import src.data.utilities_python as cu
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
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
import seaborn as sns
#sns.set()
from progressbar import (Bar, ETA, FileTransferSpeed, Percentage, ProgressBar,
                        ReverseBar, RotatingMarker)

# Extra-modules
from argparse import ArgumentParser
from argparse import HelpFormatter
from operator import attrgetter

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

def is_valid_file(parser, arg):
    """
    Check if arg is a valid file that already exists on the file system.

    Parameters
    ----------
    parser : argparse object
    arg : str

    Returns
    -------
    arg
    """
    arg = os.path.abspath(arg)
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!\n" % arg)
    else:
        return arg

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
    ##  Version
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    ## Variables
    # Size of the cube
    parser.add_argument('-sizecube',
                        dest='size_cube',
                        help='Length of simulation cube in Mpc/h',
                        type=float,
                        default=180.)
    # Median Redshift
    parser.add_argument('-zmed',
                        dest='zmedian',
                        help='Median Redshift of the survey',
                        type=float,
                        default=0.)
    # Minimum velocity of the survey
    parser.add_argument('-czmin',
                        dest='czmin',
                        help='Minimum velocity of the survey (c times zmin)',
                        type=float,
                        default=2532.)
    # Maximum velocity of the survey
    parser.add_argument('-czmax',
                        dest='czmax',
                        help='Maximum velocity of the survey (c times zmax)',
                        type=float,
                        default=7470.)
    # Type of survey
    parser.add_argument('-survey',
                        dest='survey',
                        help='Type of survey to produce. Choices: A, B, ECO',
                        type=str,
                        choices=['A','B','ECO'],
                        default='ECO')
    # Halobias file
    parser.add_argument('-hbfile',
                        dest='hbfile',
                        help='Path to the Halobias file in `.ff` format',
                        type=lambda x: is_valid_file(parser, x))#,
                        # required=True)
    # Cosmology used for the project
    parser.add_argument('-cosmo',
                        dest='cosmo_choice',
                        help='Cosmology to use. Options: 1) Planck, 2) LasDamas',
                        type=str,
                        default='Planck',
                        choices=['Planck','LasDamas'])
    # Halomass function
    parser.add_argument('-hmf',
                        dest='hmf_choice',
                        help='Halo Mass Function choice',
                        type=str,
                        default='warren',
                        choices=['warren','tinker08'])
    ## Random Seed
    parser.add_argument('-seed',
                        dest='seed',
                        help='Random seed to be used for the analysis',
                        type=int,
                        metavar='[0-4294967295]',
                        default=1)
    ## Option for removing file
    parser.add_argument('-remove',
                        dest='remove_files',
                        help='Delete pickle file containing pair counts',
                        type=_str2bool,
                        default=False)
    ## Program message
    parser.add_argument('-progmsg',
                        dest='Prog_msg',
                        help='Program message to use throught the script',
                        type=str,
                        default=cu.Program_Msg(__file__))
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

def survey_specs(param_dict):
    """
    Provides the specifications of the survey being created

    Parameters
    ----------
    param_dict: python dictionary
        dictionary with `project` variables

    Returns
    ----------
    param_dict: python dictionary
        dictionary with the 'updated' project variables
    """
    if param_dict['servey'] == 'A':
        czmin      = 2532.
        czmax      = 7470.
        survey_vol = 20957.7789388
    elif param_dict['survey'] == 'B':
        czmin      = 4250.
        czmax      = 7250.
        survey_vol = 15908.063125
    elif param_dict['survey'] == 'ECO':
        czmin      = 2532.
        czmax      = 7470.
        survey_vol = 192294.221932

    ## Saving to `param_dict`
    param_dict['czmin'     ] = czmin
    param_dict['czmax'     ] = czmax
    param_dict['survey_vol'] = survey_vol

    return param_dict

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
    # Raw directory
    raw_dir      = os.path.join( proj_dict['data_dir'],
                                'raw')
    # Halobias Files
    hb_files_dir = os.path.join(    raw_dir,
                                    'hb_files',
                                    param_dict['survey'])
    # Cosmological files
    cosmo_dir = os.path.join(   raw_dir,
                                param_dict['cosmo_choice'],
                                'cosmo_files')
    # Interim directory
    int_dir      = os.path.join( proj_dict['data_dir'],
                                'interim')
    # Mass Function
    mf_dir       = os.path.join(int_dir,
                                'MF',
                                param_dict['survey'])
    # Conditional Luminosity Function (CLF)
    clf_dir      = os.path.join(int_dir,
                                'CLF_HB',
                                param_dict['survey'] + '/')
    # Halo Ngal
    h_ngal_dir   = os.path.join(int_dir,
                                'HALO_NGAL_CLF',
                                param_dict['survey'] + '/')
    ## Catalogues
    catl_outdir = os.path.join( proj_dict['data_dir'],
                                'processed',
                                param_dict['survey'])
    ## Creating output folders for the catalogues
    mock_cat_mgc     = os.path.join(catl_outdir, 'galaxy_catalogues')
    mock_cat_mc      = os.path.join(catl_outdir, 'member_galaxy_catalogues')
    mock_cat_gc      = os.path.join(catl_outdir, 'group_galaxy_catalogues' )
    mock_cat_mc_perf = os.path.join(catl_outdir, 'perfect_member_galaxy_catalogues')
    mock_cat_gc_perf = os.path.join(catl_outdir, 'perfect_group_galaxy_catalogues' )
    ##
    ## Creating Directories
    cu.Path_Folder(cosmo_dir)
    cu.Path_Folder(catl_outdir)
    cu.Path_Folder(mock_cat_mgc)
    cu.Path_Folder(mock_cat_mc)
    cu.Path_Folder(mock_cat_gc)
    cu.Path_Folder(mock_cat_mc_perf)
    cu.Path_Folder(mock_cat_gc_perf)
    cu.Path_Folder(int_dir)
    cu.Path_Folder(mf_dir)
    cu.Path_Folder(clf_dir)
    cu.Path_Folder(h_ngal_dir)
    cu.Path_Folder(raw_dir)
    cu.Path_Folder(hb_files_dir)
    ##
    ## Adding to `proj_dict`
    proj_dict['cosmo_dir'       ] = cosmo_dir
    proj_dict['catl_outdir'     ] = catl_outdir
    proj_dict['mock_cat_mgc'    ] = mock_cat_mgc
    proj_dict['mock_cat_mc'     ] = mock_cat_mc
    proj_dict['mock_cat_gc'     ] = mock_cat_gc
    proj_dict['mock_cat_mc_perf'] = mock_cat_mc_perf
    proj_dict['mock_cat_gc_perf'] = mock_cat_gc_perf
    proj_dict['int_dir'         ] = int_dir
    proj_dict['mf_dir'          ] = mf_dir
    proj_dict['clf_dir'         ] = clf_dir
    proj_dict['h_ngal_dir'      ] = h_ngal_dir
    proj_dict['raw_dir'         ] = raw_dir
    proj_dict['hb_files_dir'    ] = hb_files_dir

    return proj_dict




    return proj_dict


def main(args):
    """
    Creates set of mock catalogues for ECO, Resolve A, and Resolve B surveys.
    """
    ## Reading all elements and converting to python dictionary
    param_dict = vars(args)
    ## Checking for correct input
    param_vals_test(param_dict)
    ## Program message
    Prog_msg = param_dict['Prog_msg']
    ## Adding additonal parameters
    param_dict = add_to_dict(param_dict)
    ##
    ## Creating Folder Structure
    # proj_dict  = directory_skeleton(param_dict, cu.cookiecutter_paths(__file__))
    proj_dict  = directory_skeleton(param_dict, cu.cookiecutter_paths('./'))
    ##
    ## Printing out project variables
    print('\n'+50*'='+'\n')
    for key, key_val in sorted(param_dict.items()):
        if key !='Prog_msg':
            print('{0} `{1}`: {2}'.format(Prog_msg, key, key_val))
    print('\n'+50*'='+'\n')


# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)
