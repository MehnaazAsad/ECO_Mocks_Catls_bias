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
import hmf
import astropy.cosmology as astrocosmo
import astropy.constants as ac
import astropy.units     as u
import astropy.table     as astro_table
import requests
from collections import Counter
import subprocess
from tqdm import tqdm
from scipy.io.idl import readsav
from astropy.table import Table
from astropy.io import fits
import copy


## Functions

## -----------| Reading input arguments |----------- ##

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

def url_checker(url_str):
    """
    Checks if the `url_str` is a valid URL

    Parameters
    ----------
    url_str: string
        url of the website to probe
    """
    request = requests.get(url_str)
    if request.status_code != 200:
        msg = '`url_str` ({0}) does not exist'.format(url_str)
        raise ValueError(msg)
    else:
        pass

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
    # Halo definition
    parser.add_argument('-halotype',
                        dest='halotype',
                        help='Type of halo definition.',
                        type=str,
                        choices=['mvir','m200'],
                        default='mvir')
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
                        dest='hmf_model',
                        help='Halo Mass Function choice',
                        type=str,
                        default='warren',
                        choices=['warren','tinker08'])
    ## CLF Choice
    parser.add_argument('-clf',
                        dest='clf_type',
                        help='Type of CLF to choose. 1) Cacciato, 2) LasDamas Best-fit',
                        type=int,
                        choices=[1,2],
                        default=2)
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
    ##
    ## Size of the cube
    assert(param_dict['size_cube'] == 180.)

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
    ## Central/Satellite designations
    cens = int(1)
    sats = int(0)
    ## HOD Parameters
    hod_dict             = {}
    hod_dict['logMmin' ] = 11.4
    hod_dict['sigLogM' ] = 0.2
    hod_dict['logM0'   ] = 10.8
    hod_dict['logM1'   ] = 12.8
    hod_dict['alpha'   ] = 1.05
    hod_dict['zmed_val'] = 'z0p000'
    hod_dict['znow'    ] = 0
    ## Choice of Survey
    choice_survey = 2
    ###
    ### URL to download catalogues
    url_catl = 'http://lss.phy.vanderbilt.edu/groups/data_eco_vc/'
    url_checker(url_catl)
    ##
    ## Adding to `param_dict`
    param_dict['cens'         ] = cens
    param_dict['sats'         ] = sats
    param_dict['url_catl'     ] = url_catl
    param_dict['hod_dict'     ] = hod_dict
    param_dict['choice_survey'] = choice_survey

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
    raw_dir      = os.path.join(proj_dict['data_dir'],
                                'raw')
    # Halobias Files
    hb_files_dir = os.path.join(raw_dir,
                                'hb_files',
                                param_dict['halotype'],
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
                                param_dict['halotype'],
                                param_dict['survey'] + '/')
    # Halo Ngal
    h_ngal_dir   = os.path.join(int_dir,
                                'HALO_NGAL_CLF',
                                param_dict['halotype'],
                                param_dict['survey'] + '/')
    ## Catalogues
    catl_outdir = os.path.join( proj_dict['data_dir'],
                                'processed',
                                param_dict['halotype'],
                                param_dict['survey'])
    ## Photometry files
    phot_dir    = os.path.join( raw_dir,
                                'surveys_phot_files')
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
    cu.Path_Folder(phot_dir)
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
    proj_dict['phot_dir'        ] = phot_dir

    return proj_dict

## -----------| Survey-related functions |----------- ##

def cosmo_create(cosmo_choice='Planck', H0=100., Om0=0.25, Ob0=0.04,
    Tcmb0=2.7255):
    """
    Creates instance of the cosmology used throughout the project.

    Parameters
    ----------
    cosmo_choice: string, optional (default = 'Planck')
        choice of cosmology
        Options:
            - Planck: Cosmology from Planck 2015
            - LasDamas: Cosmology from LasDamas simulation

    h: float, optional (default = 1.0)
        value for small cosmological 'h'.

    Returns
    ----------                  
    cosmo_obj: astropy cosmology object
        cosmology used throughout the project
    """
    ## Checking cosmology choices
    cosmo_choice_arr = ['Planck', 'LasDamas']
    assert(cosmo_choice in cosmo_choice_arr)
    ## Choosing cosmology
    if cosmo_choice == 'Planck':
        cosmo_model = astrocosmo.Planck15.clone(H0=H0)
    elif cosmo_choice == 'LasDamas':
        cosmo_model = astrocosmo.FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0, 
            Tcmb0=Tcmb0)
    ## Cosmo Paramters
    cosmo_params         = {}
    cosmo_params['H0'  ] = cosmo_model.H0.value
    cosmo_params['Om0' ] = cosmo_model.Om0
    cosmo_params['Ob0' ] = cosmo_model.Ob0
    cosmo_params['Ode0'] = cosmo_model.Ode0
    cosmo_params['Ok0' ] = cosmo_model.Ok0
    ## HMF Cosmo Model
    cosmo_hmf = hmf.cosmo.Cosmology(cosmo_model=cosmo_model)

    return cosmo_model, cosmo_hmf

def hmf_calc(cosmo_model, proj_dict, param_dict, Mmin=10, Mmax=16, 
    dlog10m=1e-3, hmf_model='warren', ext='csv', sep=',', 
    Prog_msg='1 >>   '):
    # Prog_msg=cu.Program_Msg(__file__)):
    """
    Creates file with the desired mass function

    Parameters
    ----------
    cosmo_model: astropy cosmology object
        cosmology used throughout the project

    hmf_out: string
        path to the output file for the halo mass function

    Mmin: float, optional (default = 10)
        minimum halo mass to evaluate

    Mmax: float, optional (default = 15)
        maximum halo mass to evaluate

    dlog10m: float, optional (default = 1e-2)


    hmf_model: string, optional (default = 'warren')
        Halo Mass Function choice
        Options:
            - 'warren': Uses Warren et al. (2006) HMF
            = 'tinker08': Uses Tinker et al. (2008) HMF

    ext: string, optional (default = 'csv')
        extension of output file

    sep: string, optional (default = ',')
        delimiter used for reading/writing the file

    Returns
    ----------
    hmf_pd: pandas DataFrame
        DataFrame of `log10 masses` and `cumulative number densities` for 
        halos of mass > M.
    """
    ## HMF Output file
    hmf_outfile = os.path.join( proj_dict['mf_dir'],
                                '{0}_H0_{1}_HMF_{2}.{3}'.format(
                                    param_dict['cosmo_choice'],
                                    cosmo_model.H0.value,
                                    hmf_model,
                                    ext))
    if os.path.exists(hmf_outfile):
        # Removing file
        os.remove(hmf_outfile)
    ## Halo mass function - Fitting function
    if hmf_model == 'warren':
        hmf_choice_fit = hmf.fitting_functions.Warren
    elif hmf_model == 'tinker08':
        hmf_choice_fit = hmf.fitting_functions.Tinker08
    else:
        msg = '{0} hmf_model `{1}` not supported! Exiting'.format(
            Prog_msg, hmf_model)
        raise ValueError(msg)
    # Calculating HMF
    mass_func = hmf.MassFunction(Mmin=Mmin, Mmax=Mmax, dlog10m=dlog10m,
        cosmo_model=cosmo_model, hmf_model=hmf_choice_fit)
    ## Log10(Mass) and cumulative number density of haloes
    # HMF Pandas DataFrame
    hmf_pd = pd.DataFrame({ 'logM':num.log10(mass_func.m), 
                            'ngtm':mass_func.ngtm})
    # Saving to output file
    hmf_pd.to_csv(hmf_outfile, sep=sep, index=False,
        columns=['logM','ngtm'])

    return hmf_pd

def download_files(param_dict, proj_dict):
    """
    Downloads the required files to a specific set of directories
    
    Parameters
    ------------
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    Returns
    ------------
    param_dict: python dictionary
        dictionary with the updated paths to the local files that were 
        just downloaded
    """
    ## Main ECO files directory - Web
    resolve_web_dir = os.path.join( param_dict['url_catl'],
                                    'RESOLVE',
                                    'resolve_files')
    eco_web_dir     = os.path.join( param_dict['url_catl'],
                                    'ECO',
                                    'eco_files')
    ## ECO - MHI Predictions
    mhi_file_local = os.path.join(  proj_dict['phot_dir'],
                                    'eco_mhi_prediction.txt')
    mhi_file_web   = os.path.join(  resolve_web_dir,
                                    'eco_wresa_050815_Predicted_HI.txt')
    ## ECO - Photometry file
    eco_phot_file_local = os.path.join( proj_dict['phot_dir'],
                                        'eco_wresa_050815.dat')
    eco_phot_file_web   = os.path.join( resolve_web_dir,
                                        'eco_wresa_050815.dat')
    ## Resolve-B Photometry
    res_b_phot_file_local = os.path.join(proj_dict['phot_dir'],
                                        'resolvecatalog_str_2015_07_12.fits')
    res_b_phot_file_web   = os.path.join(resolve_web_dir,
                                        'resolvecatalog_str_2015_07_12.fits')
    ## ECO-Resolve B Luminosity Function
    eco_LF_file_local = os.path.join(   proj_dict['phot_dir'],
                                        'eco_resolve_LF.csv')
    eco_LF_file_web   = os.path.join(   resolve_web_dir,
                                        'ECO_ResolveB_Lum_Function.csv')
    ## ECO Luminosities
    eco_lum_file_local = os.path.join(  proj_dict['phot_dir'],
                                        'eco_dens_mag.csv')
    eco_lum_file_web   = os.path.join(  eco_web_dir,
                                        'Dens_Mag_Interp_ECO.ascii')
    ## Halobias file
    hb_file_local      = os.path.join(  proj_dict['hb_files_dir'],
                                        'Resolve_plk_5001_so_{0}_hod1.ff'.format(
                                            param_dict['halotype']))
    hb_file_web        = os.path.join(  param_dict['url_catl'],
                                        'HB_files',
                                        param_dict['halotype'],
                                        'Resolve_plk_5001_so_{0}_hod1.ff'.format(
                                            param_dict['halotype']))
    ##
    ## Downloading files
    files_local_arr = [ mhi_file_local       , eco_phot_file_local,
                        res_b_phot_file_local, eco_LF_file_local  , 
                        eco_lum_file_local   , hb_file_local      ]
    files_web_arr   = [ mhi_file_web         , eco_phot_file_web  ,
                        res_b_phot_file_web  , eco_LF_file_web    ,
                        eco_lum_file_web     , hb_file_web        ]
    ## Checking that files exist
    for (local_ii, web_ii) in zip(files_local_arr,files_web_arr):
        ## Downloading file if not in `local`
        if not os.path.exists(local_ii):
            ## Checking for `web` file
            url_checker(web_ii)
            ## Downloading
            cu.File_Download_needed(local_ii, web_ii)
            assert(os.path.exists(local_ii))
    ##
    ## Saving paths to `param_dict'
    files_dict = {}
    files_dict['mhi_file_local'       ] = mhi_file_local
    files_dict['eco_phot_file_local'  ] = eco_phot_file_local
    files_dict['res_b_phot_file_local'] = res_b_phot_file_local
    files_dict['eco_LF_file_local'    ] = eco_LF_file_local
    files_dict['eco_lum_file_local'   ] = eco_lum_file_local
    files_dict['hb_file_local'        ] = hb_file_local
    # To `param_dict'
    param_dict['files_dict'] = files_dict
    ##
    ## Showing stored files
    print('{0} Downloaded all necessary files!'.format(param_dict['Prog_msg']))

    return param_dict

def z_comoving_calc(param_dict, proj_dict, cosmo_model, 
    zmin=0, zmax=0.5, dz=1e-3, ext='csv', sep=','):
    """
    Computes the comoving distance of an object based on its redshift
    
    Parameters
    ------------
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    cosmo_model: astropy cosmology object
        cosmology used throughout the project

    Returns
    ------------
    z_como_pd: pandas DataFrame
        DataFrame with `z, d_comoving` in units of Mpc
    """
    ## File
    z_comoving_file = os.path.join( proj_dict['cosmo_dir'],
                                    '{0}_H0_{1}_z_comoving.{2}'.format(
                                        param_dict['cosmo_choice'],
                                        cosmo_model.H0.value,
                                        ext))
    if (os.path.exists(z_comoving_file)) and (param_dict['remove_files']):
        ## Removing file
        os.remove(z_comoving_file)
    if not os.path.exists(z_comoving_file):
        ## Calculating comoving distance
        # `z_arr`     : Unitless
        # `d_comoving`:
        z_arr     = num.arange(zmin, zmax, dz)
        d_como    = cosmo_model.comoving_distance(z_arr).to(u.Mpc).value
        z_como_pd = pd.DataFrame({'z':z_arr, 'd_como':d_como})
        ## Saving to file
        z_como_pd.to_csv(z_comoving_file, sep=sep, index=False)
        cu.File_Exists(z_comoving_file)
    else:
        z_como_pd = pd.read_csv(z_comoving_file, sep=sep)

    return z_como_pd

## -----------| Survey-related functions |----------- ##

def survey_specs(param_dict, cosmo_model):
    """
    Provides the specifications of the survey being created

    Parameters
    ----------
    param_dict: python dictionary
        dictionary with `project` variables

    cosmo_model: astropy cosmology object
        cosmology used throughout the project

    Returns
    ----------
    param_dict: python dictionary
        dictionary with the 'updated' project variables
    """
    ## Redshift, volumen and r-mag limit for each survey
    if param_dict['survey'] == 'A':
        czmin      = 2532.
        czmax      = 7470.
        # survey_vol = 20957.7789388
        mr_limit   = -17.33
    elif param_dict['survey'] == 'B':
        czmin      = 4250.
        czmax      = 7250.
        # survey_vol = 15908.063125
        mr_limit   = -17.00
    elif param_dict['survey'] == 'ECO':
        czmin      = 2532.
        czmax      = 7470.
        # survey_vol = 192294.221932
        mr_limit   = -17.33
    ##
    ## Right Ascension and Declination coordinates for each survey
    if param_dict['survey'] == 'A':
        ra_min_real = 131.25
        ra_max_real = 236.25
        dec_min     = 0.
        dec_max     = 5.
        # Extras
        dec_range   = dec_max - dec_min
        ra_range    = ra_max_real - ra_min_real
        ra_min      = (180. - ra_range)/2.
        ra_max      = ra_min + ra_range
        ra_diff     = ra_max_real - ra_max
        # Assert statements
        assert(dec_min < dec_max)
        assert(ra_range >= 0)
        assert(ra_min < ra_max)
        assert(ra_min_real < ra_max_real)
    elif param_dict['survey'] == 'B':
        ra_min_real = 330.
        ra_max_real = 45.
        dec_min     = -1.25
        dec_max     = 1.25
        # Extras
        dec_range   = dec_max - dec_min
        ra_max_real - (ra_min_real - 360.)
        ra_min      = (180. - ra_range)/2.
        ra_max      = ra_min + ra_range
        ra_diff     = ra_max_real - ra_max
        # Assert statements
        assert(dec_min < dec_max)
        assert(ra_range >= 0)
        assert(ra_min < ra_max)
    elif param_dict['survey'] == 'ECO':
        ra_min_real = 130.05
        ra_max_real = 237.45
        dec_min     = -1
        dec_max     = 49.85
        # Extras
        dec_range   = dec_max - dec_min
        ra_range    = ra_max_real - ra_min_real
        ra_min      = (180. - ra_range)/2.
        ra_max      = ra_min + ra_range
        ra_diff     = ra_max_real - ra_max
        # Assert statements
        assert(dec_min < dec_max)
        assert(ra_range >= 0)
        assert(ra_min < ra_max)
        assert(ra_min_real < ra_max_real)
    ## Survey volume
    km_s       = u.km/u.s
    z_arr      = (num.array([czmin, czmax])*km_s/(ac.c.to(km_s))).value
    z_arr      = (num.array([czmin, czmax])*km_s/(3e5*km_s)).value
    r_arr      = cosmo_model.comoving_distance(z_arr).to(u.Mpc).value
    survey_vol = cu.survey_vol( [0, ra_range],
                                [dec_min, dec_max],
                                cosmo_model.comoving_distance(z_arr).value)
    ##
    ## Survey height, and other geometrical factors
    (   h_total,
        s1_top ,
        s2     ,
        h1     ) = cu.geometry_calc(r_arr[0], r_arr[1], ra_range)
    (   h_side ,
        s1_side,
        d_th   ,
        h2     ) = cu.geometry_calc(r_arr[0], r_arr[1], dec_range)

    ##
    # ra_dec dictionary
    coord_dict = {}
    coord_dict['ra_min_real'] = ra_min_real
    coord_dict['ra_max_real'] = ra_max_real
    coord_dict['dec_min'    ] = dec_min
    coord_dict['dec_max'    ] = dec_max
    coord_dict['dec_range'  ] = dec_range
    coord_dict['ra_range'   ] = ra_range
    coord_dict['ra_min'     ] = ra_min
    coord_dict['ra_max'     ] = ra_max
    coord_dict['ra_diff'    ] = ra_diff
    # Height and other geometrical objects
    coord_dict['h_total'    ] = h_total
    coord_dict['s1_top'     ] = s1_top
    coord_dict['s2'         ] = s2
    coord_dict['h1'         ] = h1
    coord_dict['h_side'     ] = h_side
    coord_dict['s1_side'    ] = s1_side
    coord_dict['d_th'       ] = d_th
    coord_dict['h2'         ] = h2
    ##
    ## Resolve-B Mr limit
    mr_eco   = -17.33
    mr_res_b = -17.00
    ## Saving to `param_dict`
    param_dict['czmin'     ] = czmin
    param_dict['czmax'     ] = czmax
    param_dict['survey_vol'] = survey_vol
    param_dict['mr_limit'  ] = mr_limit
    param_dict['mr_eco'    ] = mr_eco
    param_dict['mr_res_b'  ] = mr_res_b
    param_dict['coord_dict'] = coord_dict

    return param_dict

def eco_geometry_mocks(clf_pd, param_dict, proj_dict):
    """
    Carves out the geometry of the `ECO` survey and produces set 
    of mock catalogues
    
    Parameters
    -------------
    clf_pd: pandas DataFrame
        DataFrame containing information from Halobias + CLF procedures
    
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.
    """
    ## Coordinates dictionary
    coord_dict = param_dict['coord_dict']
    ###### ----- X-Y Upper Left Mocks  -----######
    clf_ul_pd = copy.deepcopy(clf_pd)
    # Coordinates
    ra_min_ul  = 90. - coord_dict['ra_range']
    ra_max_ul  = 90.
    ra_diff_ul = coord_dict['ra_max_real'] - ra_max_ul
    gap_ul   = 20.
    x_init_ul = 0.0 + 10.
    y_init_ul = param_dict['size_cube'] - coord_dict['ra_max'] - 5.
    z_init_ul = 10.
    # z_delta_ul = gap_ul + 
    ###### ----- X-Y Upper Right Mocks -----######
    clf_ur_pd = copy.deepcopy(clf_pd)
    ###### ----- X-Y Lower Left Mocks  -----######
    clf_ll_pd = copy.deepcopy(clf_pd)
    ###### ----- X-Y Lower Right Mocks -----######
    clf_lr_pd = copy.deepcopy(clf_pd)


## -----------| Halobias-related functions |----------- ##

def hb_file_construction_extras(param_dict, proj_dict):
    """
    Parameters
    ----------
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    Returns
    ---------
    param_dict: python dictionary
        dictionary with updated 'hb_file_mod' key, which is the 
        path to the file with number of galaxies per halo ID.

    hb_pd: pandas DataFrame
        pandas DataFrame with information from the Halobias file `hb_local`
    """
    Prog_msg = param_dict['Prog_msg']
    ## Local halobias file
    hb_local = param_dict['files_dict']['hb_file_local']
    ## HaloID extras file
    hb_file_mod = os.path.join(proj_dict['hb_files_dir'],
                                    os.path.basename(hb_local)+'.mod')
    ## Reading in file
    print('{0} Reading in `hb_file`...'.format(Prog_msg))
    with open(hb_local,'rb') as hb:
        idat    = cu.fast_food_reader('int'  , 5, hb)
        fdat    = cu.fast_food_reader('float', 9, hb)
        znow    = cu.fast_food_reader('float', 1, hb)[0]
        ngal    = int(idat[1])
        lbox    = int(fdat[0])
        x_arr   = cu.fast_food_reader('float' , ngal, hb)
        y_arr   = cu.fast_food_reader('float' , ngal, hb)
        z_arr   = cu.fast_food_reader('float' , ngal, hb)
        vx_arr  = cu.fast_food_reader('float' , ngal, hb)
        vy_arr  = cu.fast_food_reader('float' , ngal, hb)
        vz_arr  = cu.fast_food_reader('float' , ngal, hb)
        halom   = cu.fast_food_reader('float' , ngal, hb)
        cs_flag = cu.fast_food_reader('int'   , ngal, hb)
        haloid  = cu.fast_food_reader('int'   , ngal, hb)
    ##
    ## Counter of HaloIDs
    haloid_counts = Counter(haloid)
    # Array of `gals` in each `haloid`
    haloid_ngal = [[] for x in range(ngal)]
    for kk, halo_kk in enumerate(haloid):
        haloid_ngal[kk] = haloid_counts[halo_kk]
    haloid_ngal = num.asarray(haloid_ngal).astype(int)
    ## Converting to Pandas DataFrame
    # Dictionary
    hb_dict = {}
    hb_dict['x'          ] = x_arr
    hb_dict['y'          ] = y_arr
    hb_dict['z'          ] = z_arr
    hb_dict['vx'         ] = vx_arr
    hb_dict['vy'         ] = vy_arr
    hb_dict['vz'         ] = vz_arr
    hb_dict['halom'      ] = halom
    hb_dict['loghalom'   ] = num.log10(halom)
    hb_dict['cs_flag'    ] = cs_flag
    hb_dict['haloid'     ] = haloid
    hb_dict['haloid_ngal'] = haloid_ngal
    # To DataFrame
    hb_cols = ['x','y','z','vx','vy','vz','halom','loghalom',
                'cs_flag','haloid','haloid_ngal']
    hb_pd   = pd.DataFrame(hb_dict)[hb_cols]
    ## Saving to file
    hb_pd.to_csv(hb_file_mod, sep=" ", columns=hb_cols, 
        index=False, header=False)
    ## Assigning to `param_dict`
    param_dict['hb_file_mod'] = hb_file_mod
    param_dict['hb_cols'    ] = hb_cols
    ## Testing `lbox`
    try:
        assert(lbox==param_dict['size_cube'])
    except:
        msg = '{0} `lbox` ({1}) does not match `size_cube` ({2})!'.format(
            Prog_msg, lbox, param_dict['size_cube'])
        raise ValueError(msg)
    # Message
    print('\n{0} Halo_ngal file: {1}'.format(Prog_msg, hb_file_mod))
    print('{0} Creating file with Ngals in each halo ... Complete'.format(Prog_msg))

    return param_dict, hb_pd

def clf_assignment(param_dict, proj_dict, choice_survey=2):
    """
    Computes the conditional luminosity function on the halobias file
    
    Parameters
    ----------
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    Returns
    ----------
    clf_pd: pandas DataFrame
        DataFrame with information from the CLF process
        Format:
            - x, y, z, vx, vy, vz: Positions and velocity components
            - log(Mhalo): log-base 10 of the DM halo mass
            - `cs_flag`: Central (1) / Satellite (0) designation
            - `haloid`: ID of the galaxy's host DM halo
            - `halo_ngal`: Total number of galaxies in the DM halo
            - `M_r`: r-band absolute magnitude from ECO via abundance matching
            - `galid`: Galaxy ID
    """
    Prog_msg = param_dict['Prog_msg']
    ## Local halobias file
    hb_local = param_dict['files_dict']['hb_file_local']
    ## CLF Output file - ASCII and FF
    hb_clf_out = os.path.join(  proj_dict['clf_dir'],
                                os.path.basename(hb_local) +'.clf')
    hb_clf_out_ff = hb_clf_out + '.ff'
    ## HOD dictionary
    hod_dict = param_dict['hod_dict']
    ## CLF Executable
    clf_exe = os.path.join( cu.get_code_c(),
                            'CLF',
                            'CLF_with_ftread')
    cu.File_Exists(clf_exe)
    ## CLF Commands - Executing in the terminal commands
    cmd_arr = [ clf_exe,
                hod_dict['logMmin'],
                param_dict['clf_type'],
                param_dict['hb_file_mod'],
                param_dict['size_cube'],
                param_dict['hod_dict']['znow'],
                hb_clf_out_ff,
                choice_survey,
                param_dict['files_dict']['eco_lum_file_local'],
                hb_clf_out]
    cmd_str = '{0} {1} {2} {3} {4} {5} {6} {7} < {8} > {9}'
    cmd     = cmd_str.format(*cmd_arr)
    print(cmd)
    subprocess.call(cmd, shell=True)
    ##
    ## Reading in CLF file
    clf_cols = ['x','y','z','vx','vy','vz',
                'loghalom','cs_flag','haloid','halo_ngal','M_r','galid']
    clf_pd   = pd.read_csv(hb_clf_out, sep='\s+', header=None, names=clf_cols)
    clf_pd.loc[:,'galid'] = clf_pd['galid'].astype(int)
    ##
    ## Remove extra files
    os.remove(hb_clf_out_ff)

    return clf_pd

def cen_sat_distance_calc(clf_pd, param_dict):
    """
    Computes the distance between the central and its satellites in a given 
    DM halo

    Parameters
    ----------
    clf_pd: pandas DataFrame
        DataFrame with information from the CLF process
            - x, y, z, vx, vy, vz: Positions and velocity components
            - log(Mhalo): log-base 10 of the DM halo mass
            - `cs_flag`: Central (1) / Satellite (0) designation
            - `haloid`: ID of the galaxy's host DM halo
            - `halo_ngal`: Total number of galaxies in the DM halo
            - `M_r`: r-band absolute magnitude from ECO via abundance matching
            - `galid`: Galaxy ID

    param_dict: python dictionary
        dictionary with `project` variables

    Returns
    ----------
    clf_pd: pandas DataFrame
        Updated version of `clf_pd` with new columns of distances
        New key: `dist_c` --> Distance to the satellite's central galaxy
    """
    ##
    ## TO DO: Fix issue with central and satellite being really close to 
    ##        the box boundary, therefore having different final distances
    ##
    ## Centrals and Satellites
    cens            = param_dict['cens']
    sats            = param_dict['sats']
    dist_c_label    = 'dist_c'
    dist_sq_c_label = 'dist_c_sq'
    ## Galaxy coordinates
    coords  = ['x'   ,'y'   , 'z'  ]
    coords2 = ['x_sq','y_sq','z_sq']
    ## Unique HaloIDs
    haloid_unq = num.unique(clf_pd.loc[clf_pd['halo_ngal'] != 1,'haloid'])
    n_halo_unq = len(haloid_unq)
    ## CLF columns
    clf_cols = ['x','z','y','haloid','cs_flag']
    ## Copy of `clf_pd`
    clf_pd_mod = clf_pd[clf_cols].copy()
    ## Initializing new column in `clf_pd`
    clf_pd_mod.loc[:,dist_sq_c_label] = num.zeros(clf_pd.shape[0])
    ## Positions squared
    clf_pd_mod.loc[:,'x_sq'] = clf_pd_mod['x']**2
    clf_pd_mod.loc[:,'y_sq'] = clf_pd_mod['y']**2
    clf_pd_mod.loc[:,'z_sq'] = clf_pd_mod['z']**2
    ## Looping over number of halos
    # ProgressBar properties
    for ii, halo_ii in enumerate(tqdm(haloid_unq)):
        ## Halo ID subsample
        halo_ii_pd   = clf_pd_mod.loc[clf_pd_mod['haloid']==halo_ii]
        ## Cens and Sats DataFrames
        cens_coords = halo_ii_pd.loc[halo_ii_pd['cs_flag']==cens, coords]
        sats_coords = halo_ii_pd.loc[halo_ii_pd['cs_flag']==sats, coords]
        sats_idx    = sats_coords.index.values
        ## Distances from central galaxy
        cens_coords_mean = cens_coords.mean(axis=0).values
        ## Difference in coordinates
        dist_sq_arr = num.sum(
            sats_coords.subtract(cens_coords_mean, axis=1).values**2, axis=1)
        ## Assigning distances to each satellite
        clf_pd_mod.loc[sats_idx, dist_sq_c_label] = dist_sq_arr
    ##
    ## Taking the square root of distances
    clf_pd_mod.loc[:,dist_c_label] = (clf_pd_mod[dist_sq_c_label].values)**.5
    ##
    ## Assigning it to `clf_pd`
    clf_pd.loc[:, dist_c_label] = clf_pd_mod[dist_c_label].values

    return clf_pd

def mr_survey_matching(clf_pd, param_dict, proj_dict):
    """
    Finds the closest r-band absolute magnitude from ECO catalogue 
    and assigns them to mock galaxies

    Parameters
    -------------
    clf_pd: pandas DataFrame
        DataFrame containing information from Halobias + CLF procedures
    
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    Returns
    -------------
    clf_galprop_pd: pandas DataFrame
        DataFrame with updated values.
        New values included:
            - Morphology
            - Stellar mass
            - r-band apparent magnitude
            - u-band apparent magnitude
            - FSMGR
            - `Match_Flag`:
            - MHI
            - Survey flag: {1 == ECO, 0 == Resolve B}
    """
    Prog_msg   = param_dict['Prog_msg']
    failval    = 0.
    ngal_mock  = len(clf_pd)
    ## Survey flags
    eco_flag = 1
    res_flag = 0
    ## Copy of `clf_pd`
    clf_pd_mod = clf_pd.copy()
    ## Filenames
    eco_phot_file   = param_dict['files_dict']['eco_phot_file_local']
    eco_mhi_file    = param_dict['files_dict']['mhi_file_local']
    res_b_phot_file = param_dict['files_dict']['res_b_phot_file_local']
    ## ECO Photometry catalogue
    #  - reading in dictionary
    eco_phot_dict = readsav(eco_phot_file, python_dict=True)
    #  - Converting to Pandas DataFrame
    eco_phot_pd   = Table(eco_phot_dict).to_pandas()
    ##
    ## ECO - MHI measurements
    eco_mhi_pd    = pd.read_csv(eco_mhi_file, sep='\s+', names=['MHI'])
    ## Appending to `eco_phot_pd`
    try:
        assert(len(eco_phot_pd) == len(eco_mhi_pd))
        ## Adding to DataFrame
        eco_phot_pd.loc[:,'MHI'] = eco_mhi_pd['MHI'].values
    except:
        msg = '{0} `len(eco_phot_pd)` != `len(eco_mhi_pd)`! Unequal lengths!'
        msg = msg.format(Prog_msg)
        raise ValueError(msg)
    ##
    ## Cleaning up DataFrame - r-band absolute magnitudes
    eco_phot_mod_pd = eco_phot_pd.loc[(eco_phot_pd['goodnewabsr'] != failval) &
                      (eco_phot_pd['goodnewabsr'] < param_dict['mr_limit'])]
    eco_phot_mod_pd.reset_index(inplace=True, drop=True)
    ##
    ## Reading in `RESOLVE B` catalogue - r-band absolute magnitudes
    res_phot_pd = Table(fits.getdata(res_b_phot_file)).to_pandas()
    res_phot_mod_pd = res_phot_pd.loc[\
                        (res_phot_pd['ABSMAGR'] != failval) &
                        (res_phot_pd['ABSMAGR'] <= param_dict['mr_res_b']) &
                        (res_phot_pd['ABSMAGR'] >  param_dict['mr_eco'  ])]
    res_phot_mod_pd.reset_index(inplace=True, drop=True)
    ##
    ## Initializing arrays
    morph_arr       = [[] for x in range(ngal_mock)]
    dex_rmag_arr    = [[] for x in range(ngal_mock)]
    dex_umag_arr    = [[] for x in range(ngal_mock)]
    logmstar_arr    = [[] for x in range(ngal_mock)]
    fsmgr_arr       = [[] for x in range(ngal_mock)]
    survey_flag_arr = [[] for x in range(ngal_mock)]
    mhi_arr         = [[] for x in range(ngal_mock)]
    ##
    ## Assigning galaxy properties to mock galaxies
    #
    clf_mr_arr = clf_pd_mod['M_r'].values
    eco_mr_arr = eco_phot_mod_pd['goodnewabsr'].values
    res_mr_arr = res_phot_mod_pd['ABSMAGR'].values
    ## Galaxy properties column names
    eco_cols = ['goodmorph','rpsmoothrestrmagnew','rpsmoothrestumagnew',
                'rpgoodmstarsnew','rpmeanssfr']
    res_cols = ['MORPH', 'SMOOTHRESTRMAG','SMOOTHRESTUMAG','MSTARS',
                'MODELFSMGR']
    # Looping over all galaxies
    for ii in tqdm(range(ngal_mock)):
        ## Galaxy r-band absolute magnitude
        gal_mr = clf_mr_arr[ii]
        ## Choosing which catalogue to use
        if gal_mr <= param_dict['mr_eco']:
            idx_match  = cu.closest_val(gal_mr, eco_mr_arr)
            survey_tag = eco_flag
            ## Galaxy Properties
            (   morph_val,
                rmag_val ,
                umag_val ,
                logmstar_val,
                fsmgr_val) = eco_phot_mod_pd.loc[idx_match, eco_cols].values
            # MHI value
            mhi_val   = 10**(eco_phot_mod_pd['MHI'][idx_match] + logmstar_val)
        elif (gal_mr > param_dict['mr_eco']) and (gal_mr <= param_dict['mr_res_b']):
            idx_match  = cu.closest_val(gal_mr, res_mr_arr)
            survey_tag = res_flag
            ## Galaxy Properties
            (   morph_val,
                rmag_val ,
                umag_val ,
                mstar_val,
                fsmgr_val) = res_phot_mod_pd.loc[idx_match, res_cols]
            ## MHI value
            mhi_val = res_phot_mod_pd.loc[idx_match, 'MHI']
            ## Fixing issue with units
            logmstar_val = num.log10(mstar_val)
        ##
        ## Assigning them to arrays
        morph_arr       [ii] = morph_val
        dex_rmag_arr    [ii] = rmag_val
        dex_umag_arr    [ii] = umag_val
        logmstar_arr    [ii] = logmstar_val
        fsmgr_arr       [ii] = fsmgr_val
        mhi_arr         [ii] = mhi_val
        survey_flag_arr [ii] = survey_tag
    ##
    ## Assigning them to `clf_pd_mod`
    clf_pd_mod.loc[:,'morph'      ] = morph_arr
    clf_pd_mod.loc[:,'rmag'       ] = dex_rmag_arr
    clf_pd_mod.loc[:,'umag'       ] = dex_umag_arr
    clf_pd_mod.loc[:,'logmstar'   ] = logmstar_arr
    clf_pd_mod.loc[:,'fsmgr'      ] = fsmgr_arr
    clf_pd_mod.loc[:,'mhi'        ] = mhi_arr
    clf_pd_mod.loc[:,'survey_flag'] = survey_flag_arr
    clf_pd_mod.loc[:,'u_r'        ] = clf_pd_mod['umag'] - clf_pd_mod['rmag']
    ##
    ## Dropping all other columns
    galprop_cols = ['morph','rmag','umag','logmstar','fsmgr','mhi',
                    'survey_flag', 'u_r']
    clf_pd_mod_prop = clf_pd_mod[galprop_cols].copy()
    ##
    ## Merging DataFrames
    clf_galprop_pd = pd.merge(clf_pd, clf_pd_mod_prop, 
                                left_index=True, right_index=True)

    return clf_galprop_pd

## -----------| Mock-catalogues-related functions |----------- ##







## -----------| Main functions |----------- ##

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
    ##
    ## Cosmological model and Halo mass function
    cosmo_model, cosmo_hmf = cosmo_create(param_dict['cosmo_choice'])
    ## Survey Details
    param_dict = survey_specs(param_dict, cosmo_model)
    ##
    ## Mass function for given cosmology
    hmf_pd = hmf_calc(cosmo_model, proj_dict, param_dict, Mmin=6., Mmax=16.01,
        dlog10m=1.e-3, hmf_model=param_dict['hmf_model'])
    ##
    ## Downloading files
    param_dict = download_files(param_dict, proj_dict)
    ##
    ## Redshift and Comoving distance
    z_como_pd = z_comoving_calc(param_dict, proj_dict, cosmo_model)
    ## Halobias Extras file - Modified Halobias file
    (   param_dict,
        hb_pd     ) = hb_file_construction_extras(param_dict, proj_dict)
    ## Conditional Luminosity Function
    clf_pd = clf_assignment(param_dict, proj_dict)
    ## Distance from Satellites to Centrals
    clf_pd = cen_sat_distance_calc(clf_pd, param_dict)
    ## Finding closest magnitude value from ECO catalogue
    clf_pd = mr_survey_matching(clf_pd, param_dict, proj_dict)
    ## Carving out geometry of Survey and carrying out the analysis
    if param_dict['survey'] == 'ECO':
        eco_geometry_mocks(clf_pd, param_dict, proj_dict)







# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)
