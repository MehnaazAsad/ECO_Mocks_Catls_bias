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
    ## HOD Parameters
    hod_dict             = {}
    hod_dict['logMmin' ] = 11.4
    hod_dict['sigLogM' ] = 0.2
    hod_dict['logM0'   ] = 10.8
    hod_dict['logM1'   ] = 12.8
    hod_dict['alpha'   ] = 1.05
    hod_dict['zmed_val'] = 'z0p000'
    ## Choice of Survey
    choice_survey = 2
    ###
    ### URL to download catalogues
    url_catl = 'http://lss.phy.vanderbilt.edu/groups/data_eco_vc/'
    url_checker(url_catl)
    ##
    ## Adding to `param_dict`
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
    hb_file_mod = os.path.join(proj_dict['h_ngal_dir'],
                                    os.path.basename(hb_local)+'.halongal')
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
    ## Converting to Pandas DataFrame
    # Dictionary
    hb_dict = {}
    hb_dict['x'      ] = x_arr
    hb_dict['y'      ] = y_arr
    hb_dict['z'      ] = z_arr
    hb_dict['vx'     ] = vx_arr
    hb_dict['vy'     ] = vy_arr
    hb_dict['vz'     ] = vz_arr
    hb_dict['halom'  ] = halom
    hb_dict['cs_flag'] = cs_flag
    hb_dict['haloid' ] = haloid
    # To DataFrame
    hb_cols = ['x','y','z','vx','vy','vz','halom','cs_flag','haloid']
    hb_pd   = pd.DataFrame(hb_dict)[hb_cols]
    ##
    ## Counter of HaloIDs
    haloid_counts = Counter(haloid)
    # Array of `gals` in each `haloid`
    haloid_ngal = [[] for x in range(ngal)]
    for kk, halo_kk in enumerate(haloid):
        haloid_ngal[kk] = haloid_counts[halo_kk]
    haloid_ngal = num.asarray(haloid_ngal).astype(int)
    # Saving to file
    with open(hb_file_mod,'wb') as hb_ngal:
        num.savetxt(hb_ngal, haloid_ngal, fmt='%d')
    cu.File_Exists(hb_file_mod)
    ## Assigning to `param_dict`
    param_dict['hb_file_mod'] = hb_file_mod
    param_dict['hb_cols'    ] = hb_cols
    # Message
    print('\n{0} Halo_ngal file: {1}'.format(Prog_msg, hb_file_mod))
    print('{0} Creating file with Ngals in each halo ... Complete'.format(Prog_msg))

    return param_dict, hb_pd

def hb_file_create(param_dict, proj_dict, hb_pd, ext='txt'):
    """
    Creates a modified version of the Halobias file

    Parameters
    ----------
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    ext: string, optional (default = 'txt')
        extension of the output file
    """
    ## Local halobias file
    hb_local    = param_dict['files_dict']['hb_file_local']
    ## Halobias modified version file
    hb_file_mod = os.path.join( proj_dict['hb_files_dir'],
                                os.path.basename(hb_local)+'.mod')




def clf_assignment(param_dict, proj_dict):
    """
    Computes the conditional luminosity function on the halobias file
    
    Parameters
    ----------
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.
    """
    ## HOD dictionary
    hod_dict = param_dict['hod_dict']
    ## CLF Executable
    clf_exe = os.path.join( cu.get_code_c(),
                            'CLF',
                            'CLF_with_ftread')
    cu.File_Exists(clf_exe)
    ## CLF Commands
    cmd_arr = [clf_exe, hod_dict['logMmin'], ]





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
    ## Halobias Extras file
    (   param_dict,
        hb_pd     ) = hb_file_construction_extras(param_dict, proj_dict)
    ## Creating modified version of Halobias file
    hb_file_create(param_dict, proj_dict)
    ## Conditional Luminosity Function





# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)
