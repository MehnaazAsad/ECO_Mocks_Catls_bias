#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 02/21/2018
# Last Modified: 02/21/2018
# Vanderbilt University
from __future__ import print_function, division, absolute_import
__author__     =['Victor Calderon']
__copyright__  =["Copyright 2017 Victor Calderon, ECO/RESOLVE Mocks - Create"]
__email__      =['victor.calderon@vanderbilt.edu']
__maintainer__ =['Victor Calderon']
"""
Script that runs the `eco_mocks_create` scripts for `ECO`, 
`RESOLVE A` and `RESOLVE B` surveys.
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
import os
import sys
import pandas as pd

# Extra-modules
from argparse import ArgumentParser
from argparse import HelpFormatter
from operator import attrgetter
import datetime

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
    description_msg = 'Script that creates the set of synthetic SDSS catalogues'
    parser = ArgumentParser(description=description_msg,
                            formatter_class=SortingHelpFormatter,)
    ## 
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    ## Type of Abundance matching
    parser.add_argument('-abopt',
                        dest='catl_type',
                        help='Type of Abund. Matching used in catalogue',
                        type=str,
                        choices=['mr', 'mstar'],
                        default='mr')
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
    ## Option for removing file
    parser.add_argument('-remove',
                        dest='remove_files',
                        help="""
                        Delete files created by the script, in case the exist 
                        already""",
                        type=_str2bool,
                        default=False)
    ## CPU Counts
    parser.add_argument('-cpu',
                        dest='cpu_frac',
                        help='Fraction of total number of CPUs to use',
                        type=float,
                        default=0.75)
    ## Verbose
    parser.add_argument('-v','--verbose',
                        dest='verbose',
                        help='Option to print out project parameters',
                        type=_str2bool,
                        default=True)
    ## Parsing Objects
    args = parser.parse_args()

    return args

def get_analysis_params(param_dict):
    """
    Parameters for the 1-halo conformity analysis

    Parameters
    -----------
    param_dict: python dictionary
        dictionary with project variables

    Returns
    --------
    params_pd_data: pandas DataFrame
        DataFrame with necessary parameters to run `sdss data create`

    """
    ##
    ## Array of values used for the analysis.
    ## Format: (name of variable, flag, value)
    #
    ## Data
    params_arr_data = num.array([('size_cube'  ,'-sizecube' ,180     ),
                                 ('catl_type'   ,'-abopt'   ,'mr'    ),
                                 ('zmedian'     ,'-zmed'    ,0.      ),
                                 ('survey'      ,'-survey'  ,'ECO'   ),
                                 ('halotype'    ,'-halotype','mvir'  ),
                                 ('coso_choice' ,'-cosmo'   ,'Planck'),
                                 ('hmf_model'   , '-hmf'    ,'warren'),
                                 ('clf_type'    ,'-clf'     ,2       ),
                                 ('zspace'      ,'-zspace'  ,2       ),
                                 ('nmin'        ,'-nmin'    ,1       ),
                                 ('remove_files','-remove'  ,'False' ),
                                 ('verbose'     ,'-v'       ,'True'  ),
                                 ('cpu_frac'    ,'-cpu'     ,0.75    )])
    ##
    ## Converting to pandas DataFrame
    colnames        = ['Name','Flag','Value']
    # Data
    params_pd_data  = pd.DataFrame(params_arr_data, columns=colnames)
    ##
    ## Sorting out DataFrames by `name`
    params_pd_data  = params_pd_data.sort_values( by='Name').reset_index(drop=True)
    ##
    ## `Removing files`
    if param_dict['remove_files']:
        ## Overwriting `remove_files` from `params_pd_data`
        params_pd_data.loc[params_pd_data['Name']=='remove_files','Value'] = 'True'
    ##
    ## Galaxy survey
    params_pd_data.loc[params_pd_data['Name']=='survey','Value'] = param_dict['survey']
    ##
    ## Ab. Matching type
    params_pd_data.loc[params_pd_data['Name']=='catl_type','Value'] = param_dict['catl_type']
    ##
    ## CPUs
    params_pd_data.loc[params_pd_data['Name']=='cpu_frac','Value'] = param_dict['cpu_frac']
    ##
    ## HMF Model
    params_pd_data.loc[params_pd_data['Name']=='hmf_model','Value'] = param_dict['hmf_model']
    ##
    ## Cosmological model
    params_pd_data.loc[params_pd_data['Name']=='cosmo_choice','Value'] = param_dict['cosmo_choice']
    ##
    ## Halo definition
    params_pd_data.loc[params_pd_data['Name']=='halotype','Value'] = param_dict['halotype']

    return params_pd_data

def get_exec_string(params_pd_data, param_dict):
    """
    Produces string be executed in the bash file

    Parameters
    -----------
    params_pd_data: pandas DataFrame
        DataFrame with necessary parameters to run `sdss data create`

    param_dict: python dictionary
        dictionary with project variables

    Returns
    -----------
    string_dict: python dictionary
        dictionary containing strings for `data` and `mocks`
    """
    ## Current directory
    working_dir = os.path.abspath(os.path.dirname(__file__))
    ## Choosing which file to run
    CATL_MAKE_file_data  = 'eco_mocks_create.py'
    ##
    ## Getting path to `CATL_MAKE_*` files
    data_path  = os.path.join(working_dir, 'mocks_create' , CATL_MAKE_file_data )
    ## Check if File exists
    # Data
    if os.path.isfile(data_path):
        pass
    else:
        msg = '{0} `CATL_MAKE_file_data` ({1}) not found!! Exiting!'.format(
            param_dict['Prog_msg'], data_path)
        raise ValueError(msg)
    ##
    ## Constructing string for `data` and `mocks`
    # Data
    data_string = 'python {0} '.format(data_path)
    for ii in range(params_pd_data.shape[0]):
        ## Appending to string
        data_string += ' {0} {1}'.format(   params_pd_data['Flag' ][ii],
                                            params_pd_data['Value'][ii])
    ##
    ## Saving to dictionary
    string_dict = {'data':data_string}

    return string_dict

def file_construction_and_execution(params_pd_data, param_dict):
    """
    1) Creates file that has shell commands to run executable
    2) Executes the file, which creates a screen session with the executables

    Parameters:
    -----------
    params_pd_data: pandas DataFrame
        DataFrame with necessary parameters to run `sdss data create`

    param_dict: python dictionary
        dictionary with project variables
    
    """
    ##
    ## Getting today's date
    now_str = datetime.datetime.now().strftime("%x %X")
    ##
    ## Obtain MCF strings
    string_dict = get_exec_string(params_pd_data, param_dict)
    ##
    ## Parsing text that will go in file
    # Working directory
    working_dir = os.path.abspath(os.path.dirname(__file__))
    ## Obtaining path to file
    outfile_name = 'ECO_resolve_mocks_{0}_create_run.sh'.format(
        param_dict['catl_type'])
    outfile_path = os.path.join(working_dir, outfile_name)
    ##
    ## Opening file
    with open(outfile_path, 'wb') as out_f:
        out_f.write(b"""#!/usr/bin/env bash\n\n""")
        out_f.write(b"""## Author: Victor Calderon\n\n""")
        out_f.write( """## Last Edited: {0}\n\n""".format(now_str).encode())
        out_f.write(b"""### --- Variables\n""")
        out_f.write(b"""ENV_NAME="eco_mocks_catls"\n""")
        out_f.write( """WINDOW_NAME="ECO_RESOLVE_Mocks_create_{0}"\n""".format(param_dict['halotype']).encode())
        out_f.write( """WINDOW_CATL="data_{0}_{1}"\n""".format(param_dict['catl_type'], param_dict['cosmo_choice']).encode())
        out_f.write(b"""# Home Directory\n""")
        out_f.write(b"""home_dir=`eval echo "~$different_user"`\n""")
        out_f.write(b"""# Type of OS\n""")
        out_f.write(b"""ostype=`uname`\n""")
        out_f.write(b"""# Sourcing profile\n""")
        out_f.write(b"""if [[ $ostype == "Linux" ]]; then\n""")
        out_f.write(b"""    source $home_dir/.bashrc\n""")
        out_f.write(b"""else\n""")
        out_f.write(b"""    source $home_dir/.bash_profile\n""")
        out_f.write(b"""fi\n""")
        out_f.write(b"""# Activating Environment\n""")
        out_f.write(b"""activate=`which activate`\n""")
        out_f.write(b"""source $activate ${ENV_NAME}\n""")
        out_f.write(b"""###\n""")
        out_f.write(b"""### --- Python Strings\n""")
        out_f.write( """ECO_RESOLVE="{0}"\n""".format( string_dict['data' ]).encode())
        out_f.write(b"""###\n""")
        out_f.write(b"""### --- Deleting previous Screen Session\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -X quit\n""")
        out_f.write(b"""###\n""")
        out_f.write(b"""### --- Screen Session\n""")
        out_f.write(b"""screen -mdS ${WINDOW_NAME}\n""")
        out_f.write(b"""## Mocks\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -X screen -t ${WINDOW_CATL}\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -p ${WINDOW_CATL} -X stuff $"source $activate ${ENV_NAME};"\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -p ${WINDOW_CATL} -X stuff $"${ECO_RESOLVE};"\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -p ${WINDOW_CATL} -X stuff $'\\n'\n""")
        out_f.write(b"""\n""")
    ##
    ## Check if File exists
    if os.path.isfile(outfile_path):
        pass
    else:
        msg = '{0} `outfile_path` ({1}) not found!! Exiting!'.format(
            param_dict['Prog_msg'], outfile_path)
        raise ValueError(msg)
    ##
    ## Make file executable
    print(".>>> Making file executable....")
    print("     chmod +x {0}".format(outfile_path))
    os.system("chmod +x {0}".format(outfile_path))
    ##
    ## Running script
    print(".>>> Running Script....")
    os.system("{0}".format(outfile_path))

def main(args):
    """
    Computes the analysis and 
    """
    ## Reading all elements and converting to python dictionary
    param_dict = vars(args)
    ##
    ## Parameters for the analysis
    params_pd_data = get_analysis_params(param_dict)
    ##
    ## Running analysis
    file_construction_and_execution(    params_pd_data ,
                                        param_dict     )

# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)
