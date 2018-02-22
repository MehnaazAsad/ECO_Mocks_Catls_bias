#! /usr/bin/env python

import pandas as pd
import os

def reading_catls(filename, catl_format='.hdf5'):
    """
    Function to read ECO/RESOLVE catalogues.

    Parameters
    ----------
    filename: string
        path and name of the ECO/RESOLVE catalogue to read

    catl_format: string, optional (default = '.hdf5')
        type of file to read.
        Options:
            - '.hdf5': Reads in a catalogue in HDF5 format

    Returns
    -------
    mock_pd: pandas DataFrame
        DataFrame with galaxy/group information

    Examples
    --------
    # Specifying `filename`
    >>> filename = 'ECO_catl.hdf5'

    # Reading in Catalogue
    >>> mock_pd = reading_catls(filename, format='.hdf5')

    >>> mock_pd.head()
            M_h      M_r        cz       dec  galtype  halo_ngal  haloid  \
    0  11.40841 -19.2752  19890.61  1.240539        1          1   31535
    1  11.69354 -19.7106  19946.01  1.418415        1          1   31537
    2  12.85093 -19.4581  20004.53  0.548292        0          4   31539
    3  12.85093 -19.8278  20055.74  0.502586        0          4   31539
    4  12.09574 -20.3131  19610.21  1.408745        1          1   31554

        logssfr          ra  groupid    M_group  g_galtype  g_ngal  halo_rvir
    0 -10.55625  180.318422        0  11.995713        0.0     2.0   0.101174
    1 -12.01112  180.735027        1  14.148499        0.0    31.0   0.125917
    2 -11.69736  179.630277        2  12.011946        0.0     2.0   0.306085
    3 -12.08712  179.650267        2  12.011946        1.0     2.0   0.306085
    4 -10.56714  180.094054        3  11.956849        1.0     1.0   0.171519
    """
    ## Checking if file exists
    if not os.path.exists(filename):
        msg = '`filename`: {0} NOT FOUND! Exiting..'.format(filename)
        raise ValueError(msg)
    ## Reading file
    if catl_format=='.hdf5':
        mock_pd = pd.read_hdf(filename)
	else:
		msg = '`catl_format` ({0}) not supported! Exiting...'.format(catl_format)
		raise ValueError(msg)

    return mock_pd

def main():
    # Specifying filename
    filename = 'ECO_catl.hdf5'
    # Reading in ECO/RESOLVE catalogue
    mock_pd = reading_catls(filename)


if __name__=='__main__':
    main()
