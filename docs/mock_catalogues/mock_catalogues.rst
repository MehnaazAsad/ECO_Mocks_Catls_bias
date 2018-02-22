.. ECO_Resolve_Catalogues

.. _Mock_Catalogues:
===================
Mock Catalogues
===================

This is a brief overview of the different aspects of the synthetic 
catalogues produced for ECO RESOLVE-A and RESOLVE-B surveys

.. _eco_resolve_main_data:
--------------------------
ECO and RESOLVE
--------------------------

We construct a set of synthethic (mock) catalogues that have the same 
geometries as the **Environmental COntext** (ECO), **RESOLVE-A**, and 
**RESOLVE-B** galaxy surveys.

REsolved Spectroscopy Of a Local VolumE (RESOLVE) is a volume-limited 
census of stellar, gas, and dynamical mass as 
well as star formation and merging within >50,000 cubic Mpc of the nearby 
cosmic web, reaching down to the dwarf galaxy regime and up to structures 
on tens of Mpc scales such as filaments, walls, and voids.

The Environmental COntext (ECO) catalog around RESOLVE is a much larger, 
purely archival data set with pipelines and methods matched to RESOLVE, 
enabling statistically robust analyses of environmental trends and 
calibration of cosmic variance.

.. image:: ../images/full-size-survey.png
    :scale: 50 %
    :alt: Full size survey of RESOLVE
    :align: center

**This shows the right-ascension (RA) and declination (DEC) of 
galaxies in RESOLVE-A and RESOLVE-B galaxy redshift surveys.**

.. image:: ../images/resolve_spring_footprint.png
    :scale: 100 %
    :alt: 
    :align: center

**RESOLVE-A (footprint demarcated by red dashed lines) embedded within ECO 
(entire plot showing current footprint, with ECO-B in preparation)**

For more information on how the data for the different galaxy surveys 
were taken, go to the `Main ECO and RESOLVE <https://resolve.astro.unc.edu/>`_
website.

.. _mock_construction:
--------------------------
Constructing catalogues
--------------------------

We design the *synthetic* catalogues to have the exact same 
geometries and redshift limits as those of the ECO, RESOLVE-A, and 
RESOLVE-B galaxy surveys.

This is a summary of the values used to create the synthetic galaxy catalogues.
These catalogues are taking a *buffer* regions, which is an *extra* buffer 
region along the `cz` (velocity) direction in redshift-space.

+----------+-----------------+----------+-------------+-----------+--------+---------+-------------+------------+-------------+
| Survey   | RA (deg)        | RA range | DEC (deg)   | DEC range | zmin   | zmax    | Vmin (km/s) | Vmax (km/s)| Dist (Mpc)  |
+==========+=================+==========+=============+===========+========+=========+=============+============+=============+
| A        | (131.25, 236.25)| 105.0    |(0  ,+5)     | 5         | 0.00844| 0.0249  | 2532        |  7470.     |(25.32,70.02)|
+----------+-----------------+----------+-------------+-----------+--------+---------+-------------+------------+-------------+
| B        | (330.0 , 45.0  )| 75.0     |(-1.25,+1.25)| 2.5       | 0.01416| 0.024166| 4250        |  7250.     |(42.5 , 72.5)|
+----------+-----------------+----------+-------------+-----------+--------+---------+-------------+------------+-------------+
| ECO      | (130.05, 237.45)| 107.4    |(-1, +49.85) | 50.85     | 0.00844| 0.0249  | 2532        | 7470.      |(25.32,70.02)|
+----------+-----------------+----------+-------------+-----------+--------+---------+-------------+------------+-------------+

The next table provides the number of synthetic catalogues per cubic box of **L = 180 Mpc/h**, where *h* = 1.

+--------+--------------+
| Survey | Number Mocks |
+========+==============+
| A      | 59           |
+--------+--------------+
| B      | 104          |
+--------+--------------+
| ECO    | 8            |
+--------+--------------+

.. _mock_distribution_box:
-----------------------------------------------
Distribution of catalogues in simulation box
-----------------------------------------------

In order to maximize the number of catalogues per simulation, we 
have to fit as many catalogues as we can, while keeping a 
distance of ~10 Mpc/h between catalogues. We chose this distance of 
10 Mpc/h in order to avoid using the same galaxy for different 
catalogues, and also to make the catalogues as independent from each 
other as possible.

.. image:: ../images/ECO_mvir_xyz_mocks.png
    :align: center
    :alt: Distribution of mock catalogues within simulation box
    :scale: 50 %

This figure shows how the catalogues for ECO surveys are organized 
within the simulation box used for this analysis.

.. download_read_catalogues::
------------------------------------------------
Downloading and reading in data from catalogues
------------------------------------------------

The mock catalogues are located at 
`<http://lss.phy.vanderbilt.edu/groups/data_eco_vc/Mock_Catalogues/>`_.

These catalogues can be downloaded as *tar* files, and be read by 
the Python package `Pandas <https://pandas.pydata.org/>`_.

After having downloaded your file, you can read them in the following way:

.. literalinclude:: ../example_scripts/read_ECO_RESOLVE_catls.py

.. properties_description::
------------------------------------------------
Description of the *fields* in the catalogues
------------------------------------------------


Each mock catalogues contains information about the **galaxy**, 
**group galaxy**, **host halo**, and more.
