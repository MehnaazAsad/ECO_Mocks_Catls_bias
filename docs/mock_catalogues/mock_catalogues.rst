.. ECO_Resolve_Catalogues

.. _Mock_Catalogues:
===================
Mock Catalogues
===================

We construct a set of synthethic (mock) catalogues that have the same 
geometries as the **Environmental COntext** (ECO), **RESOLVE-A**, and 
**RESOLVE-B** galaxy surveys.

RESOLVE is a volume-limited census of stellar, gas, and dynamical mass as 
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

This shows the right-ascension (RA) and declination (DEC) of 
galaxies in RESOLVE-A and RESOLVE-B galaxy redshift surveys.

.. image:: ../images/resolve_spring_footprint.png
    :scale: 50 %
    :alt: 
    :align: center

RESOLVE-A (footprint demarcated by red dashed lines) embedded within ECO 
(entire plot showing current footprint, with ECO-B in preparation)

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
| A        | (131.25, 236.25)| 105.0    |(0  ,+5)     | 5         | 0.00844| 0.0249  | 2532        | 7470.      |(25.32,70.02)|
+----------+-----------------+----------+-------------+-----------+--------+---------+-------------+------------+-------------+
| B        | (330.0 , 45.0  )| 75.0     |(-1.25,+1.25)| 2.5       | 0.01416| 0.024166| 4250        | 7250.      |(42.5 , 72.5)|
+----------+-----------------+----------+-------------+-----------+--------+---------+-------------+------------+-------------+
| ECO      | (130.05, 237.45)| 107.4    |(-1, +49.85) | 50.85     | 0.00844| 0.0249  | 2532        | 7470.      |(25.32,70.02)|
+----------+-----------------+----------+-------------+-----------+--------+---------+-------------+------------+-------------+


