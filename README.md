ECO Mock Catalogues
==============================

Repository for creating ECO synthetic catalogues

**Author**: Victor Calderon ([victor.calderon@vanderbilt.edu](mailto:victor.calderon@vanderbilt.edu))

[![Documentation Status](https://readthedocs.org/projects/eco-mocks-catls/badge/?version=latest)](http://eco-mocks-catls.readthedocs.io/en/latest/?badge=latest)


## Installing Environment & Dependencies

To use the scripts in this repository, you must have _Anaconda_ installed on the systems that will be running the scripts. This will simplify the process of installing all the dependencies.

For reference, see: [https://conda.io/docs/user-guide/tasks/manage-environments.html](https://conda.io/docs/user-guide/tasks/manage-environments.html)

The package counts with a __Makefile__ with useful functions. You must use this Makefile to ensure that you have all the necessary _dependencies_, as well as the correct _conda environment_. 

* Show all available functions in the _Makefile_

```
$:  make show-help
    
    Available rules:
    
    clean               Delete all compiled Python files
    environment         Set up python interpreter environment - Using environment.yml
    lint                Lint using flake8
    remove_environment  Delete python interpreter environment
    test_environment    Test python environment is setup correctly
    update_environment  Update python interpreter environment
```

* __Create__ the environment from the `environment.yml` file:

```
    make environment
```

* __Activate__ the new environment __eco_mocks_catls__.

```
    source activate eco_mocks_catls
```

* To __update__ the `environment.yml` file (when the required packages have changed):

```
  make update_environment
```

* __Deactivate__ the new environment:

```
    source deactivate
```

### Auto-activate environment
To make it easier to activate the necessary environment, one can check out [*conda-auto-env*](https://github.com/chdoig/conda-auto-env), which activates the necessary environment automatically.


This is a summary of the values used to create these mock catalogues:
These catalogues are taking into account the extra buffer along the _cz_ direction in redshift space.

|  Survey  | RA (Deg)       | RA Range | DEC (Deg)   | DEC Range | *zmin* | *zmax*  | Vmin (km/s) | Vmax (km/s)| Dist (Mpc)  |
|----------|:--------------:|:--------:|:-----------:|:---------:|:------:|:-------:|:-----------:|:----------:|:-----------:|
| A        |(131.25, 236.25)| 105.0    |(0  ,+5)     | 5         | 0.00844| 0.0249  | 2532        | 7470.      |(25.32,70.02)|
| B        |(330.0 , 45.0  )| 75.0     |(-1.25,+1.25)| 2.5       | 0.01416| 0.024166| 4250        | 7250.      |(42.5 , 72.5)|
|ECO       |(130.05, 237.45)| 107.4    |(-1, +49.85) | 50.85     | 0.00844| 0.0249  | 2532        | 7470.      |(25.32,70.02)|

The next table provides the number of mock catalogues per cubic box of L=180 Mpc/h.

| Survey       | N Mocks   |
|:--------     |:---------:|
|**Resolve A** | 59        |
|**Resolve B** | 104       |
|**ECO**       | 8         |

---

Project Description
--------------------

## <span style='color:blue'>Mock Catalogues</span>

The mock catalogues have the same geometries as those of the real surveys. The mock catalogues consist of a total of **26 columns**, each column providing information about the individual galaxy and its host DM halo, and properties from real galaxy catalogues.
Aside from the values obtained from the simulation ( Columns 1-12), we matched properties from real catalogues (i.e. ECO and RESOLVE A/B) to the mock catalogues by finding the r-band absolute magnitude in real data that resembles that of the mock galaxy catalogues.
For r-band absolute magnitudes between -17.33 < Mr <= -17.00, we matched the mock galaxy's Mr to a galaxy in RESOLVE B (resolvecatalog_str.fits | Updated on 2015-07-16)
For r-band absolute magnitudes brighter than Mr = -17.33, we matched the mock galaxy's Mr to a galaxy in ECO (eco_wresa_050815.dat | Updated on 2015-05-08).
For each matched galaxy, we attached the galaxy's properties to the matched mock galaxy catalogue.

We also ran the Berlind2006 Friends-of-Friends algorithm on each mock catalogue, and assigned an estimated mass to the galaxy group through _Halo Abundance Matching_.

For observables in the real data, the joint probability distributions are the same as those in the real data.

For all the values, we use the following cosmology:

    Omega_M_0     : 0.302
    Omega_lambda_0: 0.698
    Omega_k_0     : 0.0
    h             : 0.698

For the Group finding, we used the following parameters and linking lengths:

    Linking Parallel: 1.1
    Linking Perpend.: 0.07
    Nmin            : 1


### <span style="color:Peru">Columns</span>
##### Theory Columns     : Columns 1 - 12
##### Observables Columns: Columns 13 - 20 
##### Group Columns      : Columns 21 - 25
   1. <span style="color:OrangeRed">__Right Ascension__</span>  : RA of the individual galaxy, given in _degrees_
   2. <span style="color:OrangeRed">__Declination__</span>      : Declination of the ind. galaxy, given in _degrees_
   3. <span style="color:OrangeRed">__CZ_Obs__</span>           : Velocity of the galaxy ( _with_ redshift distortions), given in _km/s_
   4. <span style="color:OrangeRed">__Mr__</span>               : Galaxy's magn. in the r-band. Calculated using a CLF approach, but using real photometry from survey.
   5. <span style="color:OrangeRed">__Halo ID__</span>          : DM Halo identification number, as taken from the simulation
   6. <span style="color:OrangeRed">__log(MHalo)__</span>       : Logarithmic value of the DM Halo's Mass, as __log( MHalo / (Msun/h) )__ with h=0.698.
   7. <span style="color:OrangeRed">__NGals_h__</span>          : Total number of galaxies in DM halo. Number of galaxies in the mock may differ from this value.
   8. <span style="color:OrangeRed">__Type__</span>             : Type of Galaxy, i.e. Central or Satellite.    __Halo Central__ = 1, __Halo Satellite__ = 0.
   9. <span style="color:OrangeRed">__CZ_Real__</span>          : Velocity of the galaxy ( _without_ redshift distortions), given in _km/s_.
  10. <span style="color:OrangeRed">__Dist_central__</span>     : _Real_ Distance between Halo's center and the galaxy, in _Mpc_. Here, __Central galaxy = Halo's center__.
  11. <span style="color:OrangeRed">__Vp_total__</span>         : Total value for peculiar velocity, given in _km/s_.
  12. <span style="color:OrangeRed">__Vp_tang__</span>          : Tangential component of the peculiar velocity, given in _km/s_.
  13. <span style="color:OrangeRed">__Morphology__</span>       : Galaxy morphology. 'LT': _Late Type_ ; 'ET': _Early Type_. Used either 'goodmorph' (ECO) or 'MORPH' (RESOLVE) keys. '-9999' if no matched galaxy.
  14. <span style="color:OrangeRed">__log Mstar__</span>        : Log value of Galaxy stellar mass in log _Msun_. Used either 'rpgoodmstarsnew' (ECO) or 'MSTARS' (RESOLVE) keys in the files.
  15. <span style="color:OrangeRed"> __r-band mag__</span>       : Galaxy's r-band _apparent_ magnitude. Used either 'rpsmoothrestrmagnew' (ECO) or 'SMOOTHRESTRMAG' (RESOLVE) keys in the files.
  16. <span style="color:OrangeRed">__u-band mag__</span>       : Galaxy's u-band _apparent_ magnitude. Used either 'rpsmoothrestumagnew' (ECO) or 'SMOOTHRESTUMAG' (RESOLVE) keys in the files.
  17. <span style="color:OrangeRed">__FSMGR__</span>            : Stellar mass produced over last Gyr divided by pre-extisting Stellar mass from new model set. In _(1/Gyr)_. Used 'rpmeanssfr' (ECO) or 'MODELFSMGR' (RESOLVE) keys.
  18. <span style="color:OrangeRed">__Match_Flag__</span>       : Survey, from which the properties of the real matched galaxy were extracted. 'ECOgal' = Galaxy from ECO file. 'RESgal' = Galaxy from RESOLVE file.
  19. <span style="color:OrangeRed">__u-r color__</span>        : Color of the matched galaxy, i.e. umag - rmag (Col 15 - Col 16).
  20. <span style="color:OrangeRed">__MHI mass__</span>         : HI mass in _Msun_. Used the _predicted HI_ masses (matched to the ECO file, i.e. eco_wresa_050815.dat ) and the key 'MHI' for RESOLVE galaxies. To compute MHI masses using __ECO values__, we used the formula: __10^(MHI + logMstar)__. Units in _Msun_.
  21. <span style="color:OrangeRed">__Group ID__</span>         : Group ID, to which the galaxy belongs after running the Berlind2006 FoF group finder.
  22. <span style="color:OrangeRed">__Group NGals__</span>      : Number of galaxies in a group of galaxies.
  23. <span style="color:OrangeRed">__RG projected__</span>     : Projected radius of the group of galaxies. Units in _Mpc_.
  24. <span style="color:OrangeRed">__CZ Disp. Group__</span>   : Dispersion in velocities of galaxies in the group. Units in _km/s_.
  25. <span style="color:OrangeRed">__Abund. log MHalo__</span> : Abundance matched mass of the group of galaxy. This was calculated by assuming a monotonic relation between dark matter halo mass (MHalo) and the group _total_ luminosity. For RESOLVE B, we used a modified version of the ECO group luminosity function. Units in _Msun_.
  26. <span style="color:OrangeRed">__Group Gal. Type__</span>  : Type of group galaxy. __Group central__ = 1, __Group Satellite__ = 0.

The relationship between velocities (CZ's) is the following:

    ( CZ_Obs - CZ_Real)^2 + (Vp_tang)^2 = (Vp_total)^2

---

# <span style='color:blue'>Halos Filaments</span>
   
***Author***    : Roberto Gonzalez [ <a href="mailto:regonzar@astro.puc.cl">__regonzar@astro.puc.cl__</a> or <a href="mailto:regonzar@oddjob.uchicago.edu">__regonzar@oddjob.uchicago.edu__</a> ]

***Affil***     : The University of Chicago, Universidad Católica de Chile

---

### <span style="color:Peru">Columns</span>

   1. <span style="color:OrangeRed">__Halo ID__</span>     : This corresponds to the Halo ID number for the given DM Halo in the simulation box.
   2. <span style="color:OrangeRed">__log(MHalo)__</span>  : Logarithmic value of the DM Halo's Mass, as __log( MHalo / (Msun/h) )__ with _h_ = 1.0
   3. <span style="color:OrangeRed">__ID / Type__</span>   : This is a flag that shows what the environment of the DM halo is. There are four options for this, i.e.
      a. ID = 0 ..... Not in a filament
      b. ID = 1 ..... A filament node
      c. ID = 2 ..... Part of a filament skeleton
      d. ID = 3 ..... within a close radius of a filament
   4. <span style="color:OrangeRed">__Fil.__</span>        : The ID of the filament the halo belongs to ( -1 if it is not in a filament). 
   5. <span style="color:OrangeRed">__Fil. Quality__</span>: The quality of the filament (i.e. probability that the filament is real).
   
---


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   │
    │   │   ├── utilities_python    <- General Python scripts to make the flow of the project a little easier.
    │   │   │
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
