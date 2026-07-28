# SatGen

A semi-analytical satellite galaxy and dark matter halo generator,
introduced in [Jiang et al. (2020)](https://arxiv.org/abs/2005.05974), extended in [Green et al. (2021)](https://arxiv.org/abs/2110.13044).
See these publications for more details.  

This fork extends Sheridan Green's `TreeGen_Sub.py` and 
`SubEvo.py` scripts, encorporating elements from the original
version of SatGen (e.g., tracking stellar masses) and adding
additional features (e.g., other concentration--mass models).

SatGen's modules have detailed docstrings, but please feel 
free to contact either the original authors (Fangzhou Jiang and 
Sheridan Green) or myself (Dylan Folsom) with questions about
the model.

## Model overview

SatGen generates satellite galaxy populations for host halos of a desired 
mass and redshift. It combines halo merger trees, empirical relations for 
the galaxy--halo connection, and analytic prescriptions for tidal effects, 
dynamical friction, and ram-pressure stripping. It emulates zoom-in 
cosmological hydrodynamical simulations in certain ways and outperforms 
simulations in its statistical power and numerical resolution. 

### Modules
- `config.py` -- global variables and user controls 
- `cosmo.py` -- cosmology- and merger tree-related functions 
- `profiles.py` -- halo-density-profile classes
- `init.py` for initializing satellite properties at infall 
- `galhalo.py` -- aspects of the galaxy--halo connection
- `orbit.py` -- handles orbit evolution
- `evolve.py` -- for tidal stripping and structural evolution

## Satellite generation
The above modules provide all the infrastructure needed to generate
a system of satellites, but I have provided my implementation in a series
of three scripts. The general workflow of using SatGen is as follows:

1. **Generate Trees** -- this step produces a set of merger trees for the primary host halo, describing the initial conditions of the infalling satellites that build it.
2. **Evolve Satellites** -- integrate these initial conditions until the present day, accounting for tidal stripping and other dynamical effects.
3. **Process Output** -- the above process yields data for many timesteps, and it is often helpful to reduce the data to the properties of interest. I have provided a script of mine to do so.

Usage details of my particular implementation follow.

### Tree generation
In this file, one sets the target halo properties (mass and 
redshift), as well as the resolution of the merger trees.
This is primarily based on Sheridan Green's [TreeGen_Sub](https://github.com/shergreen/SatGen/blob/master/TreeGen_Sub.py) script,
though I have added elements of the original [TreeGen](https://github.com/shergreen/SatGen/blob/master/TreeGen.py) to populate halos
with stellar masses.

I have promoted other SatGen options to environment variables for 
ease of use with a job scheduler. Of particular interest may be:
- `SATGEN_NUM_TREES` -- the number of merger trees to generate
- `SATGEN_TREES` -- the output directory for the trees
- `SATGEN_SMHMR` -- the stellar mass--halo mass relation to use, passed to `init.Mstar`
- `SATGEN_CONC` -- the concentration--mass relation to use, either `"zhao"` for [Zhao+09](https://doi.org/10.1088/0004-637X/707/1/354) or passed to [Colossus' `concentration` module](https://bdiemer.bitbucket.io/colossus/halo_concentration.html)

### Satellite evolution
In this file, one sets options relevant to the satellite evolution.
This is primarily based on the companion to TreeGen_Sub, called [SubEvo.](https://github.com/shergreen/SatGen/blob/master/SubEvo.py)
I have added an implementation for the Milky Way's disk borrowed from 
[the SatEvo script](https://github.com/shergreen/SatGen/blob/master/SatEvo.py), and have options for setting the disk mass and shape.

As with `GenerateTrees`, I have promoted many of the SatGen options
to environment variables, including
- `SATGEN_TREES` -- as above, the location of the merger trees
- `SATGEN_OUTFOLDER` -- the location of the evolved satellite output
- `SATGEN_CONC_SCATTER` -- additional scatter (in dex) to add to the concentration--mass relation: this should be zero for the Zhao+09 model, but nonzero for a Colossus model, as the merger tree will contain only the median concentration values.

### Output processing
This file contains scripts for processing individual orbit-evolved 
files, or for processing entire directories at once. The functions 
provided reduce the data to `numpy` structured arrays to make it 
easier to use the output information.

The `read_file` function here may be of particular interest, as it
shows how one might extract data of interest from the output files.

These scripts are not as well-documented as the SatGen modules,
but they are written to be somewhat self-explanatory.