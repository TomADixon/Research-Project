# Research-Project

## Dependencies

  * numpy
  * PyMieScatt
  * scipy
  * matplotlib
  * csv

## Description

This code uses PyMieScatt to deliver models of the expected scattering intensities of spherical nanoparticles of various composition. The code allows user to choose from a variety of models and input their own parameters. The models are for homogenous nanoparticles, Core-Shell nanoparticles and Core-Shell nanoparticles that have a range of thicknesses (to simulate drug-loading). The materials are listed in the code. Users can also input their own refractive index data for any material using the csv file reader function.

Upon running this code the user is prompted to choose one of five options:
    
    Core/Shell (CS) -  This option will allow the user to choose both the core and shell material
    as well as the core diameter and up to 6 different shell thicknesses.
    The code will output the simulated spectral response for the nanoparticle.
    
    Homogenous sphere (HS) - This option allows the user to simulate two homogenous (core) nanoparticles 
    of different material and diameter. The code will output the simulated spectral responses of the nanoparticles.
    
    Gold - This option is for the improved size determination of gold nanoparticles using Nanoparticle
    Tracking Analysis (NTA). The user should obtain an approximate size of their nanoparticle through NTA
    as well as the RGB values from the camera they use. Then, by inputting a range of sizes around their
    aprroximate size along with the RGB values, which can be done by writing directly into the code, the 
    programme will use the ratios between the 3 RGB values to output an improved size determination of the 
    nanoparticle.
    
    Poly - This option also provides improved size determination but instead for Polystyrene nanoparticles.
    
    Loading - This option works in the same way as the improved size determination in principle but with
    slightly different parameters. This option allows the user to input a core-shell nanoparticle where the 
    materials and sizes of the core and shell can be changed in the code. The code will output an improved
    determination of shell thickness which is mean to be analogous to the amount of drug loading on a spherical
    nanoparticle.
