#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 11:05:17 2021

@author: tomdixon
"""

# -*- coding: utf-8 -*-
# http://pymiescatt.readthedocs.io/en/latest/forward.html
from __future__ import division 
import numpy as np
from scipy.special import jv, yv
from scipy.integrate import trapz
import warnings
import matplotlib.pyplot as plt
import csv
from scipy.interpolate import interp1d
from scipy import misc 
import PyMieScatt as py
def coerceDType(d):
  if type(d) is not np.ndarray:
    return np.array(d)
  else:
    return d

  
''' 

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

'''

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

''' Below is the refractive index and wavelength data put into arrays '''

Au_wavelength1 = [300, 310, 320, 330, 340, 350, 360, 370, 380, 390,
                  400, 410, 420, 430, 440, 450, 460, 470, 480, 490,
                  500, 510, 520, 530, 540, 550, 560, 570, 580, 590,
                  600, 610, 620, 630, 640, 650, 660, 670, 680, 690,
                  700, 710, 720, 730, 740, 750]


# data from https://refractiveindex.info/?shelf=main&book=Au&page=McPeak
Au_RI1 = [1.699838715+1.973157788j, 1.746948702+1.974499878j, 1.782941044+1.959795593j,
          1.804652963+1.931950392j,
          1.804621334+1.896796816j, 1.763712432+1.870684779j, 1.713434851+1.887670254j,
          1.689202065+1.922090949j, 1.678059067+1.948381952j, 1.672504217+1.966512231j, 
          1.665616091+1.973924254j, 1.653911114+1.975227068j, 1.63743714+1.970304643j, 
          1.612343317+1.959477219j,1.581523054+1.940313163j,1.538326354+1.910745397j
          ,1.475285274+1.872315849j,1.386263242+1.827683172j,1.252905499+1.782097694j
          ,1.064360219+1.76786953j,0.848474841+1.828280492j,0.661635502+1.964525779j,0.52912664+2.129735899j
          ,0.438041087+2.294990195j,0.372502061+2.451664289j,0.323930966+2.597158307j,0.284960267+2.738978341j
          ,0.254069233+2.871855006j,0.228936734+2.999518633j,0.207122197+3.1213458j,0.188789629+3.241703491j
          ,0.172726107+3.356759495j,0.160071695+3.465634566j,0.14636792+3.577628204j,0.135417727+3.685964997j
          ,0.125499436+3.792327795j,0.117832095+3.896137113j,0.111117685+3.999543366j,0.105744888+4.102704228j
          ,0.101509306+4.205033191j,0.098607898+4.305081128j,0.097790271+4.403812319j,0.098623269+4.499191766j
          ,0.096903666+4.59580138j,0.099463321+4.687205239j,0.098537395+4.78107829j]

#data from formula adapted from http://www.iapws.org/relguide/rindex.pdf
waterRIarray = [1.35925718, 1.35684304, 1.354697543, 1.352779787, 1.35105665, 
           1.349501, 1.3480905, 1.346806, 1.34563, 1.34455721, 
           1.3435675, 1.34265433, 1.341809, 1.3410252, 1.3402958, 
           1.339615566, 1.3389797, 1.33838, 1.337824, 1.337298, 
           1.3368022, 1.336333, 1.33589, 1.335469, 1.33507, 
           1.33469, 1.3343279, 1.333982, 1.33365, 1.333334854, 
           1.333031, 1.332739, 1.3324587, 1.33218828, 1.331927, 
           1.331675, 1.33143138, 1.331195, 1.33096586, 1.3307432, 
           1.330526655, 1.3303157736, 1.3301101642, 1.329909454, 1.329713, 
           1.32952136]

# Different water array lengths to match different wavelength array lengths

waterRIarray1 = [ 
           1.3435675, 1.34265433, 1.341809, 1.3410252, 1.3402958, 
           1.339615566, 1.3389797, 1.33838, 1.337824, 1.337298, 
           1.3368022, 1.336333, 1.33589, 1.335469, 1.33507, 
           1.33469, 1.3343279, 1.333982, 1.33365, 1.333334854, 
           1.333031, 1.332739, 1.3324587, 1.33218828, 1.331927, 
           1.331675, 1.33143138, 1.331195, 1.33096586, 1.3307432, 
           1.330526655, 1.3303157736, 1.3301101642, 1.329909454, 1.329713, 
           1.32952136]

# data from https://refractiveindex.info/?shelf=main&book=Si&page=Schinke
Si_RI = [5.049+4.2900E+00j,5.091+3.6239E+00j,5.085+3.2824E+00j,5.135+3.0935E+00j,5.245+2.9573E+00j
         ,5.423+2.9078E+00j,5.914+2.9135E+00j,6.820+2.1403E+00j,6.587+9.8399E-01j,6.025+5.0308E-01j
        ,5.623+3.2627E-01j,5.341+2.4127E-01j,5.110+1.7694E-01j,4.932+1.3766E-01j,4.790+1.1201E-01j
        ,4.673+9.5362E-02j,4.572+7.9105E-02j,4.485+7.0240E-02j,4.412+5.9817E-02j,4.349+5.3810E-02j
        ,4.289+4.8542E-02j,4.235+4.3831E-02j,4.187+3.9531E-02j,4.145+3.4804E-02j,4.103+2.9896E-02j
        ,4.073+2.8038E-02j,4.038+2.6551E-02j,4.006+2.3746E-02j,3.977+2.1896E-02j,3.954+2.0076E-02j
        ,3.931+1.8521E-02j,3.908+1.7257E-02j,3.888+1.6809E-02j,3.869+1.6268E-02j
        ,3.851+1.4693E-02j,3.835+1.4447E-02j,3.817+1.3608E-02j,3.805+1.2807E-02j,3.791+1.2045E-02j
        ,3.776+1.1317E-02j,3.765+1.0623E-02j,3.753+9.9610E-03j,3.741+9.3335E-03j,3.730+8.7312E-03j
        ,3.719+8.1618E-03j,3.712+7.6156E-03j]

# data from https://refractiveindex.info/?shelf=organic&book=cellulose&page=Sultanova
celluloseRI = [1.5106018872167624+0j, 1.5068466582312627+0j, 1.5034870525057196+0j, 1.5004679970177057+0j, 1.4977438788162774+0j, 
 1.4952766264477133+0j, 1.493034236832584+0j, 1.4909896318435447+0j, 1.489119761921104+0j, 1.4874048968864806+0j, 
 1.485828060086904+0j, 1.4843745733465306+0j, 1.483031688344414+0j, 1.4817882859658718+0j, 1.480634629528875+0j, 
 1.4795621610212146+0j, 1.4785633319085856+0j, 1.4776314619073256+0j, 1.4767606205138766+0j, 1.4759455271578172+0j, 
 1.4751814666775132+0j, 1.474464217466286+0j, 1.473789990146262+0j, 1.473155375029231+0j, 1.4725572969433425+0j, 
 1.4719929762597035+0j, 1.4714598951579287+0j, 1.4709557683351768+0j, 1.4704785174974198+0j, 1.470026249081081+0j, 
 1.4695972347426924+0j, 1.4691898942278152+0j, 1.4688027802911856+0j, 1.4684345653903725+0j, 1.4680840299170552+0j, 
 1.4677500517649569+0j, 1.4674315970626906+0j, 1.4671277119243542+0j, 1.466837515091394+0j, 1.4665601913567685+0j, 
 1.466294985677279+0j, 1.4660411978925467+0j, 1.4657981779798772+0j, 1.4655653217834466+0j, 1.4653420671641217+0j, 
 1.4651278905230043+0j]

# data from https://refractiveindex.info/?shelf=organic&book=glycerol&page=Rheims
glycerolRI = [1.47997+0j, 1.4812156002624737+0j, 1.482036162109375+0j, 1.4825266104318922+0j, 1.482760771183295+0j, 1.482796322365681+0j,
 1.4826785048010975+0j, 1.482442924151127+0j, 1.4821176814941568+0j, 1.4817250039097605+0j, 1.4812824999999998+0j, 
 1.4808041321152072+0j, 1.4803009732056087+0j, 1.4797817989318476+0j, 1.479253553035995+0j, 1.4787217146776406+0j, 
 1.4781905895490652+0j, 1.4776635414425656+0j, 1.4771431770833334+0j, 1.4766314941261633+0j, 1.47613+0j, 1.4756398075933+0j, 
 1.4751617124750533+0j, 1.4746962553448897+0j, 1.4742437726295112+0j, 1.4738044375384194+0j, 1.4733782934194086+0j, 
 1.472965280883357+0j, 1.472565259875495+0j, 1.4721780276390215+0j, 1.4718033333333334+0j, 1.4714408899228295+0j, 
 1.4710903838353433+0j, 1.4707514827955201+0j, 1.470423842163086+0j, 1.4701071100451666+0j, 1.4698009314026819+0j, 
 1.4695049513309952+0j, 1.4692188176626237+0j, 1.4689421830134493+0j, 1.4686747063723447+0j, 1.4684160543165168+0j, 
 1.468165901920439+0j, 1.467923933414397+0j, 1.4676898426389196+0j, 1.4674633333333333+0j]

# data from https://refractiveindex.info/?shelf=main&book=C&page=Song
C_RI = [2.913519+2.170646j, 2.928160+2.035773j, 2.941831+1.926873j, 2.952828+1.830988j,
     2.958325+1.743864j, 2.957043+1.664817j, 2.949142+1.594230j, 
 2.935647+1.532526j, 2.917969+1.479827j, 2.897584+1.435895j,
 2.875846+1.400198j, 2.853899+1.371995j, 2.832642+1.350430j, 2.812732+1.334604j,
 2.794610+1.323634, 2.778532+1.316690j, 2.764611+1.313025j, 2.752845+1.311980j, 
 2.743155+1.312995j, 2.735408+1.315598j, 2.729439+1.319402j,
 2.725065+1.324098j, 2.722102+1.329439j, 2.720369+1.335233j, 2.719695+1.341337j, 
 2.719922+1.347641j, 2.720906+1.354067j, 2.722520+1.360561j,
 2.724652+1.367087j, 2.727203+1.373624j, 2.730089+1.380160j, 2.733236+1.386692j,
 2.736583+1.393224j, 2.740079+1.399763j, 2.743679+1.406319j,
 2.747348+1.412903j, 2.751057+1.419528j, 2.754783+1.426208j, 2.758507+1.432955j,
 2.762214+1.439782j, 2.765893+1.446701j, 2.769538+1.453724j,
 2.773143+1.460861j, 2.776705+1.468121j, 2.780222+1.475513j, 2.783697+1.483044j]

# data from https://refractiveindex.info/?shelf=organic&book=polystyren&page=Zhang
Poly_RI = [1.62552+0j, 1.62163+0j, 1.61834+0j, 1.61546+0j, 1.61279+0j, 1.61037+0j, 1.60797+0j, 
           1.60594+0j, 1.60372+0j, 1.60201+0j, 1.60021+0j, 1.59862+0j, 1.59721+0j, 1.59587+0j, 
           1.59443+0j, 1.59338+0j, 1.59217+0j, 1.59084+0j, 1.58996+0j, 1.58898+0j, 1.5882+0j, 
           1.58734+0j, 1.58653+0j, 1.58554+0j, 1.58493+0j, 1.58426+0j, 1.58359+0j, 1.583+0j,
           1.58237+0j, 1.5818+0j, 1.58128+0j, 1.58061+0j, 1.58022+0j, 1.57972+0j, 1.57926+0j, 1.57889+0j]

Glucose_RI = [1.841937037037037, 1.7749933265188338, 1.718671728515625, 1.6710709835646724, 1.630670572670346, 
              1.596246522282382, 1.5668075217192499, 1.5415460613575886, 1.5198007757767358, 1.5010272143962176, 
              1.4847749999999997, 1.4706698648257936, 1.4583994369629938, 1.447701929389865, 1.4383570896796665, 
              1.4301789208962048, 1.4230097980639005, 1.4167156902059785, 1.4111822627314814, 1.4063116842888417, 
              1.4020199999999998, 1.3982349621097732, 1.39489423164455, 1.3919438822424133, 1.3893371510666281, 
              1.3870333925278326, 1.3849971990837149, 1.3831976601650566, 1.3816077356807266, 1.3802037248787091, 
              1.3789648148148146, 1.3778726954830693, 1.37691123093032, 1.376066177520531, 1.3753249420166014, 
              1.3746763733762823, 1.3741105831670068, 1.3736187903342945, 1.3731931867434537, 1.3728268204815248, 
              1.3725134943773425, 1.372247677589688, 1.3720244284407863, 1.3718393269463414, 1.3716884157230886, 
              1.371568148148148]

Poly_wavelength = [400, 410, 420, 430, 440, 450, 460, 470, 480, 490,
                  500, 510, 520, 530, 540, 550, 560, 570, 580, 590,
                  600, 610, 620, 630, 640, 650, 660, 670, 680, 690,
                  700, 710, 720, 730, 740, 750]

''' Arrays for the scattering intensities of the various input materials '''

# Gold - Silicon scattering intensities
Au_Si_scattering = []
Au_Si_scattering1 = []
Au_Si_scattering2 = []
Au_Si_scattering3 = []
Au_Si_scattering4 = []
Au_Si_scattering5 = []

# Gold - Silver scattering intensities
Au_Ag_scattering = []
Au_Ag_scattering1 = []
Au_Ag_scattering2 = []
Au_Ag_scattering3 = []
Au_Ag_scattering4 = []
Au_Ag_scattering5 = []

# Carbon - Silicon scattering intensities
C_Si_scattering = []
C_Si_scattering1 = []
C_Si_scattering2 = []
C_Si_scattering3 = []
C_Si_scattering4 = []
C_Si_scattering5 = []

# Polystyrene - Silicon scattering intensities
Polystyrene_Si_scattering = []
Polystyrene_Si_scattering1 = []
Polystyrene_Si_scattering2 = []
Polystyrene_Si_scattering3 = []
Polystyrene_Si_scattering4 = []
Polystyrene_Si_scattering5 = []

# Gold - Glycerol scattering intensities
Au_Glyc_scattering = []
Au_Glyc_scattering1 = []
Au_Glyc_scattering2 = []
Au_Glyc_scattering3 = []
Au_Glyc_scattering4 = []
Au_Glyc_scattering5 = []

# Gold - Glucose scattering intensities
Au_Glucose_scattering = []
Au_Glucose_scattering1 = []
Au_Glucose_scattering2 = []
Au_Glucose_scattering3 = []
Au_Glucose_scattering4 = []
Au_Glucose_scattering5 = []

# Carbon - Glycerol scattering intensities
C_Glyc_scattering = []
C_Glyc_scattering1 = []
C_Glyc_scattering2 = []
C_Glyc_scattering3 = []
C_Glyc_scattering4 = []
C_Glyc_scattering5 = []

# Polystyrene - Glycerol scattering intensities
Polystyrene_Glyc_scattering = []
Polystyrene_Glyc_scattering1 = []
Polystyrene_Glyc_scattering2 = []
Polystyrene_Glyc_scattering3 = []
Polystyrene_Glyc_scattering4 = []
Polystyrene_Glyc_scattering5 = []

# Polystyrene - Glucose scattering intensities
Polystyrene_Glucose_scattering = []
Polystyrene_Glucose_scattering1 = []
Polystyrene_Glucose_scattering2 = []
Polystyrene_Glucose_scattering3 = []
Polystyrene_Glucose_scattering4 = []
Polystyrene_Glucose_scattering5 = []

# Gold - Cellulose scattering intensities
Au_Cell_scattering = []
Au_Cell_scattering1 = []
Au_Cell_scattering2 = []
Au_Cell_scattering3 = []
Au_Cell_scattering4 = []
Au_Cell_scattering5 = []

# Carbon - Cellulose scattering intensities
C_Cell_scattering = []
C_Cell_scattering1 = []
C_Cell_scattering2 = []
C_Cell_scattering3 = []
C_Cell_scattering4 = []
C_Cell_scattering5 = []

# Polystyrene - Cellulose scattering intensities
Polystyrene_Cell_scattering = []
Polystyrene_Cell_scattering1 = []
Polystyrene_Cell_scattering2 = []
Polystyrene_Cell_scattering3 = []
Polystyrene_Cell_scattering4 = []
Polystyrene_Cell_scattering5 = []

testwavelength = []

''' CSV file reader for different Silver, Gold and Silicon refractive index data incase user
 wants to input their own RI data'''

with open('McPeaksilverdata.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        
        silver_lamb = []
        real_RIs = []
        imaginary_RIs = []
        count = 1
#        wavelength_range = list(np.arange(300,750,10))
        for row in readCSV:
            wavelength = row[0]
            real_RI = row[1]
            imaginary_RI = row[2]

            if count ==1:
                count += 1
            elif count > 1:
                wave = float(wavelength)
                if wave >= 1.3 and wave <=1.7:
                    silver_lamb.append(float(wavelength))
                    real_RIs.append(float(real_RI))
                    imaginary_RIs.append(float(imaginary_RI))
                else:
                    pass
                
        silver_m = []
        for i in range(0,len(real_RIs)):
            silver_m.append(real_RIs[i] +1j*imaginary_RIs[i])
            
        #silver_lamb = [i for i in silver_lamb]    



with open('McPeakgolddata.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        
        gold_lamb = []
        real_RIs = []
        imaginary_RIs = []
        count = 1
#        wavelength_range = list(np.arange(300,750,10))
        for row in readCSV:
            wavelength = row[0]
            real_RI = row[1]
            imaginary_RI = row[2]

            if count ==1:
                count += 1
            elif count > 1:
                wave = float(wavelength)
                if wave >= 0.4 and wave <=0.75:
                    gold_lamb.append(float(wavelength))
                    real_RIs.append(float(real_RI))
                    imaginary_RIs.append(float(imaginary_RI))
                else:
                    pass
                
        gold_m = []
        for i in range(0,len(real_RIs)):
            gold_m.append(real_RIs[i] +1j*imaginary_RIs[i])
            
        gold_lamb = [i*10**3 for i in gold_lamb]    



with open('SiO2data.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        
        SiO2_lamb = []
        real_RIs = []
        imaginary_RIs = []
        count = 1
#        wavelength_range = list(np.arange(300,750,10))
        for row in readCSV:
            wavelength = row[0]
            real_RI = row[1]
            imaginary_RI = row[2]

            if count ==1:
                count += 1
            elif count > 1:
                wave = float(wavelength)
                if wave >= 400 and wave <=750:
                    SiO2_lamb.append(wave)
                    real_RIs.append(float(real_RI))
                    imaginary_RIs.append(float(imaginary_RI))
                else:
                    pass
                
        SiO2_m = []
        for i in range(0,len(real_RIs)):
            SiO2_m.append(real_RIs[i] +1j*imaginary_RIs[i])
            
            


diameters = np.arange(50,3000, 1)
sizep = []

''' Below are the arrays for the scattering intensities of Gold nanoparticles that
    varying diameters that are inputted on line 487'''
    
intensityarray1 = []
intensityarray2 = []
intensityarray3 = []
intensityarray4 = []
intensityarray5 = []
intensityarray6 = []
intensityarray7 = []

''' Below are the arrays for the scattering intensities of Core-shell nanoparticles that
    varying core diameters and varying shell thicknesses'''

intensityarray1_1 = []
intensityarray1_2 = []
intensityarray1_3 = []
intensityarray1_4 = []
intensityarray1_5 = []
intensityarray1_6 = []
intensityarray1_7 = []
intensityarray1_8 = []
intensityarray1_9 = []
intensityarray1_10 = []
intensityarray2_1 = []
intensityarray2_2 = []
intensityarray2_3 = []
intensityarray3_1 = []
intensityarray3_2 = []
intensityarray3_3 = []
intensityarray4_1 = []
intensityarray4_2 = []
intensityarray4_3 = []
intensityarray5_1 = []
intensityarray5_2 = []
intensityarray5_3 = []
intensityarray6_1 = []
intensityarray6_2 = []
intensityarray6_3 = []
intensityarray7_1 = []
intensityarray7_2 = []
intensityarray7_3 = []



Core = 0
MyInput = '0'
Particle1 = 0
Particle2 = 0
Choice = input('Core/Shell (CS) or homogenous sphere (HS): ')
if Choice == 'CS':
    Core = input('Choose core material from either Au, C, Polystyrene: ')
    Shell = input('Choose shell material from either Si, Glycerol, Cellulose, Glucose, Ag: ')
    Core_d = input('Input core diameter in nanometers (dont write nm): ')
    dShell = input('Input shell diameter in nanometers (e.g. for shell of 1nm on a 75nm core you should input 76). You can input up to 6 shell diameters, if you dont want to input a value then input 0. Input 1: ')
    if dShell != 0:    
        dShell1 = input('Input 2: ')
        if dShell != 0: 
            dShell2 = input('Input 3: ')
            if dShell != 0: 
                dShell3 = input('Input 4: ')
                if dShell != 0: 
                    dShell4 = input('Input 5: ')
                    if dShell != 0: 
                        dShell5 = input('Input 6: ')
if Choice == 'HS':
    Particle1 = input('Choose particle 1 material: Au, Poly, SiO2: ')
    Particle2 = input('Choose particle 2 material: Au, Poly, SiO2: ')
    P1size = input('Choose particle 1 size in nanometres: ')
    P2size = input('Choose particle 2 size in nanometres: ')

if Choice == 'Mie':
    
    for diameter in diameters:
    
        wavelength = 650
        m = 0.12550+3.7923j
        qext, qsca, qabs, g, qpr, qback, qratio = py.MieQ(m, wavelength, diameter, nMedium=1.331675, asDict=False, asCrossSection=False)
        r = diameter/2
        cross = qsca
        parameter = (diameter*np.pi)/wavelength
        intensityarray1.append(cross)
        sizep.append(diameter)
    plt.plot(sizep, intensityarray1)
#    plt.xscale('log')
#    plt.yscale('log')
    plt.xlabel("Size parameter (circumference/wavelength)")
    plt.ylabel("Scattering efficiency")

if Choice == 'Gold':
    
    ds = [80, 65, 70, 79, 85, 75, 82] # Gold diameter inputs
    lb = 0.9 # Lower bound
    hb = 1.1 # Upper bound
    for d in ds:
        for m, wavelength, waterRI in zip(gold_m, gold_lamb, waterRIarray1):
        
            theta, SL, SR, SU = py.ScatteringFunction(m, wavelength, d, waterRI, minAngle=83.65, maxAngle=96.35 
                                                   , angularResolution=0.5, space='theta', angleMeasure='radians', normalization=None)
            if d == float(ds[0]):
                intensity1 = np.mean(SU)
                intensityarray1.append(intensity1)
                
            elif d == float(ds[1]):
                intensity1 = np.mean(SU)
                intensityarray2.append(intensity1)
                
            elif d == float(ds[2]):
                intensity1 = np.mean(SU)
                intensityarray3.append(intensity1)
            
            elif d == float(ds[3]):
                intensity1 = np.mean(SU)
                intensityarray4.append(intensity1)
            
            elif d == float(ds[4]):
                intensity1 = np.mean(SU)
                intensityarray5.append(intensity1)
            
            elif d == float(ds[5]):
                intensity1 = np.mean(SU)
                intensityarray6.append(intensity1)
                
            elif d == float(ds[6]):
                intensity1 = np.mean(SU)
                intensityarray7.append(intensity1)
                
    f1 = interp1d(gold_lamb, intensityarray1, kind='cubic')
    f2 = interp1d(gold_lamb, intensityarray2, kind='cubic')
    f3 = interp1d(gold_lamb, intensityarray3, kind='cubic')
    f4 = interp1d(gold_lamb, intensityarray4, kind='cubic')
    f5 = interp1d(gold_lamb, intensityarray5, kind='cubic')
    f6 = interp1d(gold_lamb, intensityarray6, kind='cubic')
    f7 = interp1d(gold_lamb, intensityarray7, kind='cubic')
    xnew = np.linspace(400, 750, num=1000, endpoint=True)
    rgbratiobr = []
    rgbratio1 = (f1(450))/(f1(632.8))
    rgbratio2 = (f2(450))/(f2(632.8))
    rgbratio3 = (f3(450))/(f3(632.8))
    rgbratio4 = (f4(450))/(f4(632.8))
    rgbratio5 = (f5(450))/(f5(632.8))
    rgbratio6 = (f6(450))/(f6(632.8))
    rgbratio7 = (f7(450))/(f7(632.8))
    rgbratiobr.append(rgbratio1)
    rgbratiobr.append(rgbratio2)
    rgbratiobr.append(rgbratio3)
    rgbratiobr.append(rgbratio4)
    rgbratiobr.append(rgbratio5)
    rgbratiobr.append(rgbratio6)
    rgbratiobr.append(rgbratio7)
    rgbratiobg = [(f1(450))/(f1(532)), (f2(450))/(f2(532)), (f3(450))/(f3(532)),
                    (f4(450))/(f4(532)), (f5(450))/(f5(532)), (f6(450))/(f6(532)), (f7(450))/(f7(532))]
    rgbratiorg = [(f1(632.8))/(f1(532)), (f2(632.8))/(f2(532)), (f3(632.8))/(f3(532)),
                    (f4(632.8))/(f4(532)), (f5(632.8))/(f5(532)), (f6(632.8))/(f6(532)),
                    (f7(632.8))/(f7(532))]
    #print(f1(450), f1(532), f1(632.8))
    #x, y, z = round(f1(450)*1E6, 0), round(f1(532)*1E6, 0), round(f1(632.8)*1E6, 0)
    Red = 100
    Blue = 120
    Green = 200
    
    A = np.array([[0.4, 0.07, 0.03], [0.05, 0.53, 0.12], [0.04, 0.06, 0.49]])
    B = np.array([Red, Green, Blue])
    
    C = np.linalg.solve(A,B)
    x__, y__, z__ = C[0], C[1], C[2]
    x__, y__, z__ = 274, 856, 184
    x_, y_, z_ = x__*(f2(450)/100), y__*(f2(450)/100), z__*(f2(450)/100)
    x, y, z = x_*1E6, y_*1E6, z_*1E6
    print("red =", C[0])
    print("green =", C[1])
    print("blue =", C[2])
    xround, yround, zround = int(x), int(y), int(z)
    print(xround,yround,zround)
    #x, y, z = f2(450), f2(532), f2(632.8)
#    B = 274
#    G = 856
#    R = 184
#    ratio = B/G/R
    bluearray = []
    greenarray = []
    redarray = []
    bluex = []
    greenx = []
    redx = []
    for i in np.arange(0.6, 1.4, 0.4):
        blue = x*i
        green = y*i
        red = z*i
        bluearray.append(blue/1E6)
        greenarray.append(green/1E6)
        redarray.append(red/1E6)
        bluex.append(450)
        greenx.append(532)
        redx.append(632.8)
    
    berrs = []
    gerrs = []
    rerrs = []
    
    for blu, gre, re in zip(bluearray, greenarray, redarray):
        
        berr = (1-(lb/1))*blu
        gerr = (1-(lb/1))*gre
        rerr = (1-(lb/1))*re
        berrs.append(berr)
        gerrs.append(gerr)
        rerrs.append(rerr)
                
#    plt.plot(xnew, f1(xnew), label='{}nm Au'.format(ds[0]), color='k')
#    plt.plot(xnew, f2(xnew), label='{}nm Au'.format(ds[1]), linestyle = '--')
#    plt.plot(xnew, f3(xnew), label='{}nm Au'.format(ds[2]), linestyle = '--')
#    plt.plot(xnew, f4(xnew), label='{}nm Au'.format(ds[3]), linestyle = '--')
#    plt.plot(xnew, f5(xnew), label='{}nm Au'.format(ds[4]), linestyle = '--')
#    plt.plot(xnew, f6(xnew), label='{}nm Au'.format(ds[5]), linestyle = '--')
#    plt.plot(xnew, f7(xnew), label='{}nm Au'.format(ds[6]), linestyle = '--')
#    plt.scatter(bluex, bluearray, color='b', s=10)
#    plt.scatter(greenx, greenarray, color ='g', s=10)
#    plt.scatter(redx, redarray, color='r', s=10)
#    plt.errorbar(bluex, bluearray, berrs, fmt='none', ecolor= 'b', capsize=3, elinewidth=1.5)
#    plt.errorbar(greenx, greenarray, gerrs, fmt='none', ecolor= 'g', capsize=3, elinewidth=1.5)
#    plt.errorbar(redx, redarray, rerrs, fmt='none', ecolor= 'r', capsize=3, elinewidth=1.5)
#    plt.xlabel('Wavelength (nm)')
#    plt.ylabel('Scattering intensity')
#    plt.legend()
#    plt.show()   
    
    rgbratiobg1 = []
    rgbratiorg1 = []
    for i in (rgbratiobg):
        rgbratiobg1.append(1/i)
    for i in (rgbratiorg):
        rgbratiorg1.append(1/i)
    
    f10 = interp1d(ds, rgbratiobr)
    f11 = interp1d(ds, rgbratiobg1)
    f12 = interp1d(ds, rgbratiorg1)
    dx = np.linspace(65, 85, num=201, endpoint=True)
    
    #plt.scatter(ds, rgbratiobr, color='purple', label='Blue/Red')
    plt.plot(dx, f10(dx), color='purple', label='Blue/Red', zorder=4)
    plt.plot(dx, f11(dx), color='c', label='Green/Blue', zorder=4)
    plt.plot(dx, f12(dx), color='y', label='Green/Red', zorder=4)
    plt.axhline(f1(450)/f1(632.8))
#    plt.axhline(0.95*(f1(450)/f1(632.8)), linestyle = '--', color='r')
#    plt.axhline(1.05*(f1(450)/f1(632.8)), linestyle = '--', color='r')
    plt.axhline(f1(532)/f1(450))
#    plt.axhline(0.95*(f1(532)/f1(450)), linestyle = '--', color='r')
#    plt.axhline(1.05*(f1(532)/f1(450)), linestyle = '--', color='r')
    plt.axhline(f1(532)/f1(632.8))
#    plt.axhline(0.95*(f1(532)/f1(632.8)), linestyle = '--', color='r')
#    plt.axhline(1.05*(f1(532)/f1(632.8)), linestyle = '--', color='r')
    plt.scatter([80, 80, 80],[f1(450)/f1(632.8), f1(532)/f1(450), f1(532)/f1(632.8)], color='k', marker='x', zorder=5)
    plt.xlabel('size(nm)', fontsize=15)
    plt.ylabel('Ratio', fontsize=15)
    #plt.title("80nm Gold particle")
    plt.legend()
    plt.show()
#    
#    ranges = []
#    uncertanties = []
#    j = 80
#    k = 80
#    for uncer in range(0, 11):
#        upy = (1+(uncer/200))*(f1(450)/f1(632.8))
#        doy = (1-(uncer/200))*(f1(450)/f1(632.8))
#        for i in dx:
#            x = f10(i)
#            if x >= 0.999*upy and x <= 1.001*upy:
#                j = i
#            elif x >= 0.999*doy and x <= 1.001*doy:
#                k = i
#        print('particle size range is', j, k)
#        ranges.append(k-j)
#        uncertanties.append(uncer)
#    
#    plt.plot(uncertanties, ranges, color='r')
    #plt.xlabel("Uncertainty (%)", fontsize=15)
    #plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
    #plt.ylabel("Size range (nm)")
    
    blue1, green1, red1 = int(round(f1(450)*1E6, 0)), int(round(f1(532)*1E6, 0)), int(round(f1(632.8)*1E6, 0))
    blue2, green2, red2 = int(round(f2(450)*1E6, 0)), int(round(f2(532)*1E6, 0)), int(round(f2(632.8)*1E6, 0))
    blue3, green3, red3 = int(round(f3(450)*1E6, 0)), int(round(f3(532)*1E6, 0)), int(round(f3(632.8)*1E6, 0))
    blue4, green4, red4 = int(round(f4(450)*1E6, 0)), int(round(f4(532)*1E6, 0)), int(round(f4(632.8)*1E6, 0))
    blue5, green5, red5 = int(round(f5(450)*1E6, 0)), int(round(f5(532)*1E6, 0)), int(round(f5(632.8)*1E6, 0))
    blue6, green6, red6 = int(round(f6(450)*1E6, 0)), int(round(f6(532)*1E6, 0)), int(round(f6(632.8)*1E6, 0))
    blue7, green7, red7 = int(round(f7(450)*1E6, 0)), int(round(f7(532)*1E6, 0)), int(round(f7(632.8)*1E6, 0))
    
    #print(blue1, green1, red1)
    
    check = []
    
    #print(lb*xround)
    #print(hb*xround)
#    if blue2 in range(int(lb*xround), int(hb*xround)) and green2 in range(int(lb*yround), int(hb*yround)) and red2 in range(int(lb*zround), int(hb*zround)):
#        print("True particle size is approximately {}nm".format(ds[1]))
#        check.append(1)
#    if blue3 in range(int(lb*xround), int(hb*xround)) and green3 in range(int(lb*yround), int(hb*yround)) and red3 in range(int(lb*zround), int(hb*zround)):
#        print("True particle size is approximately {}nm".format(ds[2]))
#        check.append(1)
#    if blue4 in range(int(lb*xround), int(hb*xround)) and green4 in range(int(lb*yround), int(hb*yround)) and red4 in range(int(lb*zround), int(hb*zround)):
#        print("True particle size is approximately {}nm".format(ds[3]))
#        check.append(1)
#    if blue5 in range(int(lb*xround), int(hb*xround)) and green5 in range(int(lb*yround), int(hb*yround)) and red5 in range(int(lb*zround), int(hb*zround)):
#        print("True particle size is approximately {}nm".format(ds[4]))
#        check.append(1)
#    if blue6 in range(int(lb*xround), int(hb*xround)) and green6 in range(int(lb*yround), int(hb*yround)) and red6 in range(int(lb*zround), int(hb*zround)):
#        print("True particle size is approximately {}nm".format(ds[5]))
#        check.append(1)
#    if blue7 in range(int(lb*xround), int(hb*xround)) and green7 in range(int(lb*yround), int(hb*yround)) and red7 in range(int(lb*zround), int(hb*zround)):
#        print("True particle size is approximately {}nm".format(ds[6]))
#        check.append(1)
#    elif len(check) == 0:
#        print("True particle size is not in figure")
#    print("Within {:.2f}% of true intensity".format((1-(lb/1))*100))
    
    for b, g, r in zip(bluearray, greenarray, redarray):
        xround, yround, zround =  b*1E6, g*1E6, r*1E6
        if blue1 in range(int(lb*xround), int(hb*xround)) and green1 in range(int(lb*yround), int(hb*yround)) and red1 in range(int(lb*zround), int(hb*zround)):
            print("True particle size is approximately {}nm".format(ds[0]))
            check.append(1)
        if blue2 in range(int(lb*xround), int(hb*xround)) and green2 in range(int(lb*yround), int(hb*yround)) and red2 in range(int(lb*zround), int(hb*zround)):
            print("True particle size is approximately {}nm".format(ds[1]))
            check.append(1)
        if blue3 in range(int(lb*xround), int(hb*xround)) and green3 in range(int(lb*yround), int(hb*yround)) and red3 in range(int(lb*zround), int(hb*zround)):
            print("True particle size is approximately {}nm".format(ds[2]))
            check.append(1)
        if blue4 in range(int(lb*xround), int(hb*xround)) and green4 in range(int(lb*yround), int(hb*yround)) and red4 in range(int(lb*zround), int(hb*zround)):
            print("True particle size is approximately {}nm".format(ds[3]))
            check.append(1)
        if blue5 in range(int(lb*xround), int(hb*xround)) and green5 in range(int(lb*yround), int(hb*yround)) and red5 in range(int(lb*zround), int(hb*zround)):
            print("True particle size is approximately {}nm".format(ds[4]))
            check.append(1)
        if blue6 in range(int(lb*xround), int(hb*xround)) and green6 in range(int(lb*yround), int(hb*yround)) and red6 in range(int(lb*zround), int(hb*zround)):
            print("True particle size is approximately {}nm".format(ds[5]))
            check.append(1)
        if blue7 in range(int(lb*xround), int(hb*xround)) and green7 in range(int(lb*yround), int(hb*yround)) and red7 in range(int(lb*zround), int(hb*zround)):
            print("True particle size is approximately {}nm".format(ds[6]))
            check.append(1)
    if len(check) == 0:
        print("True particle size is not in figure")
    print("Within {:.2f}% of true intensity".format((1-(lb/1))*100))
    
if Choice == 'Loading':
    
    ds = [80, 65, 70, 79, 68, 75, 82]
    allshells = []
    lb = 0.97
    hb = 1.03
    for d in ds:
        #print(d)
        shells = [d*0.1, d*0.2, d*0.3, d*0.4, d*0.5, d*0.6, d*0.7, d*0.8, d*0.9, d]
        for dshell in shells:
            if d == float(ds[0]):
                    allshells.append(dshell)
            #print(dshell)
            for m, m1, wavelength, waterRI in zip(gold_m, SiO2_m, gold_lamb, waterRIarray1):
                theta, SL, SR, SU = py.CoreShellScatteringFunction(m, m1, wavelength, d, d+dshell, waterRI, minAngle=83.65, maxAngle=96.35 
                                                   , angularResolution=0.5, normed=False)
                if d == float(ds[0]):
                    
                    if dshell == float(shells[0]):
                        intensity1 = np.mean(SU)
                        intensityarray1_1.append(intensity1)
                    elif dshell == float(shells[1]):
                        intensity1 = np.mean(SU)
                        intensityarray1_2.append(intensity1)
                    elif dshell == float(shells[2]):
                        intensity1 = np.mean(SU)
                        intensityarray1_3.append(intensity1)
                    elif dshell == float(shells[3]):
                        intensity1 = np.mean(SU)
                        intensityarray1_4.append(intensity1)
                    elif dshell == float(shells[4]):
                        intensity1 = np.mean(SU)
                        intensityarray1_5.append(intensity1)
                    elif dshell == float(shells[5]):
                        intensity1 = np.mean(SU)
                        intensityarray1_6.append(intensity1)
                    elif dshell == float(shells[6]):
                        intensity1 = np.mean(SU)
                        intensityarray1_7.append(intensity1)
                    elif dshell == float(shells[7]):
                        intensity1 = np.mean(SU)
                        intensityarray1_8.append(intensity1)
                    elif dshell == float(shells[8]):
                        intensity1 = np.mean(SU)
                        intensityarray1_9.append(intensity1)
                    elif dshell == float(shells[9]):
                        intensity1 = np.mean(SU)
                        intensityarray1_10.append(intensity1)
                   
#                elif d == float(ds[1]):
#                    if dshell == float(shells[0]):
#                        intensity1 = np.mean(SU)
#                        intensityarray2_1.append(intensity1)
#                    elif dshell == float(shells[1]):
#                        intensity1 = np.mean(SU)
#                        intensityarray2_2.append(intensity1)
#                    elif dshell == float(shells[2]):
#                        intensity1 = np.mean(SU)
#                        intensityarray2_3.append(intensity1)
#                    
#                elif d == float(ds[2]):
#                    if dshell == float(shells[0]):
#                        intensity1 = np.mean(SU)
#                        intensityarray3_1.append(intensity1)
#                    elif dshell == float(shells[1]):
#                        intensity1 = np.mean(SU)
#                        intensityarray3_2.append(intensity1)
#                    elif dshell == float(shells[2]):
#                        intensity1 = np.mean(SU)
#                        intensityarray3_3.append(intensity1)
#                        
#                elif d == float(ds[3]):
#                    if dshell == float(shells[0]):
#                        intensity1 = np.mean(SU)
#                        intensityarray4_1.append(intensity1)
#                    elif dshell == float(shells[1]):
#                        intensity1 = np.mean(SU)
#                        intensityarray4_2.append(intensity1)
#                    elif dshell == float(shells[2]):
#                        intensity1 = np.mean(SU)
#                        intensityarray4_3.append(intensity1)
#                            
#                elif d == float(ds[4]):
#                    if dshell == float(shells[0]):
#                        intensity1 = np.mean(SU)
#                        intensityarray5_1.append(intensity1)
#                    elif dshell == float(shells[1]):
#                        intensity1 = np.mean(SU)
#                        intensityarray5_2.append(intensity1)
#                    elif dshell == float(shells[2]):
#                        intensity1 = np.mean(SU)
#                        intensityarray5_3.append(intensity1)
#                        
#                elif d == float(ds[5]):
#                    if dshell == float(shells[0]):
#                        intensity1 = np.mean(SU)
#                        intensityarray6_1.append(intensity1)
#                    elif dshell == float(shells[1]):
#                        intensity1 = np.mean(SU)
#                        intensityarray6_2.append(intensity1)
#                    elif dshell == float(shells[2]):
#                        intensity1 = np.mean(SU)
#                        intensityarray6_3.append(intensity1)
#                
#                elif d == float(ds[6]):
#                    if dshell == float(shells[0]):
#                        intensity1 = np.mean(SU)
#                        intensityarray7_1.append(intensity1)
#                    elif dshell == float(shells[1]):
#                        intensity1 = np.mean(SU)
#                        intensityarray7_2.append(intensity1)
#                    elif dshell == float(shells[2]):
#                        intensity1 = np.mean(SU)
#                        intensityarray7_3.append(intensity1)
                
    f1_1 = interp1d(gold_lamb, intensityarray1_1, kind='cubic')
    f1_2 = interp1d(gold_lamb, intensityarray1_2, kind='cubic')
    f1_3 = interp1d(gold_lamb, intensityarray1_3, kind='cubic')
    f1_4 = interp1d(gold_lamb, intensityarray1_4, kind='cubic')
    f1_5 = interp1d(gold_lamb, intensityarray1_5, kind='cubic')
    f1_6 = interp1d(gold_lamb, intensityarray1_6, kind='cubic')
    f1_7 = interp1d(gold_lamb, intensityarray1_7, kind='cubic')
    f1_8 = interp1d(gold_lamb, intensityarray1_8, kind='cubic')
    f1_9 = interp1d(gold_lamb, intensityarray1_9, kind='cubic')
    f1_10 = interp1d(gold_lamb, intensityarray1_10, kind='cubic')
#    f2_1 = interp1d(gold_lamb, intensityarray2_1, kind='cubic')
#    f2_2 = interp1d(gold_lamb, intensityarray2_2, kind='cubic')
#    f2_3 = interp1d(gold_lamb, intensityarray2_3, kind='cubic')
#    f3_1 = interp1d(gold_lamb, intensityarray3_1, kind='cubic')
#    f3_2 = interp1d(gold_lamb, intensityarray3_2, kind='cubic')
#    f3_3 = interp1d(gold_lamb, intensityarray3_3, kind='cubic')
#    f4_1 = interp1d(gold_lamb, intensityarray4_1, kind='cubic')
#    f4_2 = interp1d(gold_lamb, intensityarray4_2, kind='cubic')
#    f4_3 = interp1d(gold_lamb, intensityarray4_3, kind='cubic')
#    f5_1 = interp1d(gold_lamb, intensityarray5_1, kind='cubic')
#    f5_2 = interp1d(gold_lamb, intensityarray5_2, kind='cubic')
#    f5_3 = interp1d(gold_lamb, intensityarray5_3, kind='cubic')
#    f6_1 = interp1d(gold_lamb, intensityarray6_1, kind='cubic')
#    f6_2 = interp1d(gold_lamb, intensityarray6_2, kind='cubic')
#    f6_3 = interp1d(gold_lamb, intensityarray6_3, kind='cubic')
#    f7_1 = interp1d(gold_lamb, intensityarray7_1, kind='cubic')
#    f7_2 = interp1d(gold_lamb, intensityarray7_2, kind='cubic')
#    f7_3 = interp1d(gold_lamb, intensityarray7_3, kind='cubic')
    
    xnew = np.linspace(400, 750, num=1000, endpoint=True)
#    rgbratiobr = []
#    rgbratio1 = (f1(450))/(f1(632.8))
#    rgbratio2 = (f2(450))/(f2(632.8))
#    rgbratio3 = (f3(450))/(f3(632.8))
#    rgbratio4 = (f4(450))/(f4(632.8))
#    rgbratio5 = (f5(450))/(f5(632.8))
#    rgbratio6 = (f6(450))/(f6(632.8))
#    rgbratio7 = (f7(450))/(f7(632.8))
#    rgbratiobr.append(rgbratio1)
#    rgbratiobr.append(rgbratio2)
#    rgbratiobr.append(rgbratio3)
#    rgbratiobr.append(rgbratio4)
#    rgbratiobr.append(rgbratio5)
#    rgbratiobr.append(rgbratio6)
#    rgbratiobr.append(rgbratio7)
#    rgbratiobr1 = [(f1_1(450))/(f1_1(632.8)), (f2_1(450))/(f2_1(632.8)), (f3_1(450))/(f3_1(632.8)),
#                    (f4_1(450))/(f4_1(632.8)), (f5_1(450))/(f5_1(632.8)), (f6_1(450))/(f6_1(632.8)), (f7_1(450))/(f7_1(632.8))]
#    rgbratiobr2 = [(f1_2(450))/(f1_2(632.8)), (f2_2(450))/(f2_2(632.8)), (f3_2(450))/(f3_2(632.8)),
#                    (f4_2(450))/(f4_2(632.8)), (f5_2(450))/(f5_2(632.8)), (f6_2(450))/(f6_2(632.8)), (f7_2(450))/(f7_2(632.8))]
#    rgbratiobr3 = [(f1_3(450))/(f1_3(632.8)), (f2_3(450))/(f2_3(632.8)), (f3_3(450))/(f3_3(632.8)),
#                    (f4_3(450))/(f4_3(632.8)), (f5_3(450))/(f5_3(632.8)), (f6_3(450))/(f6_3(632.8)), (f7_3(450))/(f7_3(632.8))]
#    rgbratiobg1 = [(f1_1(450))/(f1_1(532)), (f2_1(450))/(f2_1(532)), (f3_1(450))/(f3_1(532)),
#                    (f4_1(450))/(f4_1(532)), (f5_1(450))/(f5_1(532)), (f6_1(450))/(f6_1(532)), (f7_1(450))/(f7_1(532))]
#    rgbratiobg2 = [(f1_2(450))/(f1_2(532)), (f2_2(450))/(f2_2(532)), (f3_2(450))/(f3_2(532)),
#                    (f4_2(450))/(f4_2(532)), (f5_2(450))/(f5_2(532)), (f6_2(450))/(f6_2(532)), (f7_2(450))/(f7_2(532))]
#    rgbratiobg3 = [(f1_3(450))/(f1_3(532)), (f2_3(450))/(f2_3(532)), (f3_3(450))/(f3_3(532)),
#                    (f4_3(450))/(f4_3(532)), (f5_3(450))/(f5_3(532)), (f6_3(450))/(f6_3(532)), (f7_3(450))/(f7_3(532))]
#    rgbratiorg1 = [(f1_1(632.8))/(f1_1(532)), (f2_1(632.8))/(f2_1(532)), (f3_1(632.8))/(f3_1(532)),
#                    (f4_1(632.8))/(f4_1(532)), (f5_1(632.8))/(f5_1(532)), (f6_1(632.8))/(f6_1(532)),
#                    (f7_1(632.8))/(f7_1(532))]
#    rgbratiorg2 = [(f1_2(632.8))/(f1_2(532)), (f2_2(632.8))/(f2_2(532)), (f3_2(632.8))/(f3_2(532)),
#                    (f4_2(632.8))/(f4_2(532)), (f5_2(632.8))/(f5_2(532)), (f6_2(632.8))/(f6_2(532)),
#                    (f7_2(632.8))/(f7_2(532))]
#    rgbratiorg3 = [(f1_3(632.8))/(f1_3(532)), (f2_3(632.8))/(f2_3(532)), (f3_3(632.8))/(f3_3(532)),
#                    (f4_3(632.8))/(f4_3(532)), (f5_3(632.8))/(f5_3(532)), (f6_3(632.8))/(f6_3(532)),
#                    (f7_3(632.8))/(f7_3(532))]
    rgbratiobr_l = [(f1_1(450))/(f1_1(632.8)), (f1_2(450))/(f1_2(632.8)), (f1_3(450))/(f1_3(632.8)),
                     (f1_4(450))/(f1_4(632.8)), (f1_5(450))/(f1_5(632.8)), (f1_6(450))/(f1_6(632.8)),
                    (f1_7(450))/(f1_7(632.8)), (f1_8(450))/(f1_8(632.8)), (f1_9(450))/(f1_9(632.8)),
                    (f1_10(450))/(f1_10(632.8))]
    rgbratiobg_l = [(f1_1(450))/(f1_1(532)), (f1_2(450))/(f1_2(532)), (f1_3(450))/(f1_3(532)),
                     (f1_4(450))/(f1_4(532)), (f1_5(450))/(f1_5(532)), (f1_6(450))/(f1_6(532)),
                    (f1_7(450))/(f1_7(532)), (f1_8(450))/(f1_8(532)), (f1_9(450))/(f1_9(532)),
                    (f1_10(450))/(f1_10(532))]
    rgbratiorg_l = [(f1_1(632.8))/(f1_1(532)), (f1_2(632.8))/(f1_2(532)), (f1_3(632.8))/(f1_3(532)),
                     (f1_4(632.8))/(f1_4(532)), (f1_5(632.8))/(f1_5(532)), (f1_6(632.8))/(f1_6(532)),
                    (f1_7(632.8))/(f1_7(532)), (f1_8(632.8))/(f1_8(532)), (f1_9(632.8))/(f1_9(532)),
                    (f1_10(632.8))/(f1_10(532))]
#    print(f3_2(450), f3_2(532), f3_2(632.8))
#    #x, y, z = round(f1(450)*1E6, 0), round(f1(532)*1E6, 0), round(f1(632.8)*1E6, 0)
#    Red = 100
#    Blue = 120
#    Green = 200
#    
#    A = np.array([[0.4, 0.07, 0.03], [0.05, 0.53, 0.12], [0.04, 0.06, 0.49]])
#    B = np.array([Red, Green, Blue])
#    
#    C = np.linalg.solve(A,B)
#    x__, y__, z__ = C[0], C[1], C[2]
#    x__, y__, z__ = 197.25, 681.35, 120.8
#    x_, y_, z_ = x__*(f2_1(450)/100), y__*(f2_1(450)/100), z__*(f2_1(450)/100)
#    x, y, z = x_*1E6, y_*1E6, z_*1E6
##    print("red =", C[0])
##    print("green =", C[1])
##    print("blue =", C[2])
#    xround, yround, zround = int(x), int(y), int(z)
#    print(xround,yround,zround)
#    #x, y, z = f2(450), f2(532), f2(632.8)
##    B = 274
##    G = 856
##    R = 184
##    ratio = B/G/R
#    bluearray = []
#    greenarray = []
#    redarray = []
#    bluex = []
#    greenx = []
#    redx = []
#    for i in np.arange(0.6, 1, 0.05):
#        blue = x*i
#        green = y*i
#        red = z*i
#        bluearray.append(blue/1E6)
#        greenarray.append(green/1E6)
#        redarray.append(red/1E6)
#        bluex.append(450)
#        greenx.append(532)
#        redx.append(632.8)
#    
#    berrs = []
#    gerrs = []
#    rerrs = []
#    
#    for blu, gre, re in zip(bluearray, greenarray, redarray):
#        
#        berr = (1-(lb/1))*blu
#        gerr = (1-(lb/1))*gre
#        rerr = (1-(lb/1))*re
#        berrs.append(berr)
#        gerrs.append(gerr)
#        rerrs.append(rerr)
                
##    plt.plot(xnew, f1_1(xnew), label='{}nm Au {}% loading'.format(ds[0], (allshells[0]/ds[0])*100), linestyle = '--')
##    plt.plot(xnew, f1_2(xnew), label='{}nm Au {}% loading'.format(ds[0], (allshells[1]/ds[0])*100), linestyle = '--')
##    plt.plot(xnew, f1_3(xnew), label='{}nm Au {}% loading'.format(ds[0], (allshells[2]/ds[0])*100), linestyle = '--')
#    plt.plot(xnew, f2_1(xnew), label='{}nm Au {}% loading'.format(ds[1], (allshells[3]/ds[1])*100), linestyle = '--')
#    plt.plot(xnew, f2_2(xnew), label='{}nm Au {}% loading'.format(ds[1], (allshells[4]/ds[1])*100), linestyle = '--')
#    plt.plot(xnew, f2_3(xnew), label='{}nm Au {}% loading'.format(ds[1], (allshells[5]/ds[1])*100), linestyle = '--')
#    plt.plot(xnew, f3_1(xnew), label='{}nm Au {}% loading'.format(ds[2], (allshells[6]/ds[2])*100), linestyle = '--')
#    plt.plot(xnew, f3_2(xnew), label='{}nm Au {}% loading'.format(ds[2], (allshells[7]/ds[2])*100), color='k')
#    plt.plot(xnew, f3_3(xnew), label='{}nm Au {}% loading'.format(ds[2], (allshells[8]/ds[2])*100), linestyle = '--')
##    plt.plot(xnew, f4_1(xnew), label='{}nm Au {}% loading'.format(ds[3], (allshells[9]/ds[3])*100), linestyle = '--')
##    plt.plot(xnew, f4_2(xnew), label='{}nm Au {}% loading'.format(ds[3], (allshells[10]/ds[3])*100), linestyle = '--')
##    plt.plot(xnew, f4_3(xnew), label='{}nm Au {}% loading'.format(ds[3], (allshells[11]/ds[3])*100), linestyle = '--')
#    plt.plot(xnew, f5_1(xnew), label='{}nm Au {}% loading'.format(ds[4], (allshells[12]/ds[4])*100), linestyle = '--')
#    plt.plot(xnew, f5_2(xnew), label='{}nm Au {}% loading'.format(ds[4], (allshells[13]/ds[4])*100), linestyle = '--')
#    plt.plot(xnew, f5_3(xnew), label='{}nm Au {}% loading'.format(ds[4], (allshells[14]/ds[4])*100), linestyle = '--')
##    plt.plot(xnew, f6_1(xnew), label='{}nm Au {}nm SiO2 shell'.format(ds[5], (allshells[15]/ds[5])*100), linestyle = '--')
##    plt.plot(xnew, f6_2(xnew), label='{}nm Au {}nm SiO2 shell'.format(ds[5], (allshells[16]/ds[5])*100), linestyle = '--')
##    plt.plot(xnew, f6_3(xnew), label='{}nm Au {}nm SiO2 shell'.format(ds[5], (allshells[17]/ds[5])*100), linestyle = '--')
##    plt.plot(xnew, f7_1(xnew), label='{}nm Au {:.2f}nm SiO2 shell'.format(ds[6], (allshells[18]/ds[6])*100), linestyle = '--')
##    plt.plot(xnew, f7_2(xnew), label='{}nm Au {:.2f}nm SiO2 shell'.format(ds[6], (allshells[19]/ds[6])*100), linestyle = '--')
##    plt.plot(xnew, f7_3(xnew), label='{}nm Au {:.2f}nm SiO2 shell'.format(ds[6], (allshells[20]/ds[6])*100), linestyle = '--')
#    plt.scatter(bluex, bluearray, color='darkblue', s=10)
#    plt.scatter(greenx, greenarray, color ='darkgreen', s=10)
#    plt.scatter(redx, redarray, color='darkred', s=10)
#    plt.errorbar(bluex, bluearray, berrs, fmt='none', ecolor= 'darkblue', capsize=3, elinewidth=1.5)
#    plt.errorbar(greenx, greenarray, gerrs, fmt='none', ecolor= 'darkgreen', capsize=3, elinewidth=1.5)
#    plt.errorbar(redx, redarray, rerrs, fmt='none', ecolor= 'darkred', capsize=3, elinewidth=1.5)
#    plt.xlabel('Wavelength (nm)')
#    plt.ylabel('Scattering intensity (au.)')
#    plt.legend()
#    plt.show()   
    
    rgbratiobg1_ = []
    rgbratiobg2_ = []
    rgbratiobg3_ = []
    rgbratiorg1_ = []
    rgbratiorg2_ = []
    rgbratiorg3_ = []
    rgbratiobg_l1 = []
    rgbratiobr_l1 = []
    rgbratiorg_l1 = []
#    for i, j, k, l, m in zip(rgbratiobg1, rgbratiobg2, rgbratiobg3, rgbratiobg_l, rgbratiorg_l):
#        rgbratiobg1_.append(1/i)
#        rgbratiobg2_.append(1/j)
#        rgbratiobg3_.append(1/k)
#        rgbratiobg_l1.append(1/l)
#        rgbratiorg_l1.append(1/m)
        
    for i, j, k in zip(rgbratiobg_l, rgbratiorg_l, rgbratiobr_l):
        rgbratiobg_l1.append(1/i)
        rgbratiorg_l1.append(1/j)
        rgbratiobr_l1.append(1/k)
#    for i, j, k in zip(rgbratiorg1, rgbratiorg2, rgbratiorg3):
#        rgbratiorg1_.append(1/i)
#        rgbratiorg2_.append(1/j)
#        rgbratiorg3_.append(1/k)
    
    allshells = np.asarray(allshells).squeeze()
    rgbratiobr_l1 = np.asarray(rgbratiobr_l1).squeeze()
#    f10_1 = interp1d(ds, rgbratiobr1)
#    f10_2 = interp1d(ds, rgbratiobr2)
#    f10_3 = interp1d(ds, rgbratiobr3)
    print(allshells)
    print(rgbratiobg_l1)
    print(len(allshells))
    print(len(rgbratiobg_l1))
    f10_l = interp1d(allshells, rgbratiobr_l)
    
    allshells = np.asarray(allshells).squeeze()
    rgbratiobg_l1 = np.asarray(rgbratiobg_l1).squeeze()
#    f11_1 = interp1d(ds, rgbratiobg1_)
#    f11_2 = interp1d(ds, rgbratiobg2_)
#    f11_3 = interp1d(ds, rgbratiobg3_)
    print(len(allshells))
    print(len(rgbratiobg_l1))
    f11_l = interp1d(allshells, rgbratiobg_l1)
    
    allshells = np.asarray(allshells).squeeze()
    rgbratiorg_l = np.asarray(rgbratiorg_l).squeeze()
#    f12_1 = interp1d(ds, rgbratiorg1_)
#    f12_2 = interp1d(ds, rgbratiorg2_)
#    f12_3 = interp1d(ds, rgbratiorg3_)
    f12_l = interp1d(allshells, rgbratiorg_l1)
    
    dx = np.linspace(65, 80, num=200, endpoint=True)
    dxx = np.linspace(ds[0]*0.1, ds[0], num=200, endpoint=True)
    
    #plt.scatter(ds, rgbratiobr, color='purple', label='Blue/Red')
#    plt.plot(dx, f10_1(dx), color='purple', label='Blue/Red')
#    plt.plot(dx, f11_1(dx), color='c', label='Green/Blue')
#    plt.plot(dx, f12_1(dx), color='y', label='Green/Red')
#    plt.axhline(f3_2(450)/f3_2(632.8))
#    plt.axhline(f3_2(532)/f3_2(450))
#    plt.axhline(f3_2(532)/f3_2(632.8))
#    plt.scatter([72.6, 70.5, 72.4],[f3_2(450)/f3_2(632.8), f3_2(532)/f3_2(450), f3_2(532)/f3_2(632.8)])
#    plt.xlabel('Core size(nm)')
#    plt.ylabel('Ratio')
#    plt.title("Gold ratios with {}% loading".format(10))
#    plt.legend()
#    plt.show()
    
    plt.plot(dxx, f10_l(dxx), color='purple', label='Blue/Red')
    plt.plot(dxx, f11_l(dxx), color='c', label='Green/Blue')
    plt.plot(dxx, f12_l(dxx), color='y', label='Green/Red')
    plt.axhline(f1_4(450)/f1_4(632.8), color='k')
    plt.axhline(f1_4(532)/f1_4(450), color='k')
    plt.axhline(f1_4(532)/f1_4(632.8), color='k')
    plt.scatter([32, 32, 32],[f1_4(450)/f1_4(632.8),f1_4(532)/f1_4(450), f1_4(532)/f1_4(632.8)], color='k')
    plt.xlabel('Shell size(nm)', fontsize=15)
    plt.ylabel('Ratio',fontsize=15)
    #plt.title("80nm Polystyrene core with varying SiO2 shell thickness")
    #plt.legend()
    plt.show()
#    
#    plt.plot(dx, f10_2(dx), color='purple', label='Blue/Red')
#    plt.plot(dx, f11_2(dx), color='c', label='Green/Blue')
#    plt.plot(dx, f12_2(dx), color='y', label='Green/Red')
#    plt.axhline(f3_2(450)/f3_2(632.8))
#    plt.axhline(f3_2(532)/f3_2(450))
#    plt.axhline(f3_2(532)/f3_2(632.8))
#    plt.scatter([70, 70, 70],[f3_2(450)/f3_2(632.8), f3_2(532)/f3_2(450), f3_2(532)/f3_2(632.8)])
#    plt.xlabel('Core size(nm)')
#    plt.ylabel('Ratio')
#    plt.title("Gold ratios with {}% loading".format(20))
#    plt.legend()
#    plt.show()
#    
#    plt.plot(dx, f10_3(dx), color='purple', label='Blue/Red')
#    plt.plot(dx, f11_3(dx), color='c', label='Green/Blue')
#    plt.plot(dx, f12_3(dx), color='y', label='Green/Red')
#    plt.axhline(f3_2(450)/f3_2(632.8))
#    plt.axhline(f3_2(532)/f3_2(450))
#    plt.axhline(f3_2(532)/f3_2(632.8))
#    plt.scatter([68, 68.5, 68.1],[f3_2(450)/f3_2(632.8), f3_2(532)/f3_2(450), f3_2(532)/f3_2(632.8)])
#    plt.xlabel('Core size(nm)')
#    plt.ylabel('Ratio')
#    plt.title("Gold ratios with {}% loading".format(30))
#    plt.legend()
#    plt.show()
#    
#    
#    blue1_1, green1_1, red1_1 = int(round(f1_1(450)*1E6, 0)), int(round(f1_1(532)*1E6, 0)), int(round(f1_1(632.8)*1E6, 0))
#    blue1_2, green1_2, red1_2 = int(round(f1_2(450)*1E6, 0)), int(round(f1_2(532)*1E6, 0)), int(round(f1_2(632.8)*1E6, 0))
#    blue1_3, green1_3, red1_3 = int(round(f1_3(450)*1E6, 0)), int(round(f1_3(532)*1E6, 0)), int(round(f1_3(632.8)*1E6, 0))
#    blue2_1, green2_1, red2_1 = int(round(f2_1(450)*1E6, 0)), int(round(f2_1(532)*1E6, 0)), int(round(f2_1(632.8)*1E6, 0))
#    blue2_2, green2_2, red2_2 = int(round(f2_2(450)*1E6, 0)), int(round(f2_2(532)*1E6, 0)), int(round(f2_2(632.8)*1E6, 0))
#    blue2_3, green2_3, red2_3 = int(round(f2_3(450)*1E6, 0)), int(round(f2_3(532)*1E6, 0)), int(round(f2_3(632.8)*1E6, 0))
#    blue3_1, green3_1, red3_1 = int(round(f3_1(450)*1E6, 0)), int(round(f3_1(532)*1E6, 0)), int(round(f3_1(632.8)*1E6, 0))
#    blue3_2, green3_2, red3_2 = int(round(f3_2(450)*1E6, 0)), int(round(f3_2(532)*1E6, 0)), int(round(f3_2(632.8)*1E6, 0))
#    blue3_3, green3_3, red3_3 = int(round(f3_3(450)*1E6, 0)), int(round(f3_3(532)*1E6, 0)), int(round(f3_3(632.8)*1E6, 0))
#    blue4_1, green4_1, red4_1 = int(round(f4_1(450)*1E6, 0)), int(round(f4_1(532)*1E6, 0)), int(round(f4_1(632.8)*1E6, 0))
#    blue4_2, green4_2, red4_2 = int(round(f4_2(450)*1E6, 0)), int(round(f4_2(532)*1E6, 0)), int(round(f4_2(632.8)*1E6, 0))
#    blue4_3, green4_3, red4_3 = int(round(f4_3(450)*1E6, 0)), int(round(f4_3(532)*1E6, 0)), int(round(f4_3(632.8)*1E6, 0))
#    blue5_1, green5_1, red5_1 = int(round(f5_1(450)*1E6, 0)), int(round(f5_1(532)*1E6, 0)), int(round(f5_1(632.8)*1E6, 0))
#    blue5_2, green5_2, red5_2 = int(round(f5_2(450)*1E6, 0)), int(round(f5_2(532)*1E6, 0)), int(round(f5_2(632.8)*1E6, 0))
#    blue5_3, green5_3, red5_3 = int(round(f5_3(450)*1E6, 0)), int(round(f5_3(532)*1E6, 0)), int(round(f5_3(632.8)*1E6, 0))
#    blue6_1, green6_1, red6_1 = int(round(f6_1(450)*1E6, 0)), int(round(f6_1(532)*1E6, 0)), int(round(f6_1(632.8)*1E6, 0))
#    blue6_2, green6_2, red6_2 = int(round(f6_2(450)*1E6, 0)), int(round(f6_2(532)*1E6, 0)), int(round(f6_2(632.8)*1E6, 0))
#    blue6_3, green6_3, red6_3 = int(round(f6_3(450)*1E6, 0)), int(round(f6_3(532)*1E6, 0)), int(round(f6_3(632.8)*1E6, 0))
#    blue7_1, green7_1, red7_1 = int(round(f7_1(450)*1E6, 0)), int(round(f7_1(532)*1E6, 0)), int(round(f7_1(632.8)*1E6, 0))
#    blue7_2, green7_2, red7_2 = int(round(f7_2(450)*1E6, 0)), int(round(f7_2(532)*1E6, 0)), int(round(f7_2(632.8)*1E6, 0))
#    blue7_3, green7_3, red7_3 = int(round(f7_3(450)*1E6, 0)), int(round(f7_3(532)*1E6, 0)), int(round(f7_3(632.8)*1E6, 0))
#    
#    #print(blue1, green1, red1)
#    
#    check = []
#    
#    for b, g, r in zip(bluearray, greenarray, redarray):
#        xround, yround, zround =  b*1E6, g*1E6, r*1E6
#        if blue2_1 in range(int(lb*xround), int(hb*xround)) and green2_1 in range(int(lb*yround), int(hb*yround)) and red2_1 in range(int(lb*zround), int(hb*zround)):
#            print("True particle size is approximately {}nm with {}% loading".format((ds[1]), (allshells[3]/ds[1])*100))
#            check.append(1)
#        if blue2_2 in range(int(lb*xround), int(hb*xround)) and green2_2 in range(int(lb*yround), int(hb*yround)) and red2_2 in range(int(lb*zround), int(hb*zround)):
#            print("True particle size is approximately {}nm with {}% loading".format(ds[1], (allshells[4]/ds[1])*100))
#            check.append(1)
#        if blue2_3 in range(int(lb*xround), int(hb*xround)) and green2_3 in range(int(lb*yround), int(hb*yround)) and red2_3 in range(int(lb*zround), int(hb*zround)):
#            print("True particle size is approximately {}nm with {}% loading".format(ds[1], (allshells[5]/ds[1])*100))
#            check.append(1)
#        if blue3_1 in range(int(lb*xround), int(hb*xround)) and green3_1 in range(int(lb*yround), int(hb*yround)) and red3_1 in range(int(lb*zround), int(hb*zround)):
#            print("True particle size is approximately {}nm with {}% loading".format(ds[2], (allshells[6]/ds[2])*100))
#            check.append(1)
#        if blue3_2 in range(int(lb*xround), int(hb*xround)) and green3_2 in range(int(lb*yround), int(hb*yround)) and red3_2 in range(int(lb*zround), int(hb*zround)):
#            print("True particle size is approximately {}nm with {}% loading".format(ds[2], (allshells[7]/ds[2])*100))
#            check.append(1)
#        if blue3_3 in range(int(lb*xround), int(hb*xround)) and green3_3 in range(int(lb*yround), int(hb*yround)) and red3_3 in range(int(lb*zround), int(hb*zround)):
#            print("True particle size is approximately {}nm with {}% loading".format(ds[2], (allshells[8]/ds[2])*100))
#            check.append(1)
#        if blue4_1 in range(int(lb*xround), int(hb*xround)) and green4_1 in range(int(lb*yround), int(hb*yround)) and red4_1 in range(int(lb*zround), int(hb*zround)):
#            print("True particle size is approximately {}nm with {}% loading".format(ds[3], (allshells[9]/ds[3])*100))
#            check.append(1)
#        if blue4_2 in range(int(lb*xround), int(hb*xround)) and green4_2 in range(int(lb*yround), int(hb*yround)) and red4_2 in range(int(lb*zround), int(hb*zround)):
#            print("True particle size is approximately {}nm with {}% loading".format(ds[3], (allshells[10]/ds[3])*100))
#            check.append(1)
#        if blue4_3 in range(int(lb*xround), int(hb*xround)) and green4_3 in range(int(lb*yround), int(hb*yround)) and red4_3 in range(int(lb*zround), int(hb*zround)):
#            print("True particle size is approximately {}nm with {}% loading".format(ds[3], (allshells[11]/ds[3])*100))
#            check.append(1)
#        if blue5_1 in range(int(lb*xround), int(hb*xround)) and green5_1 in range(int(lb*yround), int(hb*yround)) and red5_1 in range(int(lb*zround), int(hb*zround)):
#            print("True particle size is approximately {}nm with {}% loading".format(ds[4], (allshells[12]/ds[4])*100))
#            check.append(1)
#        if blue5_2 in range(int(lb*xround), int(hb*xround)) and green5_2 in range(int(lb*yround), int(hb*yround)) and red5_2 in range(int(lb*zround), int(hb*zround)):
#            print("True particle size is approximately {}nm with {}% loading".format(ds[4], (allshells[13]/ds[4])*100))
#            check.append(1)
#        if blue5_3 in range(int(lb*xround), int(hb*xround)) and green5_3 in range(int(lb*yround), int(hb*yround)) and red5_3 in range(int(lb*zround), int(hb*zround)):
#            print("True particle size is approximately {}nm with {}% loading".format(ds[4], (allshells[14]/ds[4])*100))
#            check.append(1)
#        if blue6_1 in range(int(lb*xround), int(hb*xround)) and green6_1 in range(int(lb*yround), int(hb*yround)) and red6_1 in range(int(lb*zround), int(hb*zround)):
#            print("True particle size is approximately {}nm with {}% loading".format(ds[5], (allshells[15]/ds[5])*100))
#            check.append(1)
#        if blue6_2 in range(int(lb*xround), int(hb*xround)) and green6_2 in range(int(lb*yround), int(hb*yround)) and red6_2 in range(int(lb*zround), int(hb*zround)):
#            print("True particle size is approximately {}nm with {}% loading".format(ds[5], (allshells[16]/ds[5])*100))
#            check.append(1)
#        if blue6_3 in range(int(lb*xround), int(hb*xround)) and green6_3 in range(int(lb*yround), int(hb*yround)) and red6_3 in range(int(lb*zround), int(hb*zround)):
#            print("True particle size is approximately {}nm with {}% loading".format(ds[5], (allshells[17]/ds[5])*100))
#            check.append(1)
#        if blue7_1 in range(int(lb*xround), int(hb*xround)) and green7_1 in range(int(lb*yround), int(hb*yround)) and red7_1 in range(int(lb*zround), int(hb*zround)):
#            print("True particle size is approximately {}nm with {}% loading".format(ds[6], (allshells[18]/ds[6])*100))
#            check.append(1)
#        if blue7_2 in range(int(lb*xround), int(hb*xround)) and green7_2 in range(int(lb*yround), int(hb*yround)) and red7_2 in range(int(lb*zround), int(hb*zround)):
#            print("True particle size is approximately {}nm with {}% loading".format(ds[6], (allshells[19]/ds[6])*100))
#            check.append(1)
#        if blue7_3 in range(int(lb*xround), int(hb*xround)) and green7_3 in range(int(lb*yround), int(hb*yround)) and red7_3 in range(int(lb*zround), int(hb*zround)):
#            print("True particle size is approximately {}nm with {}% loading".format(ds[6], (allshells[20]/ds[6])*100))
#            check.append(1)
#    if len(check) == 0:
#        print("True particle size is not in figure")
#    print("Within {:.2f}% of true intensity".format((1-(lb/1))*100))
    
if Choice == 'Poly':
    
    ds = [80, 40, 50, 60, 70, 100, 90]
    lb = 0.8
    hb = 1.2
    for d in ds:
        for m, wavelength, waterRI in zip(Poly_RI, Poly_wavelength, waterRIarray1):
        
            theta, SL, SR, SU = py.ScatteringFunction(m, wavelength, d, waterRI, minAngle=83.65, maxAngle=96.35 
                                                   , angularResolution=0.5, space='theta', angleMeasure='radians', normalization=None)
            if d == float(ds[0]):
                intensity1 = np.mean(SU)
                intensityarray1.append(intensity1)
                
            elif d == float(ds[1]):
                intensity1 = np.mean(SU)
                intensityarray2.append(intensity1)
                
            elif d == float(ds[2]):
                intensity1 = np.mean(SU)
                intensityarray3.append(intensity1)
            
            elif d == float(ds[3]):
                intensity1 = np.mean(SU)
                intensityarray4.append(intensity1)
            
            elif d == float(ds[4]):
                intensity1 = np.mean(SU)
                intensityarray5.append(intensity1)
            
            elif d == float(ds[5]):
                intensity1 = np.mean(SU)
                intensityarray6.append(intensity1)
                
            elif d == float(ds[6]):
                intensity1 = np.mean(SU)
                intensityarray7.append(intensity1)
                
    f1 = interp1d(gold_lamb, intensityarray1, kind='cubic')
    f2 = interp1d(gold_lamb, intensityarray2, kind='cubic')
    f3 = interp1d(gold_lamb, intensityarray3, kind='cubic')
    f4 = interp1d(gold_lamb, intensityarray4, kind='cubic')
    f5 = interp1d(gold_lamb, intensityarray5, kind='cubic')
    f6 = interp1d(gold_lamb, intensityarray6, kind='cubic')
    f7 = interp1d(gold_lamb, intensityarray7, kind='cubic')
    xnew = np.linspace(400, 750, num=1000, endpoint=True)
    rgbratiobr = []
    rgbratio1 = f1(450)/f1(632.8)
    rgbratio2 = f2(450)/f2(632.8)
    rgbratio3 = f3(450)/f3(632.8)
    rgbratio4 = f4(450)/f4(632.8)
    rgbratio5 = f5(450)/f5(632.8)
    rgbratio6 = f6(450)/f6(632.8)
    rgbratio7 = f7(450)/f7(632.8)
    rgbratiobr.append(rgbratio1)
    rgbratiobr.append(rgbratio2)
    rgbratiobr.append(rgbratio3)
    rgbratiobr.append(rgbratio4)
    rgbratiobr.append(rgbratio5)
    rgbratiobr.append(rgbratio6)
    rgbratiobr.append(rgbratio7)
    rgbratiobg = [(f1(450))/(f1(532)), (f2(450))/(f2(532)), (f3(450))/(f3(532)),
                    (f4(450))/(f4(532)), (f5(450))/(f5(532)), (f6(450))/(f6(532)), (f7(450))/(f7(532))]
    rgbratiorg = [(f1(632.8))/(f1(532)), (f2(632.8))/(f2(532)), (f3(632.8))/(f3(532)),
                    (f4(632.8))/(f4(532)), (f5(632.8))/(f5(532)), (f6(632.8))/(f6(532)),
                    (f7(632.8))/(f7(532))]
    print(f1(450), f1(532), f1(632.8))
    #x, y, z = round(f1(450)*1E6, 0), round(f1(532)*1E6, 0), round(f1(632.8)*1E6, 0)
    Red = 642
    Blue = 225
    Green = 77
    
    A = np.array([[0.4, 0.07, 0.03], [0.05, 0.53, 0.12], [0.04, 0.06, 0.49]])
    B = np.array([Red, Green, Blue])
    
    C = np.linalg.solve(A,B)
    x__, y__, z__ = C[0], C[1], C[2]
    x__, y__, z__ = 642.175, 224.52, 76.7243
    x_, y_, z_ = x__*(f1(450)/100), y__*(f1(450)/100), z__*(f1(450)/100)
    x, y, z = x_*1E6, y_*1E6, z_*1E6
    print("red =", C[0])
    print("green =", C[1])
    print("blue =", C[2])
    xround, yround, zround = int(x), int(y), int(z)
    print(xround,yround,zround)
    #x, y, z = f2(450), f2(532), f2(632.8)
#    B = 274
#    G = 856
#    R = 184
#    ratio = B/G/R
    bluearray = []
    greenarray = []
    redarray = []
    bluex = []
    greenx = []
    redx = []
    for i in np.arange(0.15, 0.2, 0.2):
        blue = x*i
        green = y*i
        red = z*i
        bluearray.append(blue/1E6)
        greenarray.append(green/1E6)
        redarray.append(red/1E6)
        bluex.append(450)
        greenx.append(532)
        redx.append(632.8)
    
    berrs = []
    gerrs = []
    rerrs = []
    
    for blu, gre, re in zip(bluearray, greenarray, redarray):
        
        berr = (1-(lb/1))*blu
        gerr = (1-(lb/1))*gre
        rerr = (1-(lb/1))*re
        berrs.append(berr)
        gerrs.append(gerr)
        rerrs.append(rerr)
                
#    plt.plot(xnew, f1(xnew), label='{}nm Poly'.format(ds[0]), color='k')
#    plt.plot(xnew, f2(xnew), label='{}nm Poly'.format(ds[1]), linestyle = '--')
#    plt.plot(xnew, f3(xnew), label='{}nm Poly'.format(ds[2]), linestyle = '--')
#    plt.plot(xnew, f4(xnew), label='{}nm Poly'.format(ds[3]), linestyle = '--')
#    plt.plot(xnew, f5(xnew), label='{}nm Poly'.format(ds[4]), linestyle = '--')
#    plt.plot(xnew, f6(xnew), label='{}nm Poly'.format(ds[5]), linestyle = '--')
#    plt.plot(xnew, f7(xnew), label='{}nm Poly'.format(ds[6]), linestyle = '--')
#    plt.scatter(bluex, bluearray, color='b', s=10)
#    plt.scatter(greenx, greenarray, color ='g', s=10)
#    plt.scatter(redx, redarray, color='r', s=10)
#    plt.errorbar(bluex, bluearray, berrs, fmt='none', ecolor= 'b', capsize=3, elinewidth=1.5)
#    plt.errorbar(greenx, greenarray, gerrs, fmt='none', ecolor= 'g', capsize=3, elinewidth=1.5)
#    plt.errorbar(redx, redarray, rerrs, fmt='none', ecolor= 'r', capsize=3, elinewidth=1.5)
#    plt.xlabel('Wavelength (nm)')
#    plt.ylabel('Scattering intensity')
#    plt.legend()
#    plt.show()   
    
    rgbratiobg1 = []
    rgbratiorg1 = []
    rgbratiobr1 = []
    for i in (rgbratiobg):
        rgbratiobg1.append(1/i)
    for i in (rgbratiorg):
        rgbratiorg1.append(1/i)
    for i in (rgbratiobr):
        rgbratiobr1.append(1/i)
    
    f10 = interp1d(ds, rgbratiobr1)
    f11 = interp1d(ds, rgbratiobg1)
    f12 = interp1d(ds, rgbratiorg)
    dx = np.linspace(50, 100, num=201, endpoint=True)
    
    plt.plot(dx, f10(dx), color='purple', label='Red/Blue', zorder=4)
    plt.plot(dx, f11(dx), color='c', label='Green/Blue', zorder=4)
    plt.plot(dx, f12(dx), color='y', label='Red/Green', zorder=4)
    plt.axhline(f1(632.8)/f1(450))
#    plt.axhline(0.95*(f1(632.8)/f1(450)), linestyle = '--', color='r')
#    plt.axhline(1.05*(f1(632.8)/f1(450)), linestyle = '--', color='r')
    plt.axhline(f1(532)/f1(450))
#    plt.axhline(0.95*(f1(532)/f1(450)), linestyle = '--', color='b')
#    plt.axhline(1.05*(f1(532)/f1(450)), linestyle = '--', color='b')
    plt.axhline(f1(632.8)/f1(532))
#    plt.axhline(0.95*(f1(632.8)/f1(532)), linestyle = '--', color='g')
#    plt.axhline(1.05*(f1(632.8)/f1(532)), linestyle = '--', color='g')
    plt.scatter([80, 80, 80],[f1(632.8)/f1(450), f1(532)/f1(450), f1(632.8)/f1(532)], color='k', marker='x', zorder=5)
    plt.xlabel('size(nm)', fontsize=15)
    plt.ylabel('Ratio', fontsize=15)
#    plt.title("80nm Polystyrene particle")
    plt.legend()
    plt.show()
    
#    A = (50, f10(50))
#    B = (100, f10(100))
#    C = (50, f1(632.8)/f1(450))
#    D = (100, f1(632.8)/f1(450))
#    
#    L1 = line(A, B)
#    L2 = line(C, D)
#
#    R = intersection(L1, L2)
#    if R:
#        print("Intersection detected:", R)
#    else:
#        print("No single intersection point detected")
#        
#    ranges = []
#    uncertanties = []
#    j = 80
#    k = 80
#    for uncer in range(0, 11):
#        for i in dx:
#            x = f10(i)
#            upy = (1+(uncer/200))*(f1(632.8)/f1(450))
#            doy = (1-(uncer/200))*(f1(632.8)/f1(450))
#            if x >= 0.9995*upy and x <= 1.0005*upy:
#                j = i
#            elif x >= 0.9995*doy and x <= 1.0005*doy:
#                k = i
#        print('particle size range is', j, k)
#        ranges.append(j-k)
#        uncertanties.append(uncer)
#    
#    plt.plot(uncertanties, ranges, color='k')
#    plt.xlabel("Uncertainty (%)", fontsize=15)
#    #plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
#    plt.ylabel("Size range (nm)", fontsize=15)
    
    blue1, green1, red1 = int(round(f1(450)*1E6, 0)), int(round(f1(532)*1E6, 0)), int(round(f1(632.8)*1E6, 0))
    blue2, green2, red2 = int(round(f2(450)*1E6, 0)), int(round(f2(532)*1E6, 0)), int(round(f2(632.8)*1E6, 0))
    blue3, green3, red3 = int(round(f3(450)*1E6, 0)), int(round(f3(532)*1E6, 0)), int(round(f3(632.8)*1E6, 0))
    blue4, green4, red4 = int(round(f4(450)*1E6, 0)), int(round(f4(532)*1E6, 0)), int(round(f4(632.8)*1E6, 0))
    blue5, green5, red5 = int(round(f5(450)*1E6, 0)), int(round(f5(532)*1E6, 0)), int(round(f5(632.8)*1E6, 0))
    blue6, green6, red6 = int(round(f6(450)*1E6, 0)), int(round(f6(532)*1E6, 0)), int(round(f6(632.8)*1E6, 0))
    blue7, green7, red7 = int(round(f7(450)*1E6, 0)), int(round(f7(532)*1E6, 0)), int(round(f7(632.8)*1E6, 0))
    
    #print(blue1, green1, red1)
    
    check = []
    
    #print(lb*xround)
    #print(hb*xround)
#    if blue2 in range(int(lb*xround), int(hb*xround)) and green2 in range(int(lb*yround), int(hb*yround)) and red2 in range(int(lb*zround), int(hb*zround)):
#        print("True particle size is approximately {}nm".format(ds[1]))
#        check.append(1)
#    if blue3 in range(int(lb*xround), int(hb*xround)) and green3 in range(int(lb*yround), int(hb*yround)) and red3 in range(int(lb*zround), int(hb*zround)):
#        print("True particle size is approximately {}nm".format(ds[2]))
#        check.append(1)
#    if blue4 in range(int(lb*xround), int(hb*xround)) and green4 in range(int(lb*yround), int(hb*yround)) and red4 in range(int(lb*zround), int(hb*zround)):
#        print("True particle size is approximately {}nm".format(ds[3]))
#        check.append(1)
#    if blue5 in range(int(lb*xround), int(hb*xround)) and green5 in range(int(lb*yround), int(hb*yround)) and red5 in range(int(lb*zround), int(hb*zround)):
#        print("True particle size is approximately {}nm".format(ds[4]))
#        check.append(1)
#    if blue6 in range(int(lb*xround), int(hb*xround)) and green6 in range(int(lb*yround), int(hb*yround)) and red6 in range(int(lb*zround), int(hb*zround)):
#        print("True particle size is approximately {}nm".format(ds[5]))
#        check.append(1)
#    if blue7 in range(int(lb*xround), int(hb*xround)) and green7 in range(int(lb*yround), int(hb*yround)) and red7 in range(int(lb*zround), int(hb*zround)):
#        print("True particle size is approximately {}nm".format(ds[6]))
#        check.append(1)
#    elif len(check) == 0:
#        print("True particle size is not in figure")
#    print("Within {:.2f}% of true intensity".format((1-(lb/1))*100))
    
    for b, g, r in zip(bluearray, greenarray, redarray):
        xround, yround, zround =  b*1E6, g*1E6, r*1E6
        if blue1 in range(int(lb*xround), int(hb*xround)) and green1 in range(int(lb*yround), int(hb*yround)) and red1 in range(int(lb*zround), int(hb*zround)):
            print("True particle size is approximately {}nm".format(ds[0]))
            check.append(1)
        if blue2 in range(int(lb*xround), int(hb*xround)) and green2 in range(int(lb*yround), int(hb*yround)) and red2 in range(int(lb*zround), int(hb*zround)):
            print("True particle size is approximately {}nm".format(ds[1]))
            check.append(1)
        if blue3 in range(int(lb*xround), int(hb*xround)) and green3 in range(int(lb*yround), int(hb*yround)) and red3 in range(int(lb*zround), int(hb*zround)):
            print("True particle size is approximately {}nm".format(ds[2]))
            check.append(1)
        if blue4 in range(int(lb*xround), int(hb*xround)) and green4 in range(int(lb*yround), int(hb*yround)) and red4 in range(int(lb*zround), int(hb*zround)):
            print("True particle size is approximately {}nm".format(ds[3]))
            check.append(1)
        if blue5 in range(int(lb*xround), int(hb*xround)) and green5 in range(int(lb*yround), int(hb*yround)) and red5 in range(int(lb*zround), int(hb*zround)):
            print("True particle size is approximately {}nm".format(ds[4]))
            check.append(1)
        if blue6 in range(int(lb*xround), int(hb*xround)) and green6 in range(int(lb*yround), int(hb*yround)) and red6 in range(int(lb*zround), int(hb*zround)):
            print("True particle size is approximately {}nm".format(ds[5]))
            check.append(1)
        if blue7 in range(int(lb*xround), int(hb*xround)) and green7 in range(int(lb*yround), int(hb*yround)) and red7 in range(int(lb*zround), int(hb*zround)):
            print("True particle size is approximately {}nm".format(ds[6]))
            check.append(1)
    if len(check) == 0:
        print("True particle size is not in figure")
    print("Within {:.2f}% of true intensity".format((1-(lb/1))*100))
    
    
if Particle1 == 'Au' and Particle2 == 'Poly':
    for wavelength, m1, m2, waterRI in zip(gold_lamb, gold_m, Poly_RI, waterRIarray1):
        P1size = float(P1size)
        P2size = float(P2size)
        theta, SL, SR, SU = py.ScatteringFunction(m1, wavelength, P1size, waterRI, minAngle=83.65, maxAngle=96.35 
                                               , angularResolution=0.5, space='theta', angleMeasure='radians', normalization=None)
        theta1, SL1, SR1, SU1 = py.ScatteringFunction(m2, wavelength, P2size, waterRI, minAngle=83.65, maxAngle=96.35 
                                               , angularResolution=0.5, space='theta', angleMeasure='radians', normalization=None)
        intensity1 = np.mean(SU)
        intensity2 = np.mean(SU1)
        intensityarray1.append(intensity1)
        intensityarray2.append(intensity2)
    
    f1 = interp1d(gold_lamb, intensityarray1, kind='cubic')
    f2 = interp1d(gold_lamb, intensityarray2, kind='cubic')
    xnew = np.linspace(400, 750, num=10000, endpoint=True)
    print(f1(450), f1(532), f1(632.8))
    print(f2(450), f2(532), f2(632.8))
    x, y, z = f1(450), f1(532), f1(632.8)
    x, y, z = f2(450), f2(532), f2(632.8)
    bluearray = []
    greenarray = []
    redarray = []
    bluex = []
    greenx = []
    redx = []
    for i in np.arange(0.4, 1.2, 0.2):
        blue = x*i
        green = y*i
        red = z*i
        bluearray.append(blue)
        greenarray.append(green)
        redarray.append(red)
        bluex.append(450)
        greenx.append(532)
        redx.append(632.8)
  
    plt.axvline(450, color='b', linestyle='--')
    plt.axvline(532, color='g', linestyle='--')
    plt.axvline(632.8, color='r', linestyle='--')
    plt.plot(xnew, f1(xnew), label='{}nm Au'.format(P1size), color='k')
    plt.plot(xnew, f2(xnew), label='{}nm Polystyrene'.format(P2size), color='y')
    plt.scatter(bluex, bluearray, color='b')
    plt.scatter(greenx, greenarray, color='g')
    plt.scatter(redx, redarray, color='r')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Scattering intensity (au.)')
    plt.legend()
    plt.show()
    
if Particle1 == 'Au' and Particle2 == 'Si':
    for wavelength, m1, m2, waterRI in zip(gold_lamb, gold_m, SiO2_m, waterRIarray1):
        P1size = float(P1size)
        P2size = float(P2size)
        dshell = 10
        theta, SL, SR, SU = py.ScatteringFunction(m1, wavelength, P1size, waterRI, minAngle=83.65, maxAngle=96.35 
                                               , angularResolution=0.5, space='theta', angleMeasure='radians', normalization=None)
        theta1, SL1, SR1, SU1 = py.ScatteringFunction(m1, wavelength, P2size, waterRI, minAngle=83.65, maxAngle=96.35 
                                               , angularResolution=0.5, space='theta', angleMeasure='radians', normalization=None)
        theta2, SL2, SR2, SU2 = py.CoreShellScatteringFunction(m1, m2, wavelength, P1size, P1size+dshell, waterRI, minAngle=83.65, maxAngle=96.35 
                                                   , angularResolution=0.5, normed=False)
        intensity1 = np.mean(SU)
        intensity2 = np.mean(SU1)
        intensity3 = np.mean(SU2)
        intensityarray1.append(intensity1)
        intensityarray2.append(intensity2)
        intensityarray3.append(intensity3)
    
    f1 = interp1d(gold_lamb, intensityarray1, kind='cubic')
    f2 = interp1d(gold_lamb, intensityarray2, kind='cubic')
    f3 = interp1d(gold_lamb, intensityarray3, kind='cubic')
    xnew = np.linspace(400, 750, num=10000, endpoint=True)
    print(f1(450), f1(532), f1(632.8))
    print(f2(450), f2(532), f2(632.8))
    x, y, z = f1(450), f1(532), f1(632.8)
    x, y, z = f2(450), f2(532), f2(632.8)
    bluearray = []
    greenarray = []
    redarray = []
    bluex = []
    greenx = []
    redx = []
    for i in np.arange(0.4, 1.2, 0.2):
        blue = x*i
        green = y*i
        red = z*i
        bluearray.append(blue)
        greenarray.append(green)
        redarray.append(red)
        bluex.append(450)
        greenx.append(532)
        redx.append(632.8)
  
#    plt.axvline(450, color='b', linestyle='--')
#    plt.axvline(532, color='g', linestyle='--')
#    plt.axvline(632.8, color='r', linestyle='--')
    plt.plot(xnew, f1(xnew), label='{}nm Au'.format(P1size), color='k')
    plt.plot(xnew, f2(xnew), label='{}nm Au'.format(P2size), color='y')
    plt.plot(xnew, f3(xnew), label='{}nm Au {}nm Si shell'.format(P1size, dshell*2), color='b')
#    plt.scatter(bluex, bluearray)
#    plt.scatter(greenx, greenarray)
#    plt.scatter(redx, redarray)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Scattering intensity (au.)')
    plt.legend()
    plt.show()


if Particle1 == 'Au' and Particle2 == 'SiO2':
    for wavelength, m1, m2, waterRI in zip(gold_lamb, gold_m, SiO2_m, waterRIarray1):
        P1size = float(P1size)
        P2size = float(P2size)
        
        theta, SL, SR, SU = py.ScatteringFunction(m1, wavelength, P1size, waterRI, minAngle=83.65, maxAngle=96.35 
                                               , angularResolution=0.5, space='theta', angleMeasure='radians', normalization=None)
        theta1, SL1, SR1, SU1 = py.ScatteringFunction(m2, wavelength, P2size, waterRI, minAngle=83.65, maxAngle=96.35 
                                               , angularResolution=0.5, space='theta', angleMeasure='radians', normalization=None)
        intensity1 = np.mean(SL)
        intensity2 = np.mean(SL1)
        intensityarray1.append(intensity1)
        intensityarray2.append(intensity2)
    
    f1 = interp1d(gold_lamb, intensityarray1, kind='cubic')
    f2 = interp1d(gold_lamb, intensityarray2, kind='cubic')
    xnew = np.linspace(400, 750, num=10000, endpoint=True)
    
    #plt.axvline(450, color='b', linestyle='--')
    plt.axvline(532, color='g', linestyle='--')
    #plt.axvline(632.8, color='r', linestyle='--')
    plt.plot(xnew, f1(xnew), label='{}nm Au'.format(P1size), color='k')
    plt.plot(xnew, f2(xnew), label='{}nm SiO2'.format(P2size), color='y')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Scattering intensity')
    plt.legend()
    plt.show()

if Particle1 == 'Poly' and Particle2 == 'SiO2':
    for wavelength, m1, m2, waterRI in zip(gold_lamb, Poly_RI, SiO2_m, waterRIarray1):
        P1size = float(P1size)
        P2size = float(P2size)
        theta, SL, SR, SU = py.ScatteringFunction(m1, wavelength, P1size, waterRI, minAngle=83.65, maxAngle=96.35 
                                               , angularResolution=0.5, space='theta', angleMeasure='radians', normalization=None)
        theta1, SL1, SR1, SU1 = py.ScatteringFunction(m2, wavelength, P2size, waterRI, minAngle=83.65, maxAngle=96.35 
                                               , angularResolution=0.5, space='theta', angleMeasure='radians', normalization=None)
        intensity1 = np.mean(SL)
        intensity2 = np.mean(SL1)
        intensityarray1.append(intensity1)
        intensityarray2.append(intensity2)
    
    f1 = interp1d(gold_lamb, intensityarray1, kind='cubic')
    f2 = interp1d(gold_lamb, intensityarray2, kind='cubic')
    xnew = np.linspace(400, 750, num=10000, endpoint=True)
    
    plt.axvline(450, color='b', linestyle='--')
    plt.axvline(532, color='g', linestyle='--')
    plt.axvline(632.8, color='r', linestyle='--')
    plt.plot(xnew, f1(xnew), label='{}nm Polystyrene'.format(P1size), color='k')
    plt.plot(xnew, f2(xnew), label='{}nm SiO2'.format(P2size), color='y')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Scattering intensity')
    plt.legend()
    plt.show()
    
   
if Core == 'Au' and Shell == 'Si': 
    
    for wavelength, mCore, mShell, waterRI in zip(gold_lamb, gold_m, SiO2_m, waterRIarray1):
        
        dCore = int(Core_d)
        dShell = int(dShell)
        dShell1 = int(dShell1)
        dShell2 = int(dShell2)
        dShell3 = int(dShell3)
        dShell4 = int(dShell4)
        dShell5 = int(dShell5)
        qext, qsca, qabs, g, qpr, qback, qratio = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell,waterRI,asDict=False, asCrossSection=False)
        theta, SL, SR, SU = py.CoreShellScatteringFunction(mCore, mShell, wavelength, dCore, dShell, waterRI, minAngle=83.65, maxAngle=96.35 
                                                   , angularResolution=0.5, normed=False)
        intensity1 = np.mean(SU)
        if dShell1 != 0:
            qext1, qsca1, qabs1, g1, qpr1, qback1, qratio1 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell1,waterRI,asDict=False, asCrossSection=False)
            theta, SL, SR, SU1 = py.CoreShellScatteringFunction(mCore, mShell, wavelength, dCore, dShell1, waterRI, minAngle=83.65, maxAngle=96.35 
                                                   , angularResolution=0.5, normed=False)
            intensity2 = np.mean(SU1)
        if dShell2 != 0:
            qext2, qsca2, qabs2, g2, qpr2, qback2, qratio2 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell2,waterRI,asDict=False, asCrossSection=False)
            theta, SL, SR, SU2 = py.CoreShellScatteringFunction(mCore, mShell, wavelength, dCore, dShell2, waterRI, minAngle=83.65, maxAngle=96.35 
                                                   , angularResolution=0.5, normed=False)
            intensity3 = np.mean(SU2)
        if dShell3 != 0:
            qext3, qsca3, qabs3, g3, qpr3, qback3, qratio3 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell3,waterRI,asDict=False, asCrossSection=False)
            theta, SL, SR, SU3 = py.CoreShellScatteringFunction(mCore, mShell, wavelength, dCore, dShell3, waterRI, minAngle=83.65, maxAngle=96.35 
                                                   , angularResolution=0.5, normed=False)
            intensity4 = np.mean(SU3)
        if dShell4 != 0:
            qext4, qsca4, qabs4, g4, qpr4, qback4, qratio4 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell4,waterRI,asDict=False, asCrossSection=False)
            theta, SL, SR, SU4 = py.CoreShellScatteringFunction(mCore, mShell, wavelength, dCore, dShell4, waterRI, minAngle=83.65, maxAngle=96.35 
                                                   , angularResolution=0.5, normed=False)
            intensity5 = np.mean(SU4)
        if dShell5 != 0:
            qext5, qsca5, qabs5, g5, qpr5, qback5, qratio5 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell5,waterRI,asDict=False, asCrossSection=False)
            theta, SL, SR, SU5 = py.CoreShellScatteringFunction(mCore, mShell, wavelength, dCore, dShell5, waterRI, minAngle=83.65, maxAngle=96.35 
                                                   , angularResolution=0.5, normed=False)
            intensity6 = np.mean(SU5)
        Au_Si_scattering.append(intensity1)
        Au_Si_scattering1.append(intensity2)
        Au_Si_scattering2.append(intensity3)
        Au_Si_scattering3.append(intensity4)
        Au_Si_scattering4.append(intensity5)
        Au_Si_scattering5.append(intensity6)
        
    plt.plot(gold_lamb,Au_Si_scattering, label='{}nm shell'.format((dShell-dCore)))
    if dShell1 != 0:
        plt.plot(gold_lamb,Au_Si_scattering1, label='{}nm shell'.format(dShell1-dCore))
    if dShell2 != 0:
        plt.plot(gold_lamb,Au_Si_scattering2, label='{}nm shell'.format(dShell2-dCore))
    if dShell3 != 0:
        plt.plot(gold_lamb,Au_Si_scattering3, label='{}nm shell'.format(dShell3-dCore))
    if dShell4 != 0:
        plt.plot(gold_lamb,Au_Si_scattering4, label='{}nm shell'.format(dShell4-dCore))
    if dShell5 != 0:
        plt.plot(gold_lamb,Au_Si_scattering5, label='{}nm shell'.format(dShell5-dCore))
    plt.axvline(450)
    plt.axvline(532)
    plt.axvline(632.8)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Scattering intensity (au.)")
    plt.title("{}nm Gold core/Silicon shell nanoparticles".format(dCore))
    plt.legend()
    plt.show()
    
if Core == 'Au' and Shell == 'Ag': 
    
    for wavelength, mCore, mShell, waterRI in zip(Au_wavelength1, Au_RI1, silver_m, waterRIarray):
        
        dCore = float(Core_d)
        dShell = float(dShell)
        dShell1 = float(dShell1)
        dShell2 = float(dShell2)
        dShell3 = float(dShell3)
        dShell4 = float(dShell4)
        dShell5 = float(dShell5)
        qext, qsca, qabs, g, qpr, qback, qratio = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell,waterRI,asDict=False, asCrossSection=False)
        if dShell1 != 0:
            qext1, qsca1, qabs1, g1, qpr1, qback1, qratio1 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell1,waterRI,asDict=False, asCrossSection=False)
        if dShell2 != 0:
            qext2, qsca2, qabs2, g2, qpr2, qback2, qratio2 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell2,waterRI,asDict=False, asCrossSection=False)
        if dShell3 != 0:
            qext3, qsca3, qabs3, g3, qpr3, qback3, qratio3 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell3,waterRI,asDict=False, asCrossSection=False)
        if dShell4 != 0:
            qext4, qsca4, qabs4, g4, qpr4, qback4, qratio4 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell4,waterRI,asDict=False, asCrossSection=False)
        if dShell5 != 0:
            qext5, qsca5, qabs5, g5, qpr5, qback5, qratio5 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell5,waterRI,asDict=False, asCrossSection=False)
        Au_Ag_scattering.append(qabs)
        Au_Ag_scattering1.append(qabs1)
        Au_Ag_scattering2.append(qabs2)
        Au_Ag_scattering3.append(qabs3)
        Au_Ag_scattering4.append(qabs4)
        Au_Ag_scattering5.append(qabs5)
    f = interp1d(Au_wavelength1, Au_Ag_scattering, kind='cubic')
    f1 = interp1d(Au_wavelength1, Au_Ag_scattering1, kind='cubic')
    f2 = interp1d(Au_wavelength1, Au_Ag_scattering2, kind='cubic')
    f3 = interp1d(Au_wavelength1, Au_Ag_scattering3, kind='cubic')
    f4 = interp1d(Au_wavelength1, Au_Ag_scattering4, kind='cubic')
    f5 = interp1d(Au_wavelength1, Au_Ag_scattering5, kind='cubic')
    
    xnew = np.linspace(300, 750, num=10000, endpoint=True)


  
    plt.plot(xnew, f(xnew), label='{}nm shell'.format((dShell-dCore)))
    if dShell1 != 0:
        plt.plot(xnew, f1(xnew), label='{}nm shell'.format(dShell1-dCore))
    if dShell2 != 0:
        plt.plot(xnew, f2(xnew), label='{}nm shell'.format(dShell2-dCore))
    if dShell3 != 0:
        plt.plot(xnew, f3(xnew), label='{}nm shell'.format(dShell3-dCore))
    if dShell4 != 0:
        plt.plot(xnew, f4(xnew), label='{}nm shell'.format(dShell4-dCore))
    if dShell5 != 0:
        plt.plot(xnew, f5(xnew), label='{}nm shell'.format(dShell5-dCore))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorption efficiency")
    plt.title("Absorption efficiencies of Gold core/Silver shell nanoparticles in water ({}nm core)".format(dCore))
    plt.legend()
    plt.show()
    
if Core == 'Vacuum' and Shell == 'Au': 
    
    for wavelength, mShell in zip(gold_lamb, gold_m):
        mCore = 1+0j
        waterRI = 1
        dCore = float(Core_d)
        dShell = float(dShell)
        dShell1 = float(dShell1)
        dShell2 = float(dShell2)
        dShell3 = float(dShell3)
        dShell4 = float(dShell4)
        dShell5 = float(dShell5)
        qext, qsca, qabs, g, qpr, qback, qratio = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell,waterRI,asDict=False, asCrossSection=False)
        if dShell1 != 0:
            qext1, qsca1, qabs1, g1, qpr1, qback1, qratio1 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell1,waterRI,asDict=False, asCrossSection=False)
        if dShell2 != 0:
            qext2, qsca2, qabs2, g2, qpr2, qback2, qratio2 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell2,waterRI,asDict=False, asCrossSection=False)
        if dShell3 != 0:
            qext3, qsca3, qabs3, g3, qpr3, qback3, qratio3 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell3,waterRI,asDict=False, asCrossSection=False)
        if dShell4 != 0:
            qext4, qsca4, qabs4, g4, qpr4, qback4, qratio4 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell4,waterRI,asDict=False, asCrossSection=False)
        if dShell5 != 0:
            qext5, qsca5, qabs5, g5, qpr5, qback5, qratio5 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell5,waterRI,asDict=False, asCrossSection=False)
        Au_Ag_scattering.append(qext)
        Au_Ag_scattering1.append(qext1)
        Au_Ag_scattering2.append(qext2)
        Au_Ag_scattering3.append(qext3)
        Au_Ag_scattering4.append(qext4)
        Au_Ag_scattering5.append(qext5)
    f = interp1d(gold_lamb, Au_Ag_scattering, kind='cubic')
    f1 = interp1d(gold_lamb, Au_Ag_scattering1, kind='cubic')
    f2 = interp1d(gold_lamb, Au_Ag_scattering2, kind='cubic')
    f3 = interp1d(gold_lamb, Au_Ag_scattering3, kind='cubic')
    f4 = interp1d(gold_lamb, Au_Ag_scattering4, kind='cubic')
    f5 = interp1d(gold_lamb, Au_Ag_scattering5, kind='cubic')
    
    xnew = np.linspace(300, 1400, num=10000, endpoint=True)


  
    plt.plot(xnew, f(xnew), label='{}nm shell'.format((dShell-dCore)))
    if dShell1 != 0:
        plt.plot(xnew, f1(xnew), label='{}nm shell'.format(dShell1-dCore))
    if dShell2 != 0:
        plt.plot(xnew, f2(xnew), label='{}nm shell'.format(dShell2-dCore))
    if dShell3 != 0:
        plt.plot(xnew, f3(xnew), label='{}nm shell'.format(dShell3-dCore))
    if dShell4 != 0:
        plt.plot(xnew, f4(xnew), label='{}nm shell'.format(dShell4-dCore))
    if dShell5 != 0:
        plt.plot(xnew, f5(xnew), label='{}nm shell'.format(dShell5-dCore))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Extinction efficiency")
    plt.title("Extinction efficiencies of Vacuum core/Gold shell nanoparticles in vacuum ({}nm core)".format(dCore))
    plt.legend()
    plt.show()

if Core == 'Vacuum' and Shell == 'Au1': 
    
    shellarray = []
    for i in range(1, 100, 1):
        shellarray.append(i)
    mShell560 = 0.28496 + 2.7390j
    for Shell in shellarray:
        mCore = 1+0j
        waterRI = 1
        dCore = float(Core_d)
        dShell = float(Shell*2 + dCore)
        qext, qsca, qabs, g, qpr, qback, qratio = py.MieQCoreShell(mCore,mShell560,560,dCore,dShell,waterRI,asDict=False, asCrossSection=False)
#        qext1, qsca1, qabs1, g1, qpr1, qback1, qratio1 = MieQCoreShell(mCore,mShell,wavelength,dCore,dShell,waterRI,asDict=False, asCrossSection=False)
#        qext2, qsca2, qabs2, g2, qpr2, qback2, qratio2 = MieQCoreShell(mCore,mShell,wavelength,dCore,dShell,waterRI,asDict=False, asCrossSection=False)
       
        Au_Ag_scattering.append(qext)
#        Au_Ag_scattering1.append(qext1)
#        Au_Ag_scattering2.append(qext2)
#        Au_Ag_scattering3.append(qext3)
#        Au_Ag_scattering4.append(qext4)
#        Au_Ag_scattering5.append(qext5)
#    f = interp1d(gold_lamb, Au_Ag_scattering, kind='cubic')
#    f1 = interp1d(gold_lamb, Au_Ag_scattering1, kind='cubic')
#    f2 = interp1d(gold_lamb, Au_Ag_scattering2, kind='cubic')
#    f3 = interp1d(gold_lamb, Au_Ag_scattering3, kind='cubic')
#    f4 = interp1d(gold_lamb, Au_Ag_scattering4, kind='cubic')
#    f5 = interp1d(gold_lamb, Au_Ag_scattering5, kind='cubic')
#    
#    xnew = np.linspace(300, 1400, num=10000, endpoint=True)


  
    plt.plot(shellarray, Au_Ag_scattering)
#    if dShell1 != 0:
#        plt.plot(xnew, f1(xnew), label='{}nm shell'.format(dShell1-dCore))
#    if dShell2 != 0:
#        plt.plot(xnew, f2(xnew), label='{}nm shell'.format(dShell2-dCore))
#    if dShell3 != 0:
#        plt.plot(xnew, f3(xnew), label='{}nm shell'.format(dShell3-dCore))
#    if dShell4 != 0:
#        plt.plot(xnew, f4(xnew), label='{}nm shell'.format(dShell4-dCore))
#    if dShell5 != 0:
#        plt.plot(xnew, f5(xnew), label='{}nm shell'.format(dShell5-dCore))
    plt.xlabel("Shell thickness (nm)")
    plt.ylabel("Extinction efficiency")
    plt.title("Extinction efficiencies of Vacuum core/Gold shell nanoparticles in vacuum ({}nm core)".format(dCore))
    plt.legend()
    plt.show()

if Core == 'Ag' and Shell == 'SiO2': 
    dqsca = []
    x2 = []
    shellarray = []
    for i in np.linspace(1, 10, 1000):
        shellarray.append(i)
    mShell450 = 1.4656
    mShell532 = 1.4607
    mShell63_8 = 1.4570
    mCore = 0.040932 + 2.6758j
    mCore1 = 0.042444 + 3.4510j
    mCore2 = 0.051255 + 4.3165j
    wavelength = 450
    wavelength1 = 532
    wavelength2 = 632.8
    for Shell in shellarray:
        
        waterRI = 1.33
        dCore = float(Core_d)
        dShell = float(Shell*2 + dCore)
        qext, qsca, qabs, g, qpr, qback, qratio = py.MieQCoreShell(mCore,mShell450,wavelength,dCore,dShell,waterRI,asDict=False, asCrossSection=True)
        qext1, qsca1, qabs1, g1, qpr1, qback1, qratio1 = py.MieQCoreShell(mCore1,mShell532,wavelength1,dCore,dShell,waterRI,asDict=False, asCrossSection=True)
        qext2, qsca2, qabs2, g2, qpr2, qback2, qratio2 = py.MieQCoreShell(mCore2,mShell63_8,wavelength2,dCore,dShell,waterRI,asDict=False, asCrossSection=True)
       
        Au_Ag_scattering.append(qsca)
        Au_Ag_scattering1.append(qsca1)
        Au_Ag_scattering2.append(qsca2)
        
        x = Shell
        y = qsca
        dy = np.zeros(y.shape,np.float)
        dy[0:-1] = np.diff(y)/np.diff(x)
        dy[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])

#        dqsca.append(np.diff(qsca) / np.diff(dShell))
#        x2.append((i[:-1] + i[1:]) / 2)
#        Au_Ag_scattering3.append(qext3)
#        Au_Ag_scattering4.append(qext4)
#        Au_Ag_scattering5.append(qext5)
#    f = interp1d(gold_lamb, Au_Ag_scattering, kind='cubic')
#    f1 = interp1d(gold_lamb, Au_Ag_scattering1, kind='cubic')
#    f2 = interp1d(gold_lamb, Au_Ag_scattering2, kind='cubic')
#    f3 = interp1d(gold_lamb, Au_Ag_scattering3, kind='cubic')
#    f4 = interp1d(gold_lamb, Au_Ag_scattering4, kind='cubic')
#    f5 = interp1d(gold_lamb, Au_Ag_scattering5, kind='cubic')
#    
#    xnew = np.linspace(300, 1400, num=10000, endpoint=True)
   
        
  
    plt.plot(shellarray, Au_Ag_scattering, label='{}nm wavelength'.format(wavelength))
#    plt.plot(shellarray, dqsca, label='{}nm wavelength'.format(wavelength))
    plt.plot(x, dy)
#    plt.plot(shellarray, Au_Ag_scattering1, label='{}nm wavelength'.format(wavelength1))
#    plt.plot(shellarray, Au_Ag_scattering2, label='{}nm wavelength'.format(wavelength2))
#    if dShell1 != 0:
#        plt.plot(xnew, f1(xnew), label='{}nm shell'.format(dShell1-dCore))
#    if dShell2 != 0:
#        plt.plot(xnew, f2(xnew), label='{}nm shell'.format(dShell2-dCore))
#    if dShell3 != 0:
#        plt.plot(xnew, f3(xnew), label='{}nm shell'.format(dShell3-dCore))
#    if dShell4 != 0:
#        plt.plot(xnew, f4(xnew), label='{}nm shell'.format(dShell4-dCore))
#    if dShell5 != 0:
#        plt.plot(xnew, f5(xnew), label='{}nm shell'.format(dShell5-dCore))
    plt.xlabel("Shell thickness (nm)")
    plt.ylabel("Scattering cross section")
    plt.title("Silver core/silicon dioxide shell nanoparticles in water ({}nm core)".format(dCore))
    plt.legend()
    plt.show()

if Core == 'Vacuum' and Shell == 'Ag': 
    
    for wavelength, mShell in zip(silver_lamb, silver_m):
        mCore = 1+0j
        waterRI = 1
        dCore = float(Core_d)
        dShell = float(dShell)
        dShell1 = float(dShell1)
        dShell2 = float(dShell2)
        dShell3 = float(dShell3)
        dShell4 = float(dShell4)
        dShell5 = float(dShell5)
        qext, qsca, qabs, g, qpr, qback, qratio = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell,waterRI,asDict=False, asCrossSection=False)
        if dShell1 != 0:
            qext1, qsca1, qabs1, g1, qpr1, qback1, qratio1 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell1,waterRI,asDict=False, asCrossSection=False)
        if dShell2 != 0:
            qext2, qsca2, qabs2, g2, qpr2, qback2, qratio2 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell2,waterRI,asDict=False, asCrossSection=False)
        if dShell3 != 0:
            qext3, qsca3, qabs3, g3, qpr3, qback3, qratio3 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell3,waterRI,asDict=False, asCrossSection=False)
        if dShell4 != 0:
            qext4, qsca4, qabs4, g4, qpr4, qback4, qratio4 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell4,waterRI,asDict=False, asCrossSection=False)
        if dShell5 != 0:
            qext5, qsca5, qabs5, g5, qpr5, qback5, qratio5 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell5,waterRI,asDict=False, asCrossSection=False)
        Au_Ag_scattering.append(qext)
        Au_Ag_scattering1.append(qext1)
        Au_Ag_scattering2.append(qext2)
        Au_Ag_scattering3.append(qext3)
        Au_Ag_scattering4.append(qext4)
        Au_Ag_scattering5.append(qext5)
    f = interp1d(silver_lamb, Au_Ag_scattering, kind='cubic')
    f1 = interp1d(silver_lamb, Au_Ag_scattering1, kind='cubic')
    f2 = interp1d(silver_lamb, Au_Ag_scattering2, kind='cubic')
    f3 = interp1d(silver_lamb, Au_Ag_scattering3, kind='cubic')
    f4 = interp1d(silver_lamb, Au_Ag_scattering4, kind='cubic')
    f5 = interp1d(silver_lamb, Au_Ag_scattering5, kind='cubic')
    
    xnew = np.linspace(300, 1400, num=10000, endpoint=True)


  
    plt.plot(xnew, f(xnew), label='{}nm shell'.format((dShell-dCore)))
    if dShell1 != 0:
        plt.plot(xnew, f1(xnew), label='{}nm shell'.format(dShell1-dCore))
    if dShell2 != 0:
        plt.plot(xnew, f2(xnew), label='{}nm shell'.format(dShell2-dCore))
    if dShell3 != 0:
        plt.plot(xnew, f3(xnew), label='{}nm shell'.format(dShell3-dCore))
    if dShell4 != 0:
        plt.plot(xnew, f4(xnew), label='{}nm shell'.format(dShell4-dCore))
    if dShell5 != 0:
        plt.plot(xnew, f5(xnew), label='{}nm shell'.format(dShell5-dCore))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Extinction efficiency")
    plt.title("Extinction efficiencies of Vacuum core/Gold shell nanoparticles in vacuum ({}nm core)".format(dCore))
    plt.legend()
    plt.show()

if Core == 'Ag' and Shell == 'Test': 
    
    for wavelength, mCore in zip(silver_lamb, silver_m):
        mShell = 3.4
        dCore = float(Core_d)
        dShell = float(dShell)
    
        qext, qsca, qabs, g, qpr, qback, qratio = py.MieQCoreShell(mCore,mShell,wavelength*1000,dCore,dShell,1,asDict=False, asCrossSection=False)
        
        Au_Ag_scattering.append(qsca)
        
        
    plt.plot(silver_lamb,Au_Ag_scattering, label='{}nm shell'.format((dShell-dCore)), color='k')
    plt.xlabel("Wavelength (um)", fontsize=15)
    plt.xticks([1.3,1.4,1.5,1.6,1.7])
    plt.yticks([0,5,10,15])
    plt.ylabel("Scattering efficiency", fontsize=15)
    #plt.title("Scattering efficiencies of Silver core and n=3.4 shell nanoparticles in Vacuum ({}nm core)".format(dCore))
    #plt.legend()
    plt.show()
    
if Core == 'Ag' and Shell == 'Testcore': 
    
    for wavelength, mCore in zip(silver_lamb, silver_m):
        if 0.38 <= wavelength <= 0.58:
            dCore = float(Core_d)
            print(dCore)
            print(wavelength)
            qext, qsca, qabs, g, qpr, qback, qratio = py.MieQ(mCore,wavelength*1000,dCore,1,asDict=False, asCrossSection=False)
        
            Au_Ag_scattering.append(qsca)
            testwavelength.append(wavelength)
        
    
    plt.plot(silver_lamb,Au_Ag_scattering, label='{}nm core'.format((dCore)), color='k')
    
    plt.xlabel("Wavelength (um)", fontsize=15)
    plt.xticks([0.4,0.44,0.48,0.52,0.56,0.58])
    plt.ylabel("Scattering efficiency", fontsize=15)
    plt.yticks([2,4,6])
    #plt.title("Scattering efficiencies of Silver core nanoparticles in water ({}nm core)".format(dCore))
    #plt.legend()
    plt.show()
    
if Core == 'C' and Shell == 'Si': 
    
    for wavelength, mCore, mShell, waterRI in zip(Au_wavelength1, C_RI, Si_RI, waterRIarray):
        
        dCore = int(Core_d)
        dShell = int(dShell)
        dShell1 = int(dShell1)
        dShell2 = int(dShell2)
        dShell3 = int(dShell3)
        dShell4 = int(dShell4)
        dShell5 = int(dShell5)
        qext, qsca, qabs, g, qpr, qback, qratio = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell,waterRI,asDict=False, asCrossSection=False)
        if dShell1 != 0:
            qext1, qsca1, qabs1, g1, qpr1, qback1, qratio1 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell1,waterRI,asDict=False, asCrossSection=False)
        if dShell2 != 0:
            qext2, qsca2, qabs2, g2, qpr2, qback2, qratio2 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell2,waterRI,asDict=False, asCrossSection=False)
        if dShell3 != 0:
            qext3, qsca3, qabs3, g3, qpr3, qback3, qratio3 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell3,waterRI,asDict=False, asCrossSection=False)
        if dShell4 != 0:
            qext4, qsca4, qabs4, g4, qpr4, qback4, qratio4 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell4,waterRI,asDict=False, asCrossSection=False)
        if dShell5 != 0:
            qext5, qsca5, qabs5, g5, qpr5, qback5, qratio5 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell5,waterRI,asDict=False, asCrossSection=False)
        C_Si_scattering.append(qsca)
        C_Si_scattering1.append(qsca1)
        C_Si_scattering2.append(qsca2)
        C_Si_scattering3.append(qsca3)
        C_Si_scattering4.append(qsca4)
        C_Si_scattering5.append(qsca5)
        
    plt.plot(Au_wavelength1,C_Si_scattering, label='{}nm shell'.format((dShell-dCore)))
    if dShell1 != 0:
        plt.plot(Au_wavelength1,C_Si_scattering1, label='{}nm shell'.format(dShell1-dCore))
    if dShell2 != 0:
        plt.plot(Au_wavelength1,C_Si_scattering2, label='{}nm shell'.format(dShell2-dCore))
    if dShell3 != 0:
        plt.plot(Au_wavelength1,C_Si_scattering3, label='{}nm shell'.format(dShell3-dCore))
    if dShell4 != 0:
        plt.plot(Au_wavelength1,C_Si_scattering4, label='{}nm shell'.format(dShell4-dCore))
    if dShell5 != 0:
        plt.plot(Au_wavelength1,C_Si_scattering5, label='{}nm shell'.format(dShell5-dCore))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Scattering efficiency")
    plt.title("Scattering efficiencies of Carbon core/Silicon shell nanoparticles in water ({}nm core)".format(dCore))
    plt.legend()
    plt.show()    

if Core == 'Polystyrene' and Shell == 'Si': 
    
    for wavelength, mCore, mShell, waterRI in zip(Poly_wavelength, Poly_RI, Si_RI, waterRIarray):
        
        dCore = int(Core_d)
        dShell = int(dShell)
        dShell1 = int(dShell1)
        dShell2 = int(dShell2)
        dShell3 = int(dShell3)
        dShell4 = int(dShell4)
        dShell5 = int(dShell5)
        qext, qsca, qabs, g, qpr, qback, qratio = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell,waterRI,asDict=False, asCrossSection=False)
        if dShell1 != 0:
            qext1, qsca1, qabs1, g1, qpr1, qback1, qratio1 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell1,waterRI,asDict=False, asCrossSection=False)
        if dShell2 != 0:
            qext2, qsca2, qabs2, g2, qpr2, qback2, qratio2 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell2,waterRI,asDict=False, asCrossSection=False)
        if dShell3 != 0:
            qext3, qsca3, qabs3, g3, qpr3, qback3, qratio3 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell3,waterRI,asDict=False, asCrossSection=False)
        if dShell4 != 0:
            qext4, qsca4, qabs4, g4, qpr4, qback4, qratio4 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell4,waterRI,asDict=False, asCrossSection=False)
        if dShell5 != 0:
            qext5, qsca5, qabs5, g5, qpr5, qback5, qratio5 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell5,waterRI,asDict=False, asCrossSection=False)
        Polystyrene_Si_scattering.append(qsca)
        Polystyrene_Si_scattering1.append(qsca1)
        Polystyrene_Si_scattering2.append(qsca2)
        Polystyrene_Si_scattering3.append(qsca3)
        Polystyrene_Si_scattering4.append(qsca4)
        Polystyrene_Si_scattering5.append(qsca5)
        
    plt.plot(Poly_wavelength,Polystyrene_Si_scattering, label='{}nm shell'.format((dShell-dCore)))
    if dShell1 != 0:
        plt.plot(Poly_wavelength,Polystyrene_Si_scattering1, label='{}nm shell'.format(dShell1-dCore))
    if dShell2 != 0:
        plt.plot(Poly_wavelength,Polystyrene_Si_scattering2, label='{}nm shell'.format(dShell2-dCore))
    if dShell3 != 0:
        plt.plot(Poly_wavelength,Polystyrene_Si_scattering3, label='{}nm shell'.format(dShell3-dCore))
    if dShell4 != 0:
        plt.plot(Poly_wavelength,Polystyrene_Si_scattering4, label='{}nm shell'.format(dShell4-dCore))
    if dShell5 != 0:
        plt.plot(Poly_wavelength,Polystyrene_Si_scattering5, label='{}nm shell'.format(dShell5-dCore))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Scattering efficiency")
    plt.title("Scattering efficiencies of Polystyrene core/Silicon shell nanoparticles in water ({}nm core)".format(dCore))
    plt.legend()
    plt.show()    
    
if Core == 'Au' and Shell == 'Glycerol': 
    
    for wavelength, mCore, mShell, waterRI in zip(Au_wavelength1, Au_RI1, glycerolRI, waterRIarray):
        
        dCore = int(Core_d)
        dShell = int(dShell)
        dShell1 = int(dShell1)
        dShell2 = int(dShell2)
        dShell3 = int(dShell3)
        dShell4 = int(dShell4)
        dShell5 = int(dShell5)
        qext, qsca, qabs, g, qpr, qback, qratio = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell,waterRI,asDict=False, asCrossSection=False)
        if dShell1 != 0:
            qext1, qsca1, qabs1, g1, qpr1, qback1, qratio1 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell1,waterRI,asDict=False, asCrossSection=False)
        if dShell2 != 0:
            qext2, qsca2, qabs2, g2, qpr2, qback2, qratio2 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell2,waterRI,asDict=False, asCrossSection=False)
        if dShell3 != 0:
            qext3, qsca3, qabs3, g3, qpr3, qback3, qratio3 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell3,waterRI,asDict=False, asCrossSection=False)
        if dShell4 != 0:
            qext4, qsca4, qabs4, g4, qpr4, qback4, qratio4 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell4,waterRI,asDict=False, asCrossSection=False)
        if dShell5 != 0:
            qext5, qsca5, qabs5, g5, qpr5, qback5, qratio5 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell5,waterRI,asDict=False, asCrossSection=False)
        Au_Si_scattering.append(qsca)
        Au_Si_scattering1.append(qsca1)
        Au_Si_scattering2.append(qsca2)
        Au_Si_scattering3.append(qsca3)
        Au_Si_scattering4.append(qsca4)
        Au_Si_scattering5.append(qsca5)
        
    plt.plot(Au_wavelength1,Au_Glyc_scattering, label='{}nm shell'.format((dShell-dCore)))
    if dShell1 != 0:
        plt.plot(Au_wavelength1,Au_Glyc_scattering1, label='{}nm shell'.format(dShell1-dCore))
    if dShell2 != 0:
        plt.plot(Au_wavelength1,Au_Glyc_scattering2, label='{}nm shell'.format(dShell2-dCore))
    if dShell3 != 0:
        plt.plot(Au_wavelength1,Au_Glyc_scattering3, label='{}nm shell'.format(dShell3-dCore))
    if dShell4 != 0:
        plt.plot(Au_wavelength1,Au_Glyc_scattering4, label='{}nm shell'.format(dShell4-dCore))
    if dShell5 != 0:
        plt.plot(Au_wavelength1,Au_Glyc_scattering5, label='{}nm shell'.format(dShell5-dCore))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Scattering efficiency")
    plt.title("Scattering efficiencies of Gold core/Glycerol shell nanoparticles in water ({}nm core)".format(dCore))
    plt.legend()
    plt.show()
    
if Core == 'Au' and Shell == 'Glucose': 
    
    for wavelength, mCore, mShell, waterRI in zip(Au_wavelength1, Au_RI1, Glucose_RI, waterRIarray):
        
        dCore = int(Core_d)
        dShell = int(dShell)
        dShell1 = int(dShell1)
        dShell2 = int(dShell2)
        dShell3 = int(dShell3)
        dShell4 = int(dShell4)
        dShell5 = int(dShell5)
        qext, qsca, qabs, g, qpr, qback, qratio = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell,waterRI,asDict=False, asCrossSection=True)
        Au_Glucose_scattering.append(qsca)
        if dShell1 != 0:
            qext1, qsca1, qabs1, g1, qpr1, qback1, qratio1 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell1,waterRI,asDict=False, asCrossSection=True)
            Au_Glucose_scattering1.append(qsca1)
        if dShell2 != 0:
            qext2, qsca2, qabs2, g2, qpr2, qback2, qratio2 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell2,waterRI,asDict=False, asCrossSection=True)
            Au_Glucose_scattering2.append(qsca2)
        if dShell3 != 0:
            qext3, qsca3, qabs3, g3, qpr3, qback3, qratio3 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell3,waterRI,asDict=False, asCrossSection=True)
            Au_Glucose_scattering3.append(qsca3)
        if dShell4 != 0:
            qext4, qsca4, qabs4, g4, qpr4, qback4, qratio4 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell4,waterRI,asDict=False, asCrossSection=True)
            Au_Glucose_scattering4.append(qsca4)
        if dShell5 != 0:
            qext5, qsca5, qabs5, g5, qpr5, qback5, qratio5 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell5,waterRI,asDict=False, asCrossSection=True)
            Au_Glucose_scattering5.append(qsca5)
        
        
        
    plt.plot(Au_wavelength1,Au_Glucose_scattering, label='{}nm shell'.format((dShell-dCore)))
    if dShell1 != 0:
        plt.plot(Au_wavelength1,Au_Glucose_scattering1, label='{}nm shell'.format(dShell1-dCore))
    if dShell2 != 0:
        plt.plot(Au_wavelength1,Au_Glucose_scattering2, label='{}nm shell'.format(dShell2-dCore))
    if dShell3 != 0:
        plt.plot(Au_wavelength1,Au_Glucose_scattering3, label='{}nm shell'.format(dShell3-dCore))
    if dShell4 != 0:
        plt.plot(Au_wavelength1,Au_Glucose_scattering4, label='{}nm shell'.format(dShell4-dCore))
    if dShell5 != 0:
        plt.plot(Au_wavelength1,Au_Glucose_scattering5, label='{}nm shell'.format(dShell5-dCore))
    xpoints = [450, 532, 632]
    for p in xpoints:
        plt.axvline(p)
    plt.yscale("log")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Scattering cross-section")
    plt.title("Scattering cross-section of Gold core/Glucose shell nanoparticles in water ({}nm core)".format(dCore))
    plt.legend()
    plt.show()
    
if Core == 'C' and Shell == 'Glycerol': 
    
    for wavelength, mCore, mShell, waterRI in zip(Au_wavelength1, C_RI, glycerolRI, waterRIarray):
        
        dCore = int(Core_d)
        dShell = int(dShell)
        dShell1 = int(dShell1)
        dShell2 = int(dShell2)
        dShell3 = int(dShell3)
        dShell4 = int(dShell4)
        dShell5 = int(dShell5)
        qext, qsca, qabs, g, qpr, qback, qratio = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell,waterRI,asDict=False, asCrossSection=False)
        if dShell1 != 0:
            qext1, qsca1, qabs1, g1, qpr1, qback1, qratio1 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell1,waterRI,asDict=False, asCrossSection=False)
        if dShell2 != 0:
            qext2, qsca2, qabs2, g2, qpr2, qback2, qratio2 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell2,waterRI,asDict=False, asCrossSection=False)
        if dShell3 != 0:
            qext3, qsca3, qabs3, g3, qpr3, qback3, qratio3 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell3,waterRI,asDict=False, asCrossSection=False)
        if dShell4 != 0:
            qext4, qsca4, qabs4, g4, qpr4, qback4, qratio4 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell4,waterRI,asDict=False, asCrossSection=False)
        if dShell5 != 0:
            qext5, qsca5, qabs5, g5, qpr5, qback5, qratio5 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell5,waterRI,asDict=False, asCrossSection=False)
        C_Glyc_scattering.append(qsca)
        C_Glyc_scattering1.append(qsca1)
        C_Glyc_scattering2.append(qsca2)
        C_Glyc_scattering3.append(qsca3)
        C_Glyc_scattering4.append(qsca4)
        C_Glyc_scattering5.append(qsca5)
        
    plt.plot(Au_wavelength1,C_Glyc_scattering, label='{}nm shell'.format((dShell-dCore)))
    if dShell1 != 0:
        plt.plot(Au_wavelength1,C_Glyc_scattering1, label='{}nm shell'.format(dShell1-dCore))
    if dShell2 != 0:
        plt.plot(Au_wavelength1,C_Glyc_scattering2, label='{}nm shell'.format(dShell2-dCore))
    if dShell3 != 0:
        plt.plot(Au_wavelength1,C_Glyc_scattering3, label='{}nm shell'.format(dShell3-dCore))
    if dShell4 != 0:
        plt.plot(Au_wavelength1,C_Glyc_scattering4, label='{}nm shell'.format(dShell4-dCore))
    if dShell5 != 0:
        plt.plot(Au_wavelength1,C_Glyc_scattering5, label='{}nm shell'.format(dShell5-dCore))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Scattering efficiency")
    plt.title("Scattering efficiencies of Carbon core/Glycerol shell nanoparticles in water ({}nm core)".format(dCore))
    plt.legend()
    plt.show()  
    
if Core == 'Polystyrene' and Shell == 'Glycerol': 
    
    for wavelength, mCore, mShell, waterRI in zip(Poly_wavelength, Poly_RI, glycerolRI, waterRIarray):
        
        dCore = int(Core_d)
        dShell = int(dShell)
        dShell1 = int(dShell1)
        dShell2 = int(dShell2)
        dShell3 = int(dShell3)
        dShell4 = int(dShell4)
        dShell5 = int(dShell5)
        qext, qsca, qabs, g, qpr, qback, qratio = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell,waterRI,asDict=False, asCrossSection=False)
        if dShell1 != 0:
            qext1, qsca1, qabs1, g1, qpr1, qback1, qratio1 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell1,waterRI,asDict=False, asCrossSection=False)
        if dShell2 != 0:
            qext2, qsca2, qabs2, g2, qpr2, qback2, qratio2 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell2,waterRI,asDict=False, asCrossSection=False)
        if dShell3 != 0:
            qext3, qsca3, qabs3, g3, qpr3, qback3, qratio3 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell3,waterRI,asDict=False, asCrossSection=False)
        if dShell4 != 0:
            qext4, qsca4, qabs4, g4, qpr4, qback4, qratio4 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell4,waterRI,asDict=False, asCrossSection=False)
        if dShell5 != 0:
            qext5, qsca5, qabs5, g5, qpr5, qback5, qratio5 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell5,waterRI,asDict=False, asCrossSection=False)
        Polystyrene_Glyc_scattering.append(qsca)
        Polystyrene_Glyc_scattering1.append(qsca1)
        Polystyrene_Glyc_scattering2.append(qsca2)
        Polystyrene_Glyc_scattering3.append(qsca3)
        Polystyrene_Glyc_scattering4.append(qsca4)
        Polystyrene_Glyc_scattering5.append(qsca5)
        
    plt.plot(Poly_wavelength,Polystyrene_Glyc_scattering, label='{}nm shell'.format((dShell-dCore)))
    if dShell1 != 0:
        plt.plot(Poly_wavelength,Polystyrene_Glyc_scattering1, label='{}nm shell'.format(dShell1-dCore))
    if dShell2 != 0:
        plt.plot(Poly_wavelength,Polystyrene_Glyc_scattering2, label='{}nm shell'.format(dShell2-dCore))
    if dShell3 != 0:
        plt.plot(Poly_wavelength,Polystyrene_Glyc_scattering3, label='{}nm shell'.format(dShell3-dCore))
    if dShell4 != 0:
        plt.plot(Poly_wavelength,Polystyrene_Glyc_scattering4, label='{}nm shell'.format(dShell4-dCore))
    if dShell5 != 0:
        plt.plot(Poly_wavelength,Polystyrene_Glyc_scattering5, label='{}nm shell'.format(dShell5-dCore))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Scattering efficiency")
    plt.title("Scattering efficiencies of Polystyrene core/Glycerol shell nanoparticles in water ({}nm core)".format(dCore))
    plt.legend()
    plt.show() 
    
if Core == 'Polystyrene' and Shell == 'Glucose': 
    
    for wavelength, mCore, mShell, waterRI in zip(Poly_wavelength, Poly_RI, glycerolRI, waterRIarray1):
        
        dCore = int(Core_d)
        dShell = int(dShell)
        dShell1 = int(dShell1)
        dShell2 = int(dShell2)
        dShell3 = int(dShell3)
        dShell4 = int(dShell4)
        dShell5 = int(dShell5)
        qext, qsca, qabs, g, qpr, qback, qratio = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell,waterRI,asDict=False, asCrossSection=True)
        Polystyrene_Glucose_scattering.append(qsca)
        if dShell1 != 0:
            qext1, qsca1, qabs1, g1, qpr1, qback1, qratio1 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell1,waterRI,asDict=False, asCrossSection=True)
            Polystyrene_Glucose_scattering1.append(qsca1)
        if dShell2 != 0:
            qext2, qsca2, qabs2, g2, qpr2, qback2, qratio2 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell2,waterRI,asDict=False, asCrossSection=True)
            Polystyrene_Glucose_scattering2.append(qsca2)
        if dShell3 != 0:
            qext3, qsca3, qabs3, g3, qpr3, qback3, qratio3 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell3,waterRI,asDict=False, asCrossSection=True)
            Polystyrene_Glucose_scattering3.append(qsca3)
        if dShell4 != 0:
            qext4, qsca4, qabs4, g4, qpr4, qback4, qratio4 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell4,waterRI,asDict=False, asCrossSection=True)
            Polystyrene_Glucose_scattering4.append(qsca4)
        if dShell5 != 0:
            qext5, qsca5, qabs5, g5, qpr5, qback5, qratio5 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell5,waterRI,asDict=False, asCrossSection=True)
            Polystyrene_Glucose_scattering5.append(qsca5)
        
        
    plt.plot(Poly_wavelength,Polystyrene_Glucose_scattering, label='{}nm shell'.format((dShell-dCore)))
    if dShell1 != 0:
        plt.plot(Poly_wavelength,Polystyrene_Glucose_scattering1, label='{}nm shell'.format(dShell1-dCore))
    if dShell2 != 0:
        plt.plot(Poly_wavelength,Polystyrene_Glucose_scattering2, label='{}nm shell'.format(dShell2-dCore))
    if dShell3 != 0:
        plt.plot(Poly_wavelength,Polystyrene_Glucose_scattering3, label='{}nm shell'.format(dShell3-dCore))
    if dShell4 != 0:
        plt.plot(Poly_wavelength,Polystyrene_Glucose_scattering4, label='{}nm shell'.format(dShell4-dCore))
    if dShell5 != 0:
        plt.plot(Poly_wavelength,Polystyrene_Glucose_scattering5, label='{}nm shell'.format(dShell5-dCore))
    xpoints = [450, 532, 632]
    for p in xpoints:
        plt.axvline(p)
    plt.yscale("log")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Scattering efficiency")
    plt.title("Scattering efficiencies of Polystyrene core/Glycerol shell nanoparticles in water ({}nm core)".format(dCore))
    plt.legend()
    plt.show() 
    
if Core == 'Au' and Shell == 'Cellulose': 
    
    for wavelength, mCore, mShell, waterRI in zip(Au_wavelength1, Au_RI1, celluloseRI, waterRIarray):
        
        dCore = int(Core_d)
        dShell = int(dShell)
        dShell1 = int(dShell1)
        dShell2 = int(dShell2)
        dShell3 = int(dShell3)
        dShell4 = int(dShell4)
        dShell5 = int(dShell5)
        qext, qsca, qabs, g, qpr, qback, qratio = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell,waterRI,asDict=False, asCrossSection=False)
        if dShell1 != 0:
            qext1, qsca1, qabs1, g1, qpr1, qback1, qratio1 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell1,waterRI,asDict=False, asCrossSection=False)
        if dShell2 != 0:
            qext2, qsca2, qabs2, g2, qpr2, qback2, qratio2 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell2,waterRI,asDict=False, asCrossSection=False)
        if dShell3 != 0:
            qext3, qsca3, qabs3, g3, qpr3, qback3, qratio3 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell3,waterRI,asDict=False, asCrossSection=False)
        if dShell4 != 0:
            qext4, qsca4, qabs4, g4, qpr4, qback4, qratio4 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell4,waterRI,asDict=False, asCrossSection=False)
        if dShell5 != 0:
            qext5, qsca5, qabs5, g5, qpr5, qback5, qratio5 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell5,waterRI,asDict=False, asCrossSection=False)
        Au_Cell_scattering.append(qsca)
        Au_Cell_scattering1.append(qsca1)
        Au_Cell_scattering2.append(qsca2)
        Au_Cell_scattering3.append(qsca3)
        Au_Cell_scattering4.append(qsca4)
        Au_Cell_scattering5.append(qsca5)
        
    plt.plot(Au_wavelength1,Au_Cell_scattering, label='{}nm shell'.format((dShell-dCore)))
    if dShell1 != 0:
        plt.plot(Au_wavelength1,Au_Cell_scattering1, label='{}nm shell'.format(dShell1-dCore))
    if dShell2 != 0:
        plt.plot(Au_wavelength1,Au_Cell_scattering2, label='{}nm shell'.format(dShell2-dCore))
    if dShell3 != 0:
        plt.plot(Au_wavelength1,Au_Cell_scattering3, label='{}nm shell'.format(dShell3-dCore))
    if dShell4 != 0:
        plt.plot(Au_wavelength1,Au_Cell_scattering4, label='{}nm shell'.format(dShell4-dCore))
    if dShell5 != 0:
        plt.plot(Au_wavelength1,Au_Cell_scattering5, label='{}nm shell'.format(dShell5-dCore))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Scattering efficiency")
    plt.title("Scattering efficiencies of Gold core/Cellulose shell nanoparticles in water ({}nm core)".format(dCore))
    plt.legend()
    plt.show()
    
if Core == 'C' and Shell == 'Cellulose': 
    
    for wavelength, mCore, mShell, waterRI in zip(Au_wavelength1, C_RI, celluloseRI, waterRIarray):
        
        dCore = int(Core_d)
        dShell = int(dShell)
        dShell1 = int(dShell1)
        dShell2 = int(dShell2)
        dShell3 = int(dShell3)
        dShell4 = int(dShell4)
        dShell5 = int(dShell5)
        qext, qsca, qabs, g, qpr, qback, qratio = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell,waterRI,asDict=False, asCrossSection=False)
        if dShell1 != 0:
            qext1, qsca1, qabs1, g1, qpr1, qback1, qratio1 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell1,waterRI,asDict=False, asCrossSection=False)
        if dShell2 != 0:
            qext2, qsca2, qabs2, g2, qpr2, qback2, qratio2 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell2,waterRI,asDict=False, asCrossSection=False)
        if dShell3 != 0:
            qext3, qsca3, qabs3, g3, qpr3, qback3, qratio3 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell3,waterRI,asDict=False, asCrossSection=False)
        if dShell4 != 0:
            qext4, qsca4, qabs4, g4, qpr4, qback4, qratio4 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell4,waterRI,asDict=False, asCrossSection=False)
        if dShell5 != 0:
            qext5, qsca5, qabs5, g5, qpr5, qback5, qratio5 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell5,waterRI,asDict=False, asCrossSection=False)
        C_Cell_scattering.append(qsca)
        if dShell1 != 0:
            C_Cell_scattering1.append(qsca1)
        if dShell2 != 0:
            C_Cell_scattering2.append(qsca2)
        if dShell3 != 0:
            C_Cell_scattering3.append(qsca3)
        if dShell4 != 0:
            C_Cell_scattering4.append(qsca4)
        if dShell5 != 0:
            C_Cell_scattering5.append(qsca5)
        
    plt.plot(Au_wavelength1,C_Cell_scattering, label='{}nm shell'.format((dShell-dCore)))
    plt.plot(Au_wavelength1,C_Cell_scattering1, label='{}nm shell'.format(dShell1-dCore))
    plt.plot(Au_wavelength1,C_Cell_scattering2, label='{}nm shell'.format(dShell2-dCore))
    plt.plot(Au_wavelength1,C_Cell_scattering3, label='{}nm shell'.format(dShell3-dCore))
    plt.plot(Au_wavelength1,C_Cell_scattering4, label='{}nm shell'.format(dShell4-dCore))
    plt.plot(Au_wavelength1,C_Cell_scattering5, label='{}nm shell'.format(dShell5-dCore))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Scattering efficiency")
    plt.title("Scattering efficiencies of Carbon core/Cellulose shell nanoparticles in water ({}nm core)".format(dCore))
    plt.legend()
    plt.show()    

if Core == 'Polystyrene' and Shell == 'Cellulose': 
    
    for wavelength, mCore, mShell, waterRI in zip(Poly_wavelength, Poly_RI, celluloseRI, waterRIarray):
        
        dCore = int(Core_d)
        dShell = int(dShell)
        dShell1 = int(dShell1)
        dShell2 = int(dShell2)
        dShell3 = int(dShell3)
        dShell4 = int(dShell4)
        dShell5 = int(dShell5)
        qext, qsca, qabs, g, qpr, qback, qratio = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell,waterRI,asDict=False, asCrossSection=False)
        if dShell1 != 0:
            qext1, qsca1, qabs1, g1, qpr1, qback1, qratio1 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell1,waterRI,asDict=False, asCrossSection=False)
        if dShell2 != 0:
            qext2, qsca2, qabs2, g2, qpr2, qback2, qratio2 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell2,waterRI,asDict=False, asCrossSection=False)
        if dShell3 != 0:
            qext3, qsca3, qabs3, g3, qpr3, qback3, qratio3 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell3,waterRI,asDict=False, asCrossSection=False)
        if dShell4 != 0:
            qext4, qsca4, qabs4, g4, qpr4, qback4, qratio4 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell4,waterRI,asDict=False, asCrossSection=False)
        if dShell5 != 0:
            qext5, qsca5, qabs5, g5, qpr5, qback5, qratio5 = py.MieQCoreShell(mCore,mShell,wavelength,dCore,dShell5,waterRI,asDict=False, asCrossSection=False)
        Polystyrene_Cell_scattering.append(qsca)
        Polystyrene_Cell_scattering1.append(qsca1)
        Polystyrene_Cell_scattering2.append(qsca2)
        Polystyrene_Cell_scattering3.append(qsca3)
        Polystyrene_Cell_scattering4.append(qsca4)
        Polystyrene_Cell_scattering5.append(qsca5)
        
    plt.plot(Poly_wavelength,Polystyrene_Cell_scattering, label='{}nm shell'.format((dShell-dCore)))
    if dShell1 != 0:
        plt.plot(Poly_wavelength,Polystyrene_Cell_scattering1, label='{}nm shell'.format(dShell1-dCore))
    if dShell2 != 0:
        plt.plot(Poly_wavelength,Polystyrene_Cell_scattering2, label='{}nm shell'.format(dShell2-dCore))
    if dShell3 != 0:
        plt.plot(Poly_wavelength,Polystyrene_Cell_scattering3, label='{}nm shell'.format(dShell3-dCore))
    if dShell4 != 0:
        plt.plot(Poly_wavelength,Polystyrene_Cell_scattering4, label='{}nm shell'.format(dShell4-dCore))
    if dShell5 != 0:
        plt.plot(Poly_wavelength,Polystyrene_Cell_scattering5, label='{}nm shell'.format(dShell5-dCore))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Scattering efficiency")
    plt.title("Scattering efficiencies of Polystyrene core/Cellulose shell nanoparticles in water ({}nm core)".format(dCore))
    plt.legend()
    plt.show()    
