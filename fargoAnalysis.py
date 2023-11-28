import numpy as np
#import astropy as ap
from astropy import units as u
from astropy.units import cds
import astropy.constants as c
import matplotlib.pyplot as plt
#import matplotlib.pylab as plb
import yt
yt.set_log_level(40) #suppresses warnings and info which is output by default by yt when plotting

from pylab import *
#from scipy.optimize import fsolve
#from scipy.misc import derivative

import os
import datetime as dt
import sys
import glob

#import scipy as sp

#import pandas as pd
#import seaborn as sb




#########################
### PLOTTING DEFAULTS ###
#########################

# plotting colors
lilac = '#9D9DED'
darkblue = "#0d0786"
coolorange = '#f48849'
midpurp = '#cd4a75'
nordwhite = '#D8DEE9'
norddgrey = '#2e3440'

# import nord stylesheet for use in notebooks
plt.style.use('nord')


## figure color defaults
#mpl.rcParams['figure.facecolor'] = norddgrey
#mpl.rcParams['figure.edgecolor'] = norddgrey
#mpl.rcParams['axes.facecolor'] = nordwhite
#mpl.rcParams['axes.labelcolor'] = nordwhite
#mpl.rcParams['xtick.labelcolor'] = nordwhite
#mpl.rcParams['ytick.labelcolor'] = nordwhite
#mpl.rcParams['legend.edgecolor'] = norddgrey
#mpl.rcParams['savefig.transparent'] = False

## tick direction (change color to nordwhite if you switch direction to 'out')
#mpl.rcParams['xtick.direction'] = 'in'
#mpl.rcParams['ytick.direction'] = 'in'
#mpl.rcParams['xtick.color'] = norddgrey
#mpl.rcParams['ytick.color'] = norddgrey

## grid defaults
#mpl.rcParams['axes.grid'] = True
#mpl.rcParams['grid.alpha']= 0.33
#mpl.rcParams['grid.color']= norddgrey
#mpl.rcParams['grid.linewidth']= 1.25

## figure scale defaults
#mpl.rcParams['figure.figsize'] = [12.0, 9.0]
#mpl.rcParams['font.size'] = 24
#mpl.rcParams['legend.fontsize'] = 18
#mpl.rcParams['xtick.labelsize'] = 18
#mpl.rcParams['ytick.labelsize'] = 18
#mpl.rcParams['xtick.major.width'] = 1.25
#mpl.rcParams['xtick.major.size'] = 9
#mpl.rcParams['xtick.minor.width'] = 1.
#mpl.rcParams['xtick.minor.size'] = 4.5
#mpl.rcParams['ytick.major.width'] = 1.25
#mpl.rcParams['ytick.major.size'] = 9
#mpl.rcParams['ytick.minor.width'] = 1.
#mpl.rcParams['ytick.minor.size'] = 4.5

## fonts (chosen for uniformity with LaTeX)
#mpl.rcParams['mathtext.fontset'] = 'stix'
#mpl.rcParams['font.family'] = 'STIXGeneral'

# yt fontsize
slicefont = 28 #36 #changed from 36 on Sep 19, 2022





# Look for directory containing simulation data
if ('FARGODATA' in os.environ):
    data_dir = os.environ.get('FARGODATA')
else:
    print("ERROR: Environment variable 'FARGODATA' not set. Please set 'FARGODATA' to point to the directory where FARGO3D outputs are stored.")


###############################################
### PHYSICAL CONSTANTS & RELATED QUANTITIES ###
###############################################

cds.enable()

spd_lgt      = (1.0 * cds.c).cgs
grv_const    = (1.0 * cds.G).cgs
planck_const = 6.6260755e-27 * u.erg * u.s
boltzmann    = 1.380658e-16 * u.erg/u.K
thom_cross   = 6.6524e-25*u.cm**2
protonmass   = 1.6726231e-24 * u.g
accr_eff     = 0.1
mass_ratio   = 0.01
M_prim       = (1e8 * u.solMass).cgs
M_sec        = mass_ratio * M_prim

r_G = grv_const * M_prim / spd_lgt**2
sep = 100.0*r_G
r_hill = sep*(mass_ratio/3)**(1/3)
r_AU = (1.0 * u.AU).cgs

t_orb = 2.0*np.pi * np.sqrt((sep)**3 / (grv_const * (M_prim + M_sec)))
L_edd = (4 * np.pi * grv_const * M_prim * spd_lgt * protonmass / thom_cross).to(u.erg/u.s)
Mdot_edd = (L_edd / spd_lgt**2 / accr_eff).to(u.g/u.s)


t_orb    = u.def_unit('t_orb', t_orb,format={'latex':r't_\mathrm{orb}'})
f_orb    = u.def_unit('f_orb', 1.0/((1*t_orb).to(u.s)),format={'latex':r'f_\mathrm{orb}'})
r_G      = u.def_unit('r_G',r_G,format={'latex':r'r_\mathrm{G}'})
L_edd    = u.def_unit('L_Edd',L_edd,format={'latex':r'L_\mathrm{Edd}'})
Mdot_edd = u.def_unit('Mdot_Edd',Mdot_edd,format={'latex':r'\dot{M}_\mathrm{Edd}'})
yt.units.define_unit('rg',((1.0*r_G).cgs.value, "cm"), 'R$_\mathrm{g}$')
from unyt import rg

omega0 = spd_lgt**3 / (grv_const*(1.0*u.solMass).to(u.g))


FARGO_grv  = 6.6743e-8 * u.cm**3 / u.g / u.s**2
FARGO_c0   = 2.99792458e10 * u.cm / u.s
FARGO_mass = 1.988409870698051e33 * u.g
FARGO_dist = FARGO_grv * FARGO_mass / FARGO_c0**2

FARGO_freq = np.sqrt(FARGO_grv*FARGO_mass/FARGO_dist**3)
FARGO_time = 1/FARGO_freq

t_orb_FARGO = (t_orb/FARGO_time).to(u.dimensionless_unscaled)



def cmapCols(ncols,cmap='plasma',norm_min=0,norm_max=1):

    cm = plt.get_cmap(cmap)
    cols = cm(np.linspace(norm_min,norm_max,ncols))
#    exec(f"cols=cm.{cmap}(np.linspace({norm_min},{norm_max},{ncols}))")
    return cols


class Parameters(object):
    """                                                                                                                                                                                                     
    Class for reading the simulation parameters.                                                                                                                                                            
    input: string -> name of the parfile, normally variables.par                                                                                                                                            
    """
    def __init__(self, paramfile):
        try:
            params = open(paramfile,'r') #Opening the parfile                                                                                                                                               
        except IOError:                  # Error checker.                                                                                                                                                   
            print(paramfile + " not found.")
            return
        lines = params.readlines()     # Reading the parfile                                                                                                                                                
        params.close()                 # Closing the parfile                                                                                                                                                
        par = {}                       # Allocating a dictionary                                                                                                                                            
        for line in lines:             #Iterating over the parfile                                                                                                                                          
            name, value = line.split() #Spliting the name and the value (first blank)                                                                                                                       
            try:
                float(value)           # First trying with float                                                                                                                                            
            except ValueError:         # If it is not float                                                                                                                                                 
                try:
                    int(value)         #                   we try with integer                                                                                                                              
                except ValueError:     # If it is not integer, we know it is string                                                                                                                         
                    value = '"' + value + '"'
            par[name] = value          # Filling the dictory                                                                                                                                                
        self._params = par             # A control atribute, actually not used, good for debbuging                                                                                                          
        for name in par:               # Iterating over the dictionary                                                                                                                                      
            exec("self."+name.lower()+"="+par[name]) #Making the atributes at runtime   
        
        self.ymin = self.ymin*u.cm
        self.ymax = self.ymax*u.cm
        
        self.dy = (self.ymax - self.ymin)/self.ny
        self.dx = (self.xmax - self.xmin)/self.nx
        
        self.elec_opacity = self.elec_opacity * u.cm*u.cm/u.g
        self.krmr_opacity = self.krmr_opacity * u.cm*u.cm/u.g /u.K**(-7/2) / u.g *u.cm**3
        
    
        self.protonmass    = self.protonmass*u.g
        self.boltzmann     = self.boltzmann*u.erg/u.K
        self.stefboltz     = self.stefboltz*u.erg/(u.cm**2)/u.s/(u.K**4)
        
        self.dt            = self.dt * u.s
        
        # self.temp_min      = self.temp_min * u.K
        
        self.omegaframe    = self.omegaframe*omega0
        
        self.thomcross     = self.thomcross * u.cm*u.cm

P = Parameters(data_dir + "variables.par")





class Field:
    """
    Class for defining field objects to be read or computed from simulation outputs.
    inputs: name          -> name of the field
            units         -> astropy units for the field
            yt_units      -> string for the yt-formatted units (used for yt plots)
            label         -> formatted string used to label plots of this quantity
            val_min       -> default minimum value to use for setting plot axes
            val_max       -> default maximum value to use for setting plot axes
            cmap          -> colormap to use for 2D plots of this quantity (default is viridis)
            linthresh     -> threshold below which values are plotted linearly instead of log. allows for symlog plotting of quantities with negative values.
            compute_files -> list of the simulation output fields needed to compute this quantity
    """
    
    def __init__(self,name='field',units=u.dimensionless_unscaled,yt_units='',label='',val_min=0,val_max=1,cmap='viridis',linthresh=None,compute_files=[]):
        self.name = name
        self.units = units
        self.yt_units = yt_units
        self.label = label
        self.min = val_min*self.units
        self.max = val_max*self.units
        self.cmap = cmap
        self.linthresh = linthresh
        self.compute_files = compute_files





#####################
### DEFINE FIELDS ###
#####################

energy = Field(name='energy',
               units=u.erg/u.cm/u.cm,
               yt_units = 'erg/cm**2',
               label=r'Internal Energy Density [erg/cm$^2$]',
               val_min=1e14,
               val_max=1e24,
               cmap='magma',
               linthresh=False, 
               compute_files = ['energy'])

surfdens = Field(name='surfdens',
               units=u.g/u.cm/u.cm,
               yt_units = 'g/cm**2',
               label=r'Surface Density [g/cm$^{2}$]',
               val_min=1e1,
               val_max=1e8,
               cmap='viridis',
               linthresh=False, 
               compute_files = ['dens'])

phivel = Field(name='phivel',
               units=u.cm/u.s,
               yt_units = 'cm/s',
               label=r'Azimuthal Velocity [cm/s]',
               val_min=-spd_lgt.value,
               val_max=spd_lgt.value,
               cmap='coolwarm',
               linthresh=spd_lgt.value/3e1, 
               compute_files = ['vx'])

phivel_local = Field(name='phivel_local',
               units=u.cm/u.s,
               yt_units = 'cm/s',
               label=r'v$_\phi$ - v$_{\phi,\mathrm{K}}$ [cm/s]',
               val_min=-spd_lgt.value/10,
               val_max=spd_lgt.value/10,
               cmap='coolwarm',
               linthresh=spd_lgt.value/1e3,
               compute_files = ['vx'])

phivel_residual = Field(name='phivel_residual',
               units=u.cm/u.s,
               yt_units = 'cm/s',
               label=r'(v$_\phi$ - v$_{\phi,\mathrm{K}})$/v$_{\phi,\mathrm{K}}$',
               val_min=-2.0,
               val_max=2.0,
               cmap='coolwarm',
               linthresh=1.0e-2,
               compute_files = ['vx'])


phivel_corot = Field(name='phivel_corot',
               units=u.cm/u.s,
               yt_units = 'cm/s',
               label=r'v$_\phi$ - v$_{\phi,BH2}$ [cm/s]',
               val_min=-spd_lgt.value,
               val_max=spd_lgt.value,
               cmap='coolwarm',
               linthresh=spd_lgt.value/1e2, 
               compute_files = ['vx'])

radvel = Field(name='radvel',
               units=u.cm/u.s,
               yt_units = 'cm/s',
               label=r'Radial Velocity [cm/s]',
               val_min=-spd_lgt.value,#/17e2,
               val_max=spd_lgt.value,#/17e2,
               cmap='coolwarm',
               linthresh=spd_lgt.value/1e4,
               compute_files = ['vy'])

temp = Field(name='temp',
               units=u.K,
               yt_units = 'K',
               label=r'Temperature [K]',
               val_min=1e2,
               val_max=1e7,
               cmap='plasma',
               linthresh=False, 
               compute_files = ['temp']) #,'energy'])

surftemp = Field(name='surftemp',
               units=u.K,
               yt_units = 'K',
               label=r'Surface Temperature [K]',
               val_min=1e2,
               val_max=1e7,
               cmap='plasma',
               linthresh=False, 
               compute_files = ['dens','energy','height'])

soundspeed = Field(name='soundspeed',
               units=u.cm/u.s,
               yt_units = 'cm/s',
               label=r'Sound Speed [cm/s]',
               val_min=1e5,#np.sqrt(P.gamma*P.boltzmann*temp.min/(P.meanmolecmass*P.protonmass)).value,
               val_max=1e10,#np.sqrt(P.gamma*P.boltzmann*temp.max/(P.meanmolecmass*P.protonmass)).value,
               cmap='plasma',
               linthresh=False, 
               compute_files = ['dens','energy'])

gaspressure = Field(name='gaspressure',
               units=u.cm/u.s,
               yt_units = 'erg/cm/cm',
               label=r'Gas Pressure [erg/cm/cm]',
               val_min=1e5,#np.sqrt(P.gamma*P.boltzmann*temp.min/(P.meanmolecmass*P.protonmass)).value,
               val_max=1e10,#np.sqrt(P.gamma*P.boltzmann*temp.max/(P.meanmolecmass*P.protonmass)).value,
               cmap='plasma',
               linthresh=False, 
               compute_files = ['dens','energy'])

radpressure = Field(name='radpressure',
               units=u.cm/u.s,
               yt_units = 'erg/cm/cm',
               label=r'Radiation Pressure [erg/cm/cm]',
               val_min=1e5,#np.sqrt(P.gamma*P.boltzmann*temp.min/(P.meanmolecmass*P.protonmass)).value,
               val_max=1e10,#np.sqrt(P.gamma*P.boltzmann*temp.max/(P.meanmolecmass*P.protonmass)).value,
               cmap='plasma',
               linthresh=False, 
               compute_files = ['dens','energy'])

pressure = Field(name='pressure',
               units=u.erg/u.cm/u.cm,
               yt_units = 'erg/cm/cm',
               label=r'Pressure [erg/cm/cm]',
               val_min=1e20, #np.sqrt(P.gamma*P.boltzmann*temp.min/(P.meanmolecmass*P.protonmass)).value,#1e5
               val_max=1e24, #np.sqrt(P.gamma*P.boltzmann*temp.max/(P.meanmolecmass*P.protonmass)).value,#1e10
               cmap='plasma',
               linthresh=False, 
               compute_files = ['pres'])

viscosity = Field(name='viscosity',
               units=u.cm**2 / u.s,
               yt_units = 'cm*cm/s',
               label=r'Viscosity [cm$^2$/s]',
               val_min=1e5,
               val_max=1e10,
               cmap='plasma',
               linthresh=False, 
               compute_files = [])

cooledtemp = Field(name='cooledtemp',
               units=u.K,
               yt_units = 'K',
               label=r'Cooled Temperature [K]',
               val_min=1e2,
               val_max=1e7,
               cmap='plasma',
               linthresh=False, 
               compute_files = [''])

coolrate = Field(name='coolrate',
               units=u.erg/u.s,
               yt_units = 'erg/s',
               label=r'Cooling Rate [erg/s]',
               val_min=1e34,
               val_max=1e44,
               cmap='magma',
               linthresh=False, 
               compute_files = [''])

binary   = Field(name='binary',
                units=u.dimensionless_unscaled,
                yt_units='g',
                label=r'Ratio',
                val_min=0.0,
                val_max=1.0,
                cmap='coolwarm',
                linthresh=False, 
                compute_files=[''])

radius   = Field(name='radius',
                units=u.cm,
                yt_units='cm',
                label=r'Radius [cm]',
                val_min=1.0e14,
                val_max=3.0e17,
                cmap='viridis',
                linthresh=False, 
                compute_files=[''])

scaleheight = Field(name='scaleheight',
                units=u.cm,
                yt_units='cm',
                label=r'Scale Height [cm]',
                val_min=1.0e10,
                val_max=3.0e17,
                cmap='viridis',
                linthresh=False, 
                compute_files=['height'])

force_grav = Field(name='force_grav',
                units=u.cm/u.s,
                yt_units='cm/s',
                label=r'$\Delta$v$_\mathrm{r,grav}$ [cm/s]',
                val_min=1.0e10,
                val_max=3.0e17,
                cmap='viridis',
                linthresh=False, 
                compute_files=['fgrav'])

force_cent = Field(name='force_cent',
                units=u.cm/u.s,
                yt_units='cm/s',
                label=r'$\Delta$v$_\mathrm{r,cent}$ [cm/s]',
                val_min=1.0e10,
                val_max=3.0e17,
                cmap='viridis',
                linthresh=False, 
                compute_files=['fcent'])

force_pres = Field(name='force_pres',
                units=u.cm/u.s,
                yt_units='cm/s',
                label=r'$\Delta$v$_\mathrm{r,pres}$ [cm/s]',
                val_min=1.0e10,
                val_max=3.0e17,
                cmap='viridis',
                linthresh=False, 
                compute_files=['fpres'])

beta       = Field(name='beta',
                units=u.dimensionless_unscaled,
                yt_units='cm',
                label=r'$\beta$',
                val_min=1.0e10,
                val_max=3.0e17,
                cmap='viridis',
                linthresh=False, 
                compute_files=['dens','energy'])

gamma = Field(name='gamma',
               units=u.dimensionless_unscaled,
               yt_units = 'cm',
               label=r'$\gamma$',
               val_min=4./3.,
               val_max=5./3.,
               cmap='viridis',
               linthresh=False, 
               compute_files = ['gamma'])

de_av = Field(name='de_av',
               units=u.erg/u.cm/u.cm,
               yt_units = 'erg/cm/cm',
               label=r'$\Delta$ e$_\mathrm{AV}$',
               val_min=1e20,
               val_max=1e30,
               cmap='magma',
               linthresh=False, 
               compute_files = ['delta_e_artvisc'])

potential = Field(name='potential',
                units=u.erg/u.g,
                yt_units='erg/g',
                label=r'Potential [erg/g]',
                val_min=1e16,
                val_max=1e20,
                cmap='viridis',
                linthresh=False, 
                compute_files=['potential'])

mach = Field('mach',
             units=u.dimensionless_unscaled,
             yt_units='cm',
             label='Shock Mach Number',
             compute_files=['mach'],
             val_min=0.0,
             val_max=1.0,
             linthresh=1.0e-3,
             cmap='magma')



##########################
### DEFINE GRID OBJECT ###
##########################

class fargoGrid:
    """
    Class for holding the grid structure and other simulation quantities which remain constant through time
    inputs: sim_name  -> name of the subdirectory of $FARGODATA containing the simulation outputs
            nghx      -> number of ghost zones in the X-direction
            nghy      -> number of ghost zones in the Y-direction
    """

    def __init__(self,sim_name,nghx=0,nghy=3):
        P = Parameters(data_dir + sim_name + "/" + "variables.par")

        self.name=sim_name
        self.mprim = P.mprim0*u.solMass
        self.q     = P.massratio
        self.r_G = grv_const * (P.mprim0*u.solMass).to(u.g) / spd_lgt**2
        self.t_orb = (2.0*np.pi * np.sqrt((P.separation * self.r_G)**3 / (grv_const * (1.0 + P.massratio)*P.mprim0*u.solMass))).to(u.s)
        self.L_edd = (4.0*np.pi * grv_const * self.mprim * protonmass * spd_lgt / thom_cross).to(u.erg/u.s)
        self.mdot_edd = (self.L_edd / spd_lgt**2 / accr_eff).to(u.g/u.s)
        
        self.t_orb         = u.def_unit('t_orb', self.t_orb,format={'latex':r't_\mathrm{orb}'})
        self.r_G           = u.def_unit('r_G',self.r_G,format={'latex':r'r_\mathrm{G}'})
        self.L_edd         = u.def_unit('L_edd',self.L_edd,format={'latex':r'L_\mathrm{Edd}'})
        self.mdot_edd      = u.def_unit('mdot_edd',self.mdot_edd,format={'latex':r'\dot{\mathrm{M}}_\mathrm{Edd}'})
        
        self.sep = P.separation*self.r_G
        self.dt  = P.dt

        unit=P.ymax.unit
        if P.spacing[:3] == 'lin':
            dy = (P.ymax-P.ymin)/P.ny
            self.rfaces = [P.ymin.value+dy*(j-nghy) for j in range(0,P.ny+2*nghy+1)]*unit
        elif P.spacing[:3] == 'log':
            dy = (np.log(P.ymax.value) - np.log(P.ymin.value))/P.ny
            self.rfaces = [np.exp(np.log(P.ymin.value) + dy*(j-nghy)) for j in range(0,P.ny+2*nghy+1)]*unit
        self.rfaces = self.rfaces[nghy:-nghy]
            
            
        dx = (P.xmax-P.xmin)/P.nx
        self.phifaces = [P.xmin+dx*(i-nghx) for i in range(0,P.nx+2*nghx+1)]*u.rad
        self.phifaces = [(P.xmin+P.xmax)+dx*(i-nghx) for i in range(0,P.nx+2*nghx+1)]*u.rad

        self.radius_1d = (np.array([(self.rfaces[i]+self.rfaces[i-1]).value/2 for i in range(1,len(self.rfaces))])*u.cm).to(self.r_G)
        self.radius = np.repeat(self.radius_1d,P.nx,axis=0).reshape(P.ny,P.nx,1).to(u.cm)
    
        self.phi_1d = np.array([(self.phifaces[i]+self.phifaces[i-1]).value/2 for i in range(1,len(self.phifaces))])*u.rad
        self.phi = np.tile(self.phi_1d,P.ny).reshape(P.ny,P.nx,1)

        self.zone_size_phi = np.array([(self.phifaces[i] - self.phifaces[i-1]).value for i in range(1,len(self.phifaces))]) * u.rad
        self.zone_size_phi = np.array([(self.zone_size_phi*radius.to(u.cm)).value for radius in self.radius_1d]).reshape(P.ny,P.nx,1)*u.cm
        
        self.zone_size_r = np.array([(self.rfaces[i]-self.rfaces[i-1]).value for i in range(1,len(self.rfaces))])*u.cm
        self.zone_size_r = np.repeat(self.zone_size_r,P.nx,axis=0).reshape(P.ny,P.nx,1)
        
        areas = np.array([(P.dx/2) * (self.rfaces[i+1].cgs.value**2 - self.rfaces[i].cgs.value**2) for i in range(P.ny)])
        self.areas = np.repeat(areas,P.nx,axis=0).reshape(P.ny,P.nx,1)*u.cm**2

        self.xcen = self.radius*np.cos(self.phi)
        self.ycen = self.radius*np.sin(self.phi)

        self.hillrad  = self.sep * (self.q/3)**(1/3) 
        self.hillmask = np.sqrt(self.ycen**2 + (self.xcen+self.sep)**2)>self.hillrad
        
        xgrid = self.phifaces
        ygrid = self.rfaces/r_AU
        zgrid = np.array([0.0,1.0])
        self.coords, self.conn = yt.hexahedral_connectivity(ygrid,xgrid,zgrid)


        self.bbox = np.array([[np.min(ygrid).value,np.max(ygrid).value],
                         [np.min(xgrid).value,np.max(xgrid).value],
                         [np.min(zgrid),np.max(zgrid)]])





##########################
### DEFINE DISK OBJECT ###
##########################

class fargoData:
    """
    Class for loading and processing the simulation state at a given output timestep
    inputs: sim_name  -> name of the subdirectory of $FARGODATA containing the simulation outputs
            timestep  -> the output timestep to be loaded/analyzed
    NOTE: fargoData() requires that planet0.dat be present in $FARGODATA/sim_name 
          as this is where the physical time and frame rotation rate are read from
    """
 

    def __init__(self,sim_name,timestep):
        P = Parameters(data_dir + sim_name + "/" + "variables.par")
        #IN GENERAL: Things that are unique to a SPECIFIC fargo dataset should be defined/assigned here
        #Things that would be common to all of them (I think just methods for me) should be assigned outside of init
        self.name=sim_name
        self.timestep = timestep
        self.mprim = P.mprim0*u.solMass
#        self.q     = P.massratio
        self.r_G = grv_const * (P.mprim0*u.solMass).to(u.g) / spd_lgt**2
        self.t_orb = (2.0*np.pi * np.sqrt((P.separation * self.r_G)**3 / (grv_const * (1.0 + P.massratio)*P.mprim0*u.solMass))).to(u.s)
        
        self.t_orb    = u.def_unit('t_orb', self.t_orb,format={'latex':r't_\mathrm{orb}'})
        self.r_G      = u.def_unit('r_G',self.r_G,format={'latex':r'r_\mathrm{G}'})
        
        self.sep = P.separation*self.r_G
        
        self.ny  = P.ny
        self.nx  = P.nx
        


        with open(data_dir + sim_name + "/" + "planet0.dat") as file:
            for i, line in enumerate(file):
                if line.split("\t")[0] == str(self.timestep):
                    self.time = np.float64(line.split("\t")[8])*u.s
                    self.omegaframe = np.float64(line.split("\t")[9])*(1/u.s)
                    #break

       
        
            
        ############################################################################
        ### MOVE THIS BLOCK INTO ITS OWN THING, TO ONLY CALL WHEN MDOT IS NEEDED ###
        ############################################################################
        
#         with open(data_dir + sim_name + "/" + "accr0.dat") as file:
#             for i, line in enumerate(file):
#                 if line.split("\t")[0] == str(self.timestep):
#                     mdot_bh2_data = line.split("\t")
#                     break
                 
#         try:
#             self.mdot = np.float64(mdot_bh2_data[2])*u.g/u.s
#         except UnboundLocalError:
#             self.mdot = np.nan
    
        ####################
        ### END OF BLOCK ###
        ####################
       
        #this is also where we will assign values to attributes for holding 
        #the imported data (primitive variables, accretion/cooling outputs, etc)
        
        
        #Be sure to use units for imported data
        #Define some useful new units like t_orb here as well, based on the imported parameters (I guess from the orbit files)



    def _readCoarseDataFile(self,field,file_num,sub_ind):
        if (field=='dens') or (field=='energy') or (field=='vx') or (field=='vy') or (field=='gamma') or (field=='pres') or (field=='height') or (field=='temp'):
            filename = data_dir + self.name + '/gas' + field + '_cat' + str(file_num) + '.dat'
        else:
            filename = data_dir + self.name + '/' + field + '_cat' + str(file_num) + '.dat'
        try:
            data = fromfile(filename)[self.nx*self.ny*sub_ind:self.nx*self.ny*(sub_ind+1)]
        except FileNotFoundError:
            if (field=='dens') or (field=='energy') or (field=='vx') or (field=='vy') or (field=='gamma') or (field=='pres') or (field=='height') or (field=='temp'):
                filename = data_dir + self.name + '/gas' + field + str(self.timestep) + '.dat'
            else:
                filename = data_dir + self.name + '/' + field + str(self.timestep) + '.dat'

            data = fromfile(filename)
    
        return data



    def getFieldData(self,field,grid=None):
        P = Parameters(data_dir + self.name + "/" + "variables.par")
        data_needed = field.compute_files
    
        timestep = self.timestep
        file_num = int(timestep/10)
        sub_ind  = timestep%10

        field_data = np.array([self._readCoarseDataFile(quant,file_num,sub_ind) for quant in data_needed])
    
        if field.name == 'phivel':
            field_data = field_data + (self.omegaframe*grid.radius).flatten().to(u.cm/u.s).value
            
        if field.name == 'phivel_local':
            field_data = field_data + (self.omegaframe*grid.radius).flatten().to(u.cm/u.s).value
            field_data = field_data - np.sqrt(grv_const * P.mprim0*u.solMass / grid.radius).flatten().to(u.cm/u.s).value

        if field.name == 'phivel_residual':
            field_data = field_data + (self.omegaframe*grid.radius).flatten().to(u.cm/u.s).value
            field_data = (field_data - np.sqrt(grv_const * P.mprim0*u.solMass / grid.radius).flatten().to(u.cm/u.s).value)/np.sqrt(grv_const * P.mprim0*u.solMass / grid.radius).flatten().to(u.cm/u.s).value

 
    
        if field.name == 'phivel_corot':
            field_data = field_data
 
        if field.name == 'potential':
            field_data = np.abs(field_data)

#         if field.name == 'temp': #Convert internal energy frame to temperature
#             field_data = field_data[1,:]/field_data[0,:]
#             field_data = field_data*(P.gamma - 1.0)
#             field_data = field_data*(P.meanmolecmass * P.protonmass)
#             field_data = field_data/(P.boltzmann)


        elif field.name == 'surftemp': #Convert internal energy frame to temperature
            dens_data     = field_data[0,:]*u.g/u.cm/u.cm
            scale_height  = field_data[2,:]*u.cm 
            field_data    = field_data[1,:]*energy.units/dens_data
            field_data    = field_data*(P.gamma - 1.0)
            field_data    = field_data*(P.meanmolecmass * P.protonmass)
            field_data    = field_data/(P.boltzmann)
            temp_midplane = field_data

#             rad_temp = np.linspace(P.ymin+0.5*P.dy,P.ymax-0.5*P.dy,P.ny)
#             rad_data = np.repeat(rad_temp,P.nx,axis=0)
#             rad_data = rad_data.reshape(P.ny,P.nx,1)
#             ang_vel = np.sqrt(grv_const * (P.mprim0*u.solMass).to(u.g) / (rad_data**3))
#             cs_2    = P.gamma * P.boltzmann * temp_midplane / (P.meanmolecmass * P.protonmass)
#             ang_vel = ang_vel.reshape(P.ny*P.nx)
#             scale_height = (np.sqrt(cs_2) / ang_vel).to(u.cm)

    

            #tau_bf = dens_data * (P.krmr_opacity * ((dens_data/(2*scale_height))/(1.0*u.g/u.cm**3)) * (temp_midplane/(1.0*u.K))**(-3.5))
            tau_bf = dens_data * (P.krmr_opacity * ((dens_data/(2*scale_height))) * (temp_midplane)**(-3.5))

            tau_e  = dens_data * P.elec_opacity
            
            ##### DOUBLE CHECK THIS #####
            #tau_term = (tau_bf / (1.0 + tau_bf**2)) + (tau_e / (1.0 + tau_e**2))
#             print('tau_bf',tau_bf.unit)
#             print('tau_e',tau_e.unit)
#             print('dens',dens_data.unit)
#             print('krmr_k',P.krmr_opacity.unit)
#             print('scale_h',scale_height.unit)
#             print('temp_c',temp_midplane.unit)
#             print('elec_k',P.elec_opacity.unit)
            
            tau_tot = tau_e + tau_bf
            tau_term = tau_tot / (1.0+ tau_tot**2)
            ##### DOUBLE CHECK THIS #####
            
            
            field_data = temp_midplane * tau_term**(0.25)
    
        elif field.name == 'gaspressure':
            field_data = (P.gamma-1.0)* field_data[1,:]*(u.erg/u.cm/u.cm)
            
                
        elif field.name == 'radpressure':
            
            pres = (P.gamma-1.0)* field_data[1,:]*(u.erg/u.cm/u.cm)
                
            temp = pres*(P.meanmolecmass*P.protonmass)/P.boltzmann/(field_data[0,:]*(u.g/u.cm/u.cm))
                
            try:
                radpres = 4.0*P.stefboltz*(temp**4)*self.scaleheight.flatten()/3/spd_lgt
            except AttributeError:
                self.getFieldData(scaleheight)
                radpres = 4.0*P.stefboltz*(temp**4)*self.scaleheight.flatten()/3/spd_lgt

            field_data = radpres
    
    
    
#         elif field.name == 'pressure':
#             if (P.useradpressure):
#                 try:
#                     field_data = self.gaspressure + self.radpressure
#                 except AttributeError:
#                     self.getFieldData(gaspressure)
#                     self.getFieldData(radpressure)
#                     field_data = self.gaspressure + self.radpressure
#             else:
#                 try:
#                     field_data = self.gaspressure
#                 except AttributeError:
#                     self.getFieldData(gaspressure)
#                     field_data = self.gaspressure
    
        elif field.name == 'viscosity':
            omega = np.sqrt(grv_const * P.mprim0*u.solMass / self.radius**3).flatten().cgs
            try:
                visc = P.alpha * self.soundspeed.flatten()**2 / omega
            except AttributeError:
                self.getFieldData(soundspeed)
                visc = P.alpha * self.soundspeed.flatten()**2 / omega
                
            field_data = visc.cgs.value
            
    
        elif field.name == 'soundspeed':
            if (P.useradpressure):
                #pres = (P.gamma-1.0)* field_data[1,:]*(u.erg/u.cm/u.cm)
                
                #temp = pres*(P.meanmolecmass*P.protonmass)/P.boltzmann/(field_data[0,:]*(u.g/u.cm/u.cm))
                
                try:
                    field_data = P.gamma * self.pressure.flatten() / self.surfdens.flatten()
#                     pres += 4.0*P.stefboltz*(temp**4)*self.scaleheight.flatten()/3/spd_lgt
                except AttributeError:
                    self.getFieldData(pressure)
                    self.getFieldData(surfdens)
                    field_data = P.gamma * self.pressure.flatten() / self.surfdens.flatten()     
#                     self.getFieldData(scaleheight)
#                     pres += 4.0*P.stefboltz*(temp**4)*self.scaleheight.flatten()/3/spd_lgt

                
#                 field_data = P.gamma*pres/field_data[0,:]
                field_data = np.sqrt(field_data)
                    
                
            else:
                field_data = field_data[1,:]/field_data[0,:]
                field_data = P.gamma*(P.gamma - 1.0)*field_data
                field_data = np.sqrt(field_data)
            
    
#         elif field.name == 'scaleheight':
#             if (P.useradpressure):
#                 ang_vel = sqrt( grv_const * M_prim / (self.radius**3) ).flatten()

#                 temp = (P.gamma - 1.0) * field_data[1,:]*(u.erg/u.cm/u.cm) * P.meanmolecmass * P.protonmass / (P.boltzmann * field_data[0,:]*u.g/u.cm/u.cm)


#                 term1 = 2.0 * P.stefboltz * temp**4 / (3.0 * spd_lgt * field_data[0,:]*(u.g/u.cm/u.cm))
#                 term2 = P.gamma / (ang_vel*ang_vel)
#                 term3 = (P.gamma - 1.0) * field_data[1,:]*(u.erg/u.cm/u.cm) / (field_data[0,:]*(u.g/u.cm/u.cm))

#                 field_data = term1*term2*(1.0 + sqrt(1.0 + term3/(term1*term1*term2)))
                
#             else:
#                 c_s = field_data[1,:]/field_data[0,:]
#                 c_s = P.gamma*(P.gamma - 1.0)*field_data
#                 c_s = np.sqrt(field_data)
                
#                 ang_vel = sqrt( grv_const * M_prim / (self.radius**3) )
                
#                 field_data = c_s/ang_vel
                
        field_data = field_data.reshape(self.ny,self.nx,1)
                

        exec("self."+field.name+"=(field_data*u.dimensionless_unscaled).value*field.units")
        return
    


    def _pullFieldData(self,field,grid=None):
        try:
            exec('global field_data; field_data=self.'+field.name,globals(), locals())
        except AttributeError:
            self.getFieldData(field,grid)
            exec('global field_data; field_data=self.'+field.name,globals(), locals())
        return field_data
    
    

    def plotField(self,field,grid=None,width=None, plotLog=False, save=False,high_res=False,xcen=0,ycen=0,zcen=0,frameon=False,labelcol=nordwhite,tickcol=norddgrey,bgcol=None):#nordwhite+"B8"):#500*(self.mprim/(1e8*u.solMass)).value*(self.sep/(100*self.r_G)).value, save=False,high_res=False):  
        
        if width==None:
            width = 500*(self.mprim/(1e8*u.solMass)).value*(self.sep/(100*self.r_G)).value
        
        P = Parameters(data_dir + self.name + "/" + "variables.par")
        field_data = self._pullFieldData(field,grid).value
#         print(field_data.min(),field_data.max())
#         print(P.dt,P.ninterm,self.timestep)
        name_index = '0'*(4-len(str(self.timestep))) + str(self.timestep)
#         if self.timestep <= 12073:
#             num_orb = (self.timestep * P.dt * P.ninterm).to(t_orb)
#         else:
#             num_orb = ((12073 * P.dt * P.ninterm + (self.timestep-12073) * P.dt * (P.ninterm/10))).to(t_orb)
#         if self.timestep < 87:
#             num_orb = (self.timestep * P.dt * P.ninterm).to(t_orb)
#         else:
#             num_orb = ((87 * P.dt * 20 + (self.timestep-87) * P.dt * (P.ninterm))).to(t_orb)
        num_orb = self.time.to(self.t_orb) #(self.timestep * P.dt * P.ninterm).to(self.t_orb)
        time_now = "T = " + '{:.9f}'.format(num_orb.value)[:9] + r" $\mathrm{t_{orb}}$"
#         print(time_now)
    
#        if (field_data.min() >= 0.0 and field != binary):#(field != 'vy' and field != 'vx' and field != 'bound'):
#            threshold_indices = field_data < field.min/1e3
#            field_data[threshold_indices] = field.min/1e3
            #The above just sets a floor on the data which prevents visualization glitches in logspace

#         print(field_data.min(),field_data.max())
#         print(P.spacing,P.outputdir)
        if P.spacing == 'log':
            if grid==None:
                print("ERROR: You must specify a FARGO3D grid object to visualize a log-spaced domain.")
                return
#             print('Log spacing!')

#             with open(data_dir + self.name + "/" + "domain_x.dat") as file:
#                 phifaces = np.array([np.float64(line[:]) for line in file])*u.rad
             
#            xgrid = grid.phifaces #self.phifaces
#            ygrid = grid.rfaces/r_AU #self.rfaces/r_AU
#            zgrid = np.array([0.0,1.0])
#            coords, conn = yt.hexahedral_connectivity(ygrid,xgrid,zgrid)


#            bbox = np.array([[np.min(ygrid).value,np.max(ygrid).value],
#                             [np.min(xgrid).value,np.max(xgrid).value],
#                             [np.min(zgrid),np.max(zgrid)]])
            data = dict(density = (field_data, field.yt_units))
#            ds = yt.load_hexahedral_mesh(data,conn,coords,length_unit="AU",bbox=bbox,geometry= 'polar')
            ds = yt.load_hexahedral_mesh(data,grid.conn,grid.coords,length_unit="AU",bbox=grid.bbox,geometry= 'polar')
        else:
            bbox = np.array([[P.ymin/r_AU,P.ymax/r_AU],[-np.pi,np.pi],[0.0,1.0]])
            data = dict(density = (field_data, field.yt_units))
            ds = yt.load_uniform_grid(data,field_data.shape,length_unit="AU",bbox=bbox,geometry="polar")
        
#        s = yt.SlicePlot(ds,'z', ('gas','density'), center=[xcen, ycen, zcen], fontsize=slicefont,width=(width,'AU'))
#        s = yt.SlicePlot(ds,'z', ('gas','density'), origin=[xcen, ycen, zcen], fontsize=slicefont,width=(width,'AU'))


        yt.units.define_unit('rg',((1.0*r_G).cgs.value, "cm"), 'R$_\mathrm{g}$',registry=ds.unit_registry)

        s = yt.SlicePlot(ds,'z', ('gas','density'), fontsize=slicefont,width=(width,'rg'))
            
            
#            plt.setp(s.plots[('gas','density')].axes.get_xticklabels(), color="red")
#            fig.set_edgecolor("red")
#            cax.set_facecolor("black")
#            ax.set_facecolor("black")
#            print('i be doin work')
 
# And create a mini-panel of a gaussian histogram inside the plot
#            rect = (0.2, 0.2, 0.2, 0.2)
#            new_ax = fig.add_axes(rect)

#            n, bins, patches = new_ax.hist(
#                np.random.randn(1000) + 20, 50, facecolor="black", edgecolor="black"
#            )

# Make sure its visible
#            new_ax.tick_params(colors="white")
#            new_ax.set_facecolor("red")

# And label it
#            la = new_ax.set_xlabel("Dinosaurs per furlong")
#            la.set_color("white")

#         print(field.min.value,field.max.value)
#        if (np.nanmin(field_data) > 0.0 and field != binary and field != gamma) or plotLog:    
#            s.set_log(('gas','density'),True)
#        else:
#            s.set_log(('gas','density'),False)
#        s.set_log(('gas','density'),True,linthresh=field.linthresh)
        if (field.linthresh==False):
            s.set_log(('gas','density'),True)
        else:
            s.set_log(('gas','density'),linthresh=field.linthresh)

#        slc = ds.slice('z',0.0)
        s.set_xlabel('x [R$_\mathrm{g}$]') #x [AU]')
        s.set_ylabel('y [R$_\mathrm{g}$]') #x [AU]')
#        s.set_ylabel('y [AU]')
        s.set_zlim(field=('gas','density'),zmin=field.min.value,zmax=field.max.value) 
#        s.annotate_title(time_now)
#        s.set_width(width,'AU')
        s.set_cmap(cmap=field.cmap,field=('gas','density'))
        s.set_colorbar_label(label=field.label,field=('gas','density'))
#        if not(save):
#            s.set_background_color(field=('gas','density'),color=(0,0,0,0))
#            s.hide_axes(draw_frame=True)
#            fig  = s.plots[('gas','density')].figure
#            ax   = s.plots[('gas','density')].axes
#            cax  = s.plots[('gas','density')].cax
#            fig.set_facecolor("red") #(0,0,0,0)
#            fig.set_edgecolor("red")
#            cax.set_facecolor("black")
#            ax.set_facecolor("black")
#            print('i be doin work')
        
#        if not(save):
##            s.set_background_color(field=('gas','density'),color=bgcol)
#            s.hide_axes(draw_frame=True)
#            fig  = s.plots[('gas','density')].figure
#            ax   = s.plots[('gas','density')].axes
#            cax  = s.plots[('gas','density')].cax
#            fig.patch.set_facecolor("red") #(0,0,0,0)
#            s.plots[('gas','density')].figure.patch.set_facecolor('black')
##            s.plots[('gas','density')].figure.set_frameon(frameon)
##            s.plots[('gas','density')].figure.suptitle(time_now,color=labelcol,fontsize=36,y=0.995)
##            s.plots[('gas','density')].figure.title(time_now,color=labelcol,fontsize=36,y=0.995)
##            s.plots[('gas','density')].axes.xaxis.label.set_color(labelcol)
##            s.plots[('gas','density')].axes.yaxis.label.set_color(labelcol)
##            s.plots[('gas','density')].axes.yaxis.label.set_fontsize(10)
##            s.plots[('gas','density')].axes.tick_params(which='both',axis='x', labelcolor=labelcol,labelsize=10,color=tickcol,grid_alpha=0)
##            s.plots[('gas','density')].axes.tick_params(which='both',axis='y', labelcolor=labelcol,labelsize=24,color=tickcol,grid_alpha=0)
##            s.plots[('gas','density')].cax.tick_params(which='both',labelcolor=labelcol,color=tickcol)
##            s.plots[('gas','density')].cax.yaxis.label.set_color(labelcol)

##            s.plots[('gas','density')].axes.xaxis.set_label('x [AU]')
#            s.plots[('gas','density')].axes.xaxis.grid(visible=None)
#            s.plots[('gas','density')].axes.yaxis.label.set_color(labelcol)

        s.set_background_color(field=('gas','density'),color=bgcol)
#            s.hide_axes(draw_frame=True)
#            fig  = s.plots[('gas','density')].figure
#            ax   = s.plots[('gas','density')].axes
#            cax  = s.plots[('gas','density')].cax
#            fig.patch.set_facecolor("red") #(0,0,0,0)
#            s.plots[('gas','density')].figure.patch.set_facecolor('black')
        s.plots[('gas','density')].figure.set_frameon(frameon)
        s.plots[('gas','density')].figure.suptitle(time_now,color=labelcol,fontsize=36,y=0.995)
        s.plots[('gas','density')].axes.xaxis.label.set_color(labelcol)
        s.plots[('gas','density')].axes.yaxis.label.set_color(labelcol)
        s.plots[('gas','density')].axes.yaxis.label.set_fontsize(10)
        s.plots[('gas','density')].axes.tick_params(which='both',axis='x', labelcolor=labelcol,labelsize=10,color=tickcol,grid_alpha=0)
        s.plots[('gas','density')].axes.tick_params(which='both',axis='y', labelcolor=labelcol,labelsize=24,color=tickcol,grid_alpha=0)
        s.plots[('gas','density')].cax.tick_params(which='both',labelcolor=labelcol,color=tickcol)
        s.plots[('gas','density')].cax.yaxis.label.set_color(labelcol)

        s.plots[('gas','density')].axes.xaxis.set_label('x [R$_\mathrm{g}$]')
        s.plots[('gas','density')].axes.yaxis.set_label('y [R$_\mathrm{g}$]')
        
#        xmin = xcen-width/2
#        xmax = xcen+width/2
#        ymin = ycen-width/2
#        ymax = ycen+width/2
#        print(s.plots[('gas','density')].axes.get_xlim())
#        print(s.plots[('gas','density')].axes.get_ylim())
#        s.plots[('gas','density')].axes.xaxis.set_view_interval(xmin,xmax)
#        s.plots[('gas','density')].axes.set_ylim(ymin,ymax)
#        print(s.plots[('gas','density')].axes.get_xlim())
#        print(s.plots[('gas','density')].axes.get_ylim())
        
        s.set_center([xcen,ycen],unit='rg')

#        if save:
#            s.annotate_title(time_now)

        if save:
        
            s.plots[('gas','density')].figure.set_rasterized(True)
            s.plots[('gas','density')].figure.figsize = (6.5,4.875)
            if high_res:
                s.save(data_dir + 'Images/' + self.name + '/' + field.name + '_width' + str(int(width)) + '_' + str(name_index) + '.pdf', mpl_kwargs={'dpi':300,'format':'pdf','facecolor':'white','edgecolor':'white'})
            else:
                s.save(data_dir + 'Images/' + self.name + '/' + field.name + '_width' + str(int(width)) + '_' + str(name_index) + '.png')
        else:
            with np.errstate(invalid='ignore'):
                s.show()
   


    def radialProfile(self,field,grid=None):
        field_data = self._pullFieldData(field,grid)
        
        exec("self."+field.name+"_1d=(field_data.mean(axis=1)).reshape(self.ny)")

    

    def massWeightedRadialProfile(self,field):
        field_data = self._pullFieldData(field)
        
        y_mins = np.linspace(P.ymin,P.ymax-P.dy,P.ny)
        y_maxs = np.linspace(P.ymin+P.dy,P.ymax,P.ny)
        areas = (P.dx/2) * (y_maxs**2 - y_mins**2)
        area_arr = np.repeat(areas,P.nx,axis=0).reshape(P.ny,P.nx,1)
        
        exec("self."+field.name+"_1d_massweighted=(field_data.mean(axis=1)).reshape(P.ny)")
        
        return ((field_data*self.surfdens*area_arr).sum(axis=1)/(self.surfdens*area_arr).sum(axis=1)).reshape(P.ny)
        
        

    def _planckFunc(self,temp,freq,freq_max=0,n_freqs=100):
        if not(freq_max) or (freq == freq_max):
            freq = freq.to(u.Hz, equivalencies=u.spectral())
    
            consts = 2*planck_const/(spd_lgt**2)
            term2  = (freq**3)/(np.exp(planck_const*freq/(P.boltzmann*temp)) - 1.0)
            
            return (consts*term2).to('erg/cm2')

        
        else:
            freq = freq.to(u.Hz, equivalencies=u.spectral())
            freq_max = freq_max.to(u.Hz, equivalencies=u.spectral())
    
            if (freq < freq_max):
                freqs = np.linspace(freq,freq_max,n_freqs)
            
            else:
                freqs = np.linspace(freq_max,freq,n_freqs)
                
            
            frames = np.array([self._planckFunc(temp,f,freq_max=0) for f in freqs])*u.erg/u.cm**2            
            frames = frames.reshape(n_freqs,P.ny,P.nx,1)
            
            planck_slice = np.trapz(frames,freqs,axis=0)

            return planck_slice
        
    
            
    def plotLumSlice(self,freq,freq_max=0,n_freqs=1000,min_val=4e34*u.erg/u.s,max_val=4e43*u.erg/u.s, width=None, save=False,high_res=False):#500*(self.mprim/(1e8*u.solMass)).value*(self.sep/(100*self.r_G)).value, save=False,high_res=False):
        
        if width==None:
            width = 500*(self.mprim/(1e8*u.solMass)).value*(self.sep/(100*self.r_G)).value
        
        P = Parameters(data_dir + self.name + "/" + "variables.par")
        name_index = '0'*(6-len(str(self.timestep))) + str(self.timestep)
#         if self.timestep <= 12073:
#             num_orb = (self.timestep * P.dt * P.ninterm).to(t_orb)
#         else:
#             num_orb = ((12073 * P.dt * P.ninterm + (self.timestep-12073) * P.dt * (P.ninterm/10))).to(t_orb)
#         time_now = "T = " + '{:.12f}'.format(num_orb.value)[:9] + r" $\mathrm{t_{orb}}$"
        num_orb = self.time.to(self.t_orb) #(self.timestep * P.dt * P.ninterm).to(self.t_orb)
        time_now = "T = " + '{:.9f}'.format(num_orb.value)[:9] + r" $\mathrm{t_{orb}}$"
    
    
#         y_mins = np.linspace(P.ymin,P.ymax-P.dy,P.ny)
#         y_maxs = np.linspace(P.ymin+P.dy,P.ymax,P.ny)
#         areas = (P.dx/2) * (y_maxs**2 - y_mins**2)
#         area_arr = np.repeat(areas,P.nx,axis=0).reshape(P.ny,P.nx,1)
        
        try:
            lum_data = self._planckFunc(self.surftemp,freq,freq_max,n_freqs)* self.areas# * (freq).to('Hz',equivalencies=u.spectral())
        except AttributeError:
            self.getFieldData(surftemp)
            lum_data = self._planckFunc(self.surftemp,freq,freq_max,n_freqs)* self.areas# * (freq).to('Hz',equivalencies=u.spectral())
        if not(freq_max) or (freq == freq_max):
            lum_data *= (freq).to('Hz',equivalencies=u.spectral())

        
        threshold_indices = lum_data < min_val
        lum_data[threshold_indices] = min_val/1e1

#         bbox = np.array([[P.ymin/r_G,(P.ymax)/r_G],[-np.pi,np.pi],[0.0,1.0]])
#         data = dict(density = (lum_data, 'erg/s'))
#         ds = yt.load_uniform_grid(data,(P.ny,P.nx,1),length_unit="AU",bbox=bbox,geometry="polar")
        
        
        
        if P.spacing == 'log':
#             print('Log spacing!')

#             with open(data_dir + self.name + "/" + "domain_x.dat") as file:
#                 phifaces = np.array([np.float64(line[:]) for line in file])*u.rad
        
            xgrid = self.phifaces
            ygrid = self.rfaces/r_AU
            zgrid = np.array([0.0,1.0])
            coords, conn = yt.hexahedral_connectivity(ygrid,xgrid,zgrid)


            bbox = np.array([[np.min(ygrid).value,np.max(ygrid).value],
                             [np.min(xgrid).value,np.max(xgrid).value],
                             [np.min(zgrid),np.max(zgrid)]])
            data = dict(density = (lum_data, 'erg/s'))
            ds = yt.load_hexahedral_mesh(data,conn,coords,length_unit="AU",bbox=bbox,geometry= 'polar')
        else:
            bbox = np.array([[P.ymin/r_AU,P.ymax/r_AU],[-np.pi,np.pi],[0.0,1.0]])
            data = dict(density = (lum_data, 'erg/s'))
            ds = yt.load_uniform_grid(data,lum_data.shape,length_unit="AU",bbox=bbox,geometry="polar")
    
        s = yt.SlicePlot(ds,'z', ('gas','density'),fontsize=slicefont)
        s.set_log(('gas','density'),True)
        slc = ds.slice('z',0.0)
    
        s.set_xlabel(r'x [$\mathrm{R_g}$]')#,fontsize=label_fontsize)                                                                                                             \                                              
        s.set_ylabel(r'y [$\mathrm{R_g}$]')
        s.set_zlim(field=('gas','density'),zmin=min_val.value,zmax=max_val.value)                                                                                                                                                            
        s.set_width(width,'AU')
        s.annotate_title(time_now)#,fontsize=title_fontsize)
        s.set_cmap(cmap='magma',field=('gas','density'))
        if not(freq_max) or (freq == freq_max):
            cbar_label=str(freq.value).split('.')[0]+r"$\,$"+"{0:latex}".format(freq.unit)+r'$\:\,\mathrm{\nu L_{\nu}'+'}}$'+r'$\;\,$[erg/s]'
        else:
            if freq < freq_max:
                cbar_label=str(freq.value).split('.')[0]+r"$-$"+str(freq_max.value).split('.')[0]+r"$\,$"+"{0:latex}".format(freq.unit)+r'$\:\,\mathrm{Luminosity'+'}}$'+r'$\;\,$[erg/s]'
            else:
                cbar_label=str(freq_max.value).split('.')[0]+r"$-$"+str(freq.value).split('.')[0]+r"$\,$"+"{0:latex}".format(freq.unit)+r'$\:\,\mathrm{Luminosity'+'}}$'+r'$\;\,$[erg/s]'

                
        s.set_colorbar_label(label=cbar_label,field=('gas','density'))

        if save:
            if high_res:
                s.save(data_dir + 'Images/' + self.name + '_' + str(freq.value).split('.')[0]+str(freq.unit) + '_' +str(name_index) +'.pdf', mpl_kwargs={'dpi':400,'format':'pdf','facecolor':'white','edgecolor':'white'})
            else:
                s.save(data_dir + 'Images/' + self.name + '_' + str(freq.value).split('.')[0]+str(freq.unit) + '_' +str(name_index) + '.png')    
        else:
            s.show()

            

    def doTownsendCooling(self,dt=P.dt,elecOnly=False,uniformCooling=0):
        if uniformCooling != 0:
            print('Under construction')
            return
        
        elif elecOnly:
            at_floor = np.less(self.temp,P.temp_min)
            
            dont_cool = at_floor * self.temp
            do_cool = np.logical_not(at_floor)*self.temp
            
            
#             T = temp;
#             energy = BOLTZMANN * T * surf_dens / ((GAMMA - 1.0) * MEANMOLECMASS * PROTONMASS);
        
                
            tau = self.surfdens * P.elec_opacity / 2.0;
        
            Lambda = (16.0 * P.stefboltz / 3.0)##;#//Lambda_k * pow((T / T_k ),alpha_k) * (Sigma_[j] / surf_dens);               \
                                                                                                                             
                                                                                                                                         
            Lambda = Lambda * (tau / (1.0 + tau*tau));
            Lambda = Lambda * pow(do_cool, 4.0);
        
            t_cool = self.energy / Lambda;
        
#             //T_f = T * pow((1.0 - (1.0 - alpha_k)*(dt / t_cool)),(1.0/(1.0 - alpha_k)));                                   \
                                                                                                                             
            T_f = do_cool * pow((1.0 + 3.0*(dt / t_cool)),(-1.0/3.0));
        
            under_floor = np.less(T_f,P.temp_min)
            became_nan  = np.isnan(T_f)
            
            do_flooring = np.logical_or(under_floor,became_nan)
            floored = do_flooring*P.temp_min
            not_floored = np.logical_not(do_flooring)*T_f
        
#             test_bool = do_flooring
#             print((1.0*test_bool).min(),(1.0*test_bool).max(),test_bool.mean())
#             if (T_f < P.temp_min || T_f != T_f) {
        
#               T_f = TEMP_MIN;
        
#             }
            
    
            T_f = floored + not_floored + dont_cool
            energy = P.boltzmann * T_f * self.surfdens / ((P.gamma - 1.0) * P.meanmolecmass * P.protonmass);
            self.cooledtemp=T_f
            self.coolrate = (((self.energy - energy)*self.area)/dt).to('erg/s')
            return                                                                                                      
    

        
        
        else:
            return
    
    #Define methods for making plots (separately for 1D (e.g., rad profs) 
    #and 2D (e.g. surfdens)datasets)
    
    

    def plot2D(self, data, grid=None, vmin=-1.0, vmax=1.0, units='cm', logscale=True, label='', cmap='viridis', width=None, center=[0.,0.,0.], xcen=0.0,ycen=0.0, cen_unit='AU', origin= ((0., 'AU'), (0., 'AU'),'window'), origin_unit='AU', save=False, high_res=False, filename='custom'):# 500*(self.mprim/(1e8*u.solMass)).value*(self.sep/(100*self.r_G)).value, save=False, high_res=False, filename='custom'):  
#         print(data.max(),data.min(),data.mean(),np.median(data))
        if width==None:
            width = 500*(self.mprim/(1e8*u.solMass)).value*(self.sep/(100*self.r_G)).value
        
        
                
        
        P = Parameters(data_dir + self.name + "/" + "variables.par")
        field_data = data #self._pullFieldData(field)
#         print(field_data.min(),field_data.max())
#         print(P.dt,P.ninterm,self.timestep)
        name_index = '0'*(4-len(str(self.timestep))) + str(self.timestep)
#         if self.timestep <= 12073:
#             num_orb = (self.timestep * P.dt * P.ninterm).to(t_orb)
#         else:
#             num_orb = ((12073 * P.dt * P.ninterm + (self.timestep-12073) * P.dt * (P.ninterm/10))).to(t_orb)
#         if self.timestep < 87:
#             num_orb = (self.timestep * P.dt * P.ninterm).to(t_orb)
#         else:
#             num_orb = ((87 * P.dt * 20 + (self.timestep-87) * P.dt * (P.ninterm))).to(t_orb)
        num_orb = self.time.to(self.t_orb) #(self.timestep * P.dt * P.ninterm).to(self.t_orb)
        time_now = "T = " + '{:.9f}'.format(num_orb.value)[:9] + r" $\mathrm{t_{orb}}$"
#         print(time_now)
    
        if (field_data.min() >= 0.0):# and field != binary):#(field != 'vy' and field != 'vx' and field != 'bound'):
            threshold_indices = field_data < vmin/1e3
            field_data[threshold_indices] = vmin/1e3
            #The above just sets a floor on the data which prevents visualization glitches in logspace

#         print(field_data.min(),field_data.max())
#         print(P.spacing,P.outputdir)
        if P.spacing == 'log':
#             print('Log spacing!')

#             with open(data_dir + self.name + "/" + "domain_x.dat") as file:
#                 phifaces = np.array([np.float64(line[:]) for line in file])*u.rad
        
#            xgrid = self.phifaces
#            ygrid = self.rfaces/r_AU
#            zgrid = np.array([0.0,1.0])
#            coords, conn = yt.hexahedral_connectivity(ygrid,xgrid,zgrid)


#            bbox = np.array([[np.min(ygrid).value,np.max(ygrid).value],
#                             [np.min(xgrid).value,np.max(xgrid).value],
#                             [np.min(zgrid),np.max(zgrid)]])
            data = dict(density = (field_data, units))
#            ds = yt.load_hexahedral_mesh(data,conn,coords,length_unit="AU",bbox=bbox,geometry= 'polar')
            ds = yt.load_hexahedral_mesh(grid.data,grid.conn,coords,length_unit="AU",bbox=grid.bbox,geometry= 'polar')
        else:
            bbox = np.array([[P.ymin/r_AU,P.ymax/r_AU],[-np.pi,np.pi],[0.0,1.0]])
            data = dict(density = (field_data, units))
            ds = yt.load_uniform_grid(data,field_data.shape,length_unit="AU",bbox=bbox,geometry="polar")
        
        s = yt.SlicePlot(ds,'z', ('gas','density'),center=center,fontsize=slicefont)
            
#         print(field.min.value,field.max.value)
#         if (field_data.min() > 0.0): and field != binary):    
        s.set_log(('gas','density'),logscale)
#         else:
#             s.set_log('density',False)
#         slc = ds.slice('z',0.0)
        s.set_xlabel('x [AU]')
        s.set_ylabel('y [AU]')
        s.set_zlim(field=('gas','density'),zmin=vmin,zmax=vmax) 
        s.set_origin(origin)
        s.annotate_title(time_now)
        s.set_center((ycen,xcen),unit=cen_unit)
        s.set_width(width,'AU')
        s.set_cmap(cmap=cmap,field=('gas','density'))
        s.set_colorbar_label(label=label,field=('gas','density'))
        
        if save:
            if high_res:
                s.save(data_dir + 'Images/' + self.name + '_' + filename + '_' +str(name_index) + '.pdf', mpl_kwargs={'dpi':400,'format':'pdf','facecolor':'white','edgecolor':'white'})
            else:
                s.save(data_dir + 'Images/' + self.name + '_' + filename + '_' +str(name_index) + '.png')
        else:
            s.show()
            
            

    def calcCFL(self, which=0, CVNR=np.sqrt(2), CVNL=0.05, tensor=False):
        P = Parameters(data_dir + self.name + "/" + "variables.par")
        cfl1 = 0.0
        cfl2 = 0.0
        cfl3 = 0.0
        cfl4 = 0.0
        cfl5 = 0.0
        cfl6 = 0.0
        cfl7 = 0.0
        cfl8 = 0.0
        cfl9 = 0.0
        
        
        if not(which) or which==1: #soundspeed
            try:
                cfl1_a = self.soundspeed/self.zone_size_phi
                cfl1_b = self.soundspeed/self.zone_size_r
            except AttributeError:
                self.getFieldData(soundspeed)
                cfl1_a = self.soundspeed/self.zone_size_phi
                cfl1_b = self.soundspeed/self.zone_size_r
                
            cfl1   = np.max([cfl1_a.cgs.value,cfl1_b.cgs.value],axis=0)*(1/u.s)
        
        if not(which) or which==2: #phi velocity
            try:
                cfl2 = np.abs((self.phivel-self.omegaframe*self.radius) - np.repeat((self.phivel-self.omegaframe*self.radius).mean(axis=1),P.nx,axis=0).reshape(P.ny,P.nx,1)) / self.zone_size_phi
            except AttributeError:
                self.getFieldData(phivel)
                cfl2 = np.abs((self.phivel-self.omegaframe*self.radius) - np.repeat((self.phivel-self.omegaframe*self.radius).mean(axis=1),P.nx,axis=0).reshape(P.ny,P.nx,1))/self.zone_size_phi
        
        if not(which) or which==3: #radial velocity
            try:
                cfl3 = np.abs(self.radvel)/self.zone_size_r
            except AttributeError:
                self.getFieldData(radvel)
                cfl3 = np.abs(self.radvel)/self.zone_size_r
        
        if not(which) or which==5 or which==6: #artificial viscosity     
            
            
            try:
                del_vphi = np.diff(self.phivel.cgs.value, n=1, axis=1, append=self.phivel[:,0,:].reshape(P.ny,1,1).cgs.value)*u.cm/u.s
            except AttributeError:
                self.getFieldData(phivel)
                del_vphi = np.diff(self.phivel.cgs.value, n=1, axis=1, append=self.phivel[:,0,:].reshape(P.ny,1,1).cgs.value)*u.cm/u.s

            try:
                del_vr = np.diff(self.radvel.cgs.value, n=1, axis=0, append=self.radvel[0,:,:].reshape(1,P.nx,1).cgs.value)*u.cm/u.s
            except AttributeError:
                self.getFieldData(radvel)
                del_vr = np.diff(self.radvel.cgs.value, n=1, axis=0, append=self.radvel[0,:,:].reshape(1,P.nx,1).cgs.value)*u.cm/u.s  
            if not(tensor):
                cfl5_a = del_vphi/self.zone_size_phi
                cfl5_b = del_vr/self.zone_size_r

                cfl5   = np.max([cfl5_a.cgs.value,cfl5_b.cgs.value],axis=0)*4.0*CVNR*(1.0/u.s) #NEED TO PULL CVNR FROM SOMEWHERE
            else:
                Ql = CVNR**2 * np.max([self.zone_size_phi,self.zone_size_r],axis=0)
                divv = (del_vphi/self.zone_size_phi)
                divv += (1./self.radius) * (np.diff(self.radvel.cgs.value*self.radius.cgs.value, n=1, axis=0, append=self.radvel[0,:,:].reshape(1,P.nx,1).cgs.value*self.radius[0,:,:].reshape(1,P.nx,1).cgs.value)*u.cm**2/u.s)/self.zone_size_r
                
                cfl5_a = 1.0/self.zone_size_phi**2
                cfl5_b = 1.0/self.zone_size_r**2
                
                cfl5 = np.max([cfl5_a.cgs.value,cfl5_b.cgs.value],axis=0)*4.0*Ql*Ql*np.abs(divv)
                
                
        
        if not(which) or which==6: #artificial viscosity (linear term)
            cfl6 = cfl5/CVNR*CVNL #NEED TO PULL CVNR AND CVNL FROM SOMEWHERE
            if which==6:
                cfl5=0.0
        
        if not(which) or which==7: #kinematic viscosity
            cfl7_a = 1.0/self.zone_size_phi.cgs.value
            cfl7_b = 1.0/self.zone_size_r.cgs.value
            
            try:
                cfl7   = 4.0*self.viscosity*((np.max([cfl7_a,cfl7_b],axis=0)*1.0/u.cm)**2)
            except AttributeError:
                self.getFieldData(viscosity)
                cfl7   = 4.0*self.viscosity*((np.max([cfl7_a,cfl7_b],axis=0)*1.0/u.cm)**2)
#             cfl7 = 0.0
#             cfl7   = 4.0*self.viscosity*(np.max([cfl7_a,cfl7_b])**2) #NEED TO ENSURE SELF.VISCOSITY IS CALCULATED
            
        if not(which) or which==9: #artificial viscosity heating            
            try:
                del_vphi = np.diff(self.phivel.cgs.value, n=1, axis=1, append=self.phivel[:,0,:].reshape(P.ny,1,1).cgs.value)*u.cm/u.s
            except AttributeError:
                self.getFieldData(phivel)
                del_vphi = np.diff(self.phivel.cgs.value, n=1, axis=1, append=self.phivel[:,0,:].reshape(P.ny,1,1).cgs.value)*u.cm/u.s

            try:
                del_vr = np.diff(self.radvel.cgs.value, n=1, axis=0, append=self.radvel[0,:,:].reshape(1,P.nx,1).cgs.value)*u.cm/u.s
            except AttributeError:
                self.getFieldData(radvel)
                del_vr = np.diff(self.radvel.cgs.value, n=1, axis=0, append=self.radvel[0,:,:].reshape(1,P.nx,1).cgs.value)*u.cm/u.s
                
            try:
                cfl9_a = (del_vphi < 0.0) * CVNR*CVNR*self.surfdens*del_vphi**2
            except AttributeError:
                self.getFieldData(surfdens)
                cfl9_a = (del_vphi < 0.0) * CVNR*CVNR*self.surfdens*del_vphi**2
            cfl9_b = (del_vr < 0.0) * CVNR*CVNR*self.surfdens*del_vr**2
            
            try:
                cfl9_a -= (del_vphi < 0.0) * CVNL*self.surfdens*self.soundspeed*del_vphi
            except AttributeError:
                self.getFieldData(soundspeed)
                cfl9_a -= (del_vphi < 0.0) * CVNL*self.surfdens*self.soundspeed*del_vphi
            cfl9_b -= (del_vr < 0.0) * CVNL*self.surfdens*self.soundspeed*del_vr
            
            cfl9_a *= -del_vphi/self.zone_size_phi
            cfl9_b *= -del_vr/self.zone_size_r
           
            try:
                cfl9   = 4.0 * (cfl9_a + cfl9_b) / self.energy
            except AttributeError:
                self.getFieldData(energy)
                cfl9   = 4.0 * (cfl9_a + cfl9_b) / self.energy
        
            cfl9 = (self.surfdens.cgs.value>P.noshockdens)*cfl9
            if which == 9:
                cfl9 = (cfl9>0)*cfl9 + (cfl9<=0.0)*1.0e-25*cfl9.unit

        dtime = P.cfl / np.sqrt(cfl1*cfl1 + cfl2*cfl2 + cfl3*cfl3 + cfl4*cfl4 + cfl5*cfl5 + cfl6*cfl6 + cfl7*cfl7 + cfl8*cfl8 + cfl9*cfl9)
        
        return dtime
        
        

    def getFieldCFL(self,which=0,CVNR=np.sqrt(2),CVNL=0.05,tensor=False):
        data = self.calcCFL(which,CVNR,CVNL).to(self.t_orb)
        
        if not(which):
            exec("self.cfl"+"=data")
        else:
            exec("self.cfl"+str(which)+"=data")
            
        return
            
            

    def plotCFL(self, which=0, vmin=None, vmax=None, units='s', logscale=True, label='', cmap='viridis_r', width=None, save=False, high_res=False, filename='custom', CVNR=np.sqrt(2), CVNL=0.05, tensor=False):  
        if width==None:
            width = 500*(self.mprim/(1e8*u.solMass)).value*(self.sep/(100*self.r_G)).value
        if vmin==None:
            vmin = (1e-6 * self.t_orb).to(u.s).value
        if vmax==None:
            vmax = (1e0 * self.t_orb).to(u.s).value
        P = Parameters(data_dir + self.name + "/" + "variables.par")
        
        
        if not(which):
            label = r'CFL $\Delta t$ [t$_\mathrm{orb}$]'
            
        elif which==1: #soundspeed
            label=r'c$_\mathrm{s}$ $\Delta \mathrm{t}$ [t$_\mathrm{orb}$]'
        
        elif which==2: #phi velocity
            label=r'v$_\phi$ $\Delta t$ [t$_\mathrm{orb}$]'
        
        elif which==3: #radial velocity
            label=r'v$_\mathrm{r}$ $\Delta t$ [t$_\mathrm{orb}$]'

        elif which==5 or which==6: #artificial viscosity
            label=r'Artificial Viscosity $\Delta t$ [t$_\mathrm{orb}$]'
    
        elif which==6: #artificial viscosity (linear term)
            label=r'Strong Shock $\Delta t$ [t$_\mathrm{orb}$]'
        
        elif which==7: #kinematic viscosity
            label=r'Kinematic Viscosity $\Delta t$ [t$_\mathrm{orb}$]'
           
        elif which==9: #artificial viscosity heating
            label=r'Artificial Viscous Heating $\Delta t$ [t$_\mathrm{orb}$]'
            
               
        field_data = self.calcCFL(which, CVNR, CVNL) #dtime 
#         print(field_data.max().to(self.t_orb),field_data.min().to(self.t_orb),field_data.mean().to(self.t_orb),np.median(field_data).to(self.t_orb))
        
        name_index = '0'*(4-len(str(self.timestep))) + str(self.timestep)
        num_orb = self.time.to(self.t_orb) 
        time_now = "T = " + '{:.9f}'.format(num_orb.value)[:9] + r" $\mathrm{t_{orb}}$"

        if (field_data.min() >= 0.0) and (logscale):
            threshold_indices = field_data.cgs.value < vmin/1e3
            field_data[threshold_indices] = vmin*(field_data.cgs.unit)/1e3
            #The above just sets a floor on the data which prevents visualization glitches in logspace
        if P.spacing == 'log':
            xgrid = self.phifaces
            ygrid = self.rfaces/r_AU
            zgrid = np.array([0.0,1.0])
            coords, conn = yt.hexahedral_connectivity(ygrid,xgrid,zgrid)
            bbox = np.array([[np.min(ygrid).value,np.max(ygrid).value],
                             [np.min(xgrid).value,np.max(xgrid).value],
                             [np.min(zgrid),np.max(zgrid)]])
            data = dict(density = (field_data.to(self.t_orb), units))
            ds = yt.load_hexahedral_mesh(data,conn,coords,length_unit="AU",bbox=bbox,geometry= 'polar')
        else:
            bbox = np.array([[P.ymin/r_AU,P.ymax/r_AU],[-np.pi,np.pi],[0.0,1.0]])
            data = dict(density = (field_data.to(self.t_orb), units))
            ds = yt.load_uniform_grid(data,field_data.shape,length_unit="AU",bbox=bbox,geometry="polar")
        
        s = yt.SlicePlot(ds,'z', ('gas','density'),fontsize=slicefont)
   
        s.set_log(('gas','density'),logscale)
#         else:
#             s.set_log('density',False)
#         slc = ds.slice('z',0.0)
        s.set_xlabel('x [AU]')
        s.set_ylabel('y [AU]')
        s.set_zlim(field=('gas','density'),zmin=(vmin*u.s).to(self.t_orb).value,zmax=(vmax*u.s).to(self.t_orb).value) 
        s.annotate_title(time_now)
        s.set_width(width,'AU')
        s.set_cmap(cmap=cmap,field=('gas','density'))
        s.set_colorbar_label(label=label,field=('gas','density'))
        
        if save:
            if high_res:
                s.save(data_dir + 'Images/' + self.name + '_' + filename + '_' +str(name_index) + '.pdf', mpl_kwargs={'dpi':400,'format':'pdf','facecolor':'white','edgecolor':'white'})
            else:
                s.save(data_dir + 'Images/' + self.name + '_' + filename + '_' +str(name_index) + '.png')
        else:
            s.show()
    


##############################
### DEFINE EMISSION OBJECT ###
##############################
    
class fargoEmission():   

    def __init__(self,sim_name,nghx=0,nghy=3):
        P = Parameters(data_dir + sim_name + "/" + "variables.par")
        #IN GENERAL: Things that are unique to a SPECIFIC fargo dataset should be defined/assigned here
        #Things that would be common to all of them (I think just methods for me) should be assigned outside of init
        self.name=sim_name
        self.mprim = P.mprim0*u.solMass
        self.q     = P.massratio
        self.r_G = grv_const * (P.mprim0*u.solMass).to(u.g) / spd_lgt**2
        self.t_orb = (2.0*np.pi * np.sqrt((P.separation * self.r_G)**3 / (grv_const * (1.0 + P.massratio)*P.mprim0*u.solMass))).to(u.s)
        
        self.t_orb    = u.def_unit('t_orb', self.t_orb,format={'latex':r't_\mathrm{orb}'})
        self.r_G      = u.def_unit('r_G',self.r_G,format={'latex':r'r_\mathrm{G}'})
        
        self.sep = P.separation*self.r_G
        self.dt  = P.dt

        unit=P.ymax.unit
        if P.spacing[:3] == 'lin':
            dy = (P.ymax-P.ymin)/P.ny
#         print(dy)
            self.rfaces = [P.ymin.value+dy*(j-nghy) for j in range(0,P.ny+2*nghy+1)]*unit
        elif P.spacing[:3] == 'log':
            dy = (np.log(P.ymax.value) - np.log(P.ymin.value))/P.ny
            self.rfaces = [np.exp(np.log(P.ymin.value) + dy*(j-nghy)) for j in range(0,P.ny+2*nghy+1)]*unit
        self.rfaces = self.rfaces[nghy:-nghy]
            
            
        dx = (P.xmax-P.xmin)/P.nx
        self.phifaces = [P.xmin+dx*(i-nghx) for i in range(0,P.nx+2*nghx+1)]*u.rad

        areas = np.array([(P.dx/2) * (self.rfaces[i+1].cgs.value**2 - self.rfaces[i].cgs.value**2) for i in range(P.ny)])
        self.areas = np.repeat(areas,P.nx,axis=0).reshape(P.ny,P.nx,1)*u.cm**2



    def getLumData(self,which=1):
   
        if which == 1:
            file = open(data_dir + self.name + '/cooling.dat')
        elif which == 2:
            file = open(data_dir + sim_name + '/cooling2.dat')
        elif (which == 'both') or (which == 'all'):
            file = open(data_dir + self.name + '/cooling.dat')
            file2 = open(data_dir + self.name + '/cooling2.dat')
#        lst = []
        lst = np.array([[np.float64(el) for el in line.split()] for line in file]) * u.erg
#        for line in file:
#          lst.append(line.split())
    
        #dt = float(lst[1][1]) - float(lst[0][1])
        if which == 1:
            self.cooldata = lst #np.float64(np.asarray(lst))
        elif which == 2:
            self.seccooldata = lst
        elif (which == 'both') or (which == 'all'):
            lst2 = np.array([[np.float64(el) for el in line.split()] for line in file2]) * u.erg
            self.cooldata = lst
            self.seccooldata = lst2
    


    def getAreaData(self):
        file = open(data_dir + self.name + '/coolarea.dat')
        lst = np.array([[np.float64(el) for el in line.split()] for line in file]) * u.cm**2
        
        self.coolarea = lst
        


    def _stefanBoltzmann(self,temp):
        const = 2.0*np.pi*(np.pi * boltzmann)**4 / (15.0 * spd_lgt**2 * planck_const**3)
        return 2.0 * const * temp**4



    def _planck(self,freq,temp):
        const = 2.0*planck_const/(spd_lgt**2)
        term  = (freq**3)/(np.exp(planck_const*freq/(boltzmann*temp)) - 1.0)

        return const*term



    def getLumAreas(self,which=1):
        
#        em_rate = self.cooldata/self.dt
        try:
            self.temps = 10**(np.tile(np.arange(100),self.cooldata.shape[0]).reshape(self.cooldata.shape[0],100)/10.0) * u.K
        except(AttributeError):
            self.getLumData(which)
            self.temps = 10**(np.tile(np.arange(100),self.cooldata.shape[0]).reshape(self.cooldata.shape[0],100)/10.0) * u.K

        if which == 1:
            self.areas = (self.cooldata[:,2:]/self.dt) / self._stefanBoltzmann(self.temps)
        elif which == 2:
            self.secareas = (self.seccooldata[:,2:]/self.dt) / self._stefanBoltzmann(self.temps)
        elif (which == 'both') or (which == 'all'):
            self.areas = (self.cooldata[:,2:]/self.dt) / self._stefanBoltzmann(self.temps)
            self.secareas = (self.seccooldata[:,2:]/self.dt) / self._stefanBoltzmann(self.temps)



    def getLumAtFreq(self,freq,which=1):
        freq = freq.to(u.Hz, equivalencies=u.spectral())
        if which == 1:
            return np.sum(freq*self.areas*self._planck(freq,self.temps),axis=1).cgs
        elif which == 2:
            return np.sum(freq*self.secareas*self._planck(freq,self.temps),axis=1).cgs
        elif (which == 'both') or (which == 'all'):
            return np.sum(freq*self.areas*self._planck(freq,self.temps),axis=1).cgs + np.sum(freq*self.secareas*self._planck(freq,self.temps),axis=1).cgs



    def getSED(self,freq_min=1e13,freq_max=1e22,n_freqs=5000,time=None,which=1):
        try:
            self.temps = 10**(np.tile(np.arange(100),self.cooldata.shape[0]).reshape(self.cooldata.shape[0],100)/10.0) * u.K
        except(AttributeError):
            self.getLumData(which)
            self.temps = 10**(np.tile(np.arange(100),self.cooldata.shape[0]).reshape(self.cooldata.shape[0],100)/10.0) * u.K
        
        self.freqs = np.logspace(np.log10(freq_min),np.log10(freq_max),n_freqs)*u.Hz
        if time == None:
            self.sed = np.repeat(self.freqs,self.areas.shape[0]).reshape(self.freqs.shape[0],self.areas.shape[0]) * np.dot(self.areas, np.array([self._planck(self.freqs,temp).to(u.erg/u.cm**2).value for temp in self.temps[0,:]])*(u.erg/u.cm**2)).T
        else:
            self.sed = self.freqs * np.dot(self.areas[time,:], np.array([self._planck(self.freqs,temp).to(u.erg/u.cm**2).value for temp in self.temps[0,:]])*(u.erg/u.cm**2)).T



    def plotLC(self,freq,which=1):
        try:
            plt.plot(np.arange(self.areas.shape[0])*self.dt.to(self.t_orb),self.getLumAtFreq(freq,which))
        except(AttributeError):
            self.getLumAreas(which)
            plt.plot(np.arange(self.areas.shape[0])*self.dt.to(self.t_orb),self.getLumAtFreq(freq,which))

        plt.yscale('log')
        plt.ylabel(str(freq.value).split('.')[0]+r"$\,$"+"{0:latex}".format(freq.unit)+r'$\:\,\mathrm{\nu L_{\nu}'+'}}$'+r'$\;\,$[erg/s]')
        plt.xlabel('Time [t$_\mathrm{orb}$]')
        plt.show()
        


    def highlightRegion(self,arr,bg=surfdens,width=None,save=False,high_res=False):
        # want to plot true points from arr in red alpha=1 overtop an alpha=0.3 background of the field specified by bg
        if width==None:
            width = 500*(self.mprim/(1e8*u.solMass)).value*(self.sep/(100*self.r_G)).value
        
        P = Parameters(data_dir + self.name + "/" + "variables.par")
        field_data = self._pullFieldData(bg)
        name_index = '0'*(4-len(str(self.timestep))) + str(self.timestep)

        num_orb = self.time.to(self.t_orb) #(self.timestep * P.dt * P.ninterm).to(self.t_orb)
        time_now = "T = " + '{:.9f}'.format(num_orb.value)[:9] + r" $\mathrm{t_{orb}}$"
    
        if (field_data.min() >= 0.0 and bg != binary):#(field != 'vy' and field != 'vx' and field != 'bound'):
            threshold_indices = field_data < bg.min/1e3
            field_data[threshold_indices] = bg.min/1e3
            #The above just sets a floor on the data which prevents visualization glitches in logspace

        if P.spacing == 'log':
 
            xgrid = self.phifaces
            ygrid = self.rfaces/r_AU
            zgrid = np.array([0.0,1.0])
            coords, conn = yt.hexahedral_connectivity(ygrid,xgrid,zgrid)


            bbox = np.array([[np.min(ygrid).value,np.max(ygrid).value],
                             [np.min(xgrid).value,np.max(xgrid).value],
                             [np.min(zgrid),np.max(zgrid)]])
            data = dict(density = (field_data, bg.yt_units))
            ds = yt.load_hexahedral_mesh(data,conn,coords,length_unit="AU",bbox=bbox,geometry= 'polar')
        else:
            bbox = np.array([[P.ymin/r_AU,P.ymax/r_AU],[-np.pi,np.pi],[0.0,1.0]])
            data = dict(density = (field_data, bg.yt_units))
            ds = yt.load_uniform_grid(data,field_data.shape,length_unit="AU",bbox=bbox,geometry="polar")
        
        s = yt.SlicePlot(ds,'z', ('gas','density'),fontsize=slicefont)
            
        if (field_data.min() > 0.0 and bg != binary):    
            s.set_log(('gas','density'),True)
        else:
            s.set_log(('gas','density'),False)
        slc = ds.slice('z',0.0)
        s.set_xlabel('x [AU]')
        s.set_ylabel('y [AU]')
        s.set_zlim(field=('gas','density'),zmin=bg.min.value,zmax=bg.max.value) 
        s.annotate_title(time_now)
        s.set_width(width,'AU')
        s.set_cmap(cmap=bg.cmap,field=('gas','density'))
        s.set_colorbar_label(label=bg.label,field=('gas','density'))
        s.annotate_clumps(arr)
        
        if save:
            if high_res:
                s.save(data_dir + 'Images/' + self.name + '_gas' + bg.name + '_' +str(name_index) + '.pdf', mpl_kwargs={'dpi':400,'format':'pdf','facecolor':'white','edgecolor':'white'})
            else:
                s.save(data_dir + 'Images/' + self.name + '_gas' + bg.name + '_' +str(name_index) + '.png',mpl_kwargs={'dpi':400})
        else:
            s.show()





####################################
### DEFINE SHOCK EMISSION OBJECT ###
####################################
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class fargoShock():   

    def __init__(self,sim_name,grid,nghx=0,nghy=3,n_orb_window=10,which='artvisc_1d_Y_raw.dat'):
        P = Parameters(data_dir + sim_name + "/" + "variables.par")
        #IN GENERAL: Things that are unique to a SPECIFIC fargo dataset should be defined/assigned here
        #Things that would be common to all of them (I think just methods for me) should be assigned outside of init
        self.name  = sim_name
        self.ny    = P.ny
        self.t_orb = grid.t_orb
        self.r_G   = grid.r_G
        self.bins  = P.numtempbins

        data = fromfile(data_dir + self.name + '/monitor/gas/' + which) #shocks_1d_Y_raw.dat')
        self.shockenergy = np.array(data).reshape(int(len(data)/self.ny), self.ny) * u.erg
        self.time = np.arange(len(self.shockenergy[:,0])) * P.dt.to(self.t_orb)

#        n_orb_window = 10
        n = int(n_orb_window / 2 * 1000)

        self.shockavg = moving_average(self.shockenergy[:,:].sum(axis=1).cgs.value,n=n)#[shock.shockenergy[i-5000:i+5000,:700].sum(axis=1).mean().cgs.value for i in range(5000,shock.shockenergy.shape[0]-5000)]
        
        
    def calcFFT(self,which=None,fftname=None,frqname=None):
        if which is not None:
            lc = which.sum(axis=1)
        else:
            lc = self.shockenergy.sum(axis=1)

        dt  = self.time[1] - self.time[0]
        Fs  = 1/dt #Sampling 
        l   = len(lc)
        T   = l/Fs
        k   = np.arange(l)
        frq = k/T

        if (fftname is not None) ^ (frqname is not None): #if only one of the attribute name variables is being customized
            print("WARNING: You only specified a custom attribute name for one of the two calcFFT() products. An existing attribute may have been overwritten.")
        if fftname is not None:
            exec(f"self.{fftname} = np.fft.rfft(lc)/l")
        else:
            self.fft = np.fft.rfft(lc)/l
        if frqname is not None:
            exec(f"self.{frqname} = np.fft.rfftfreq(len(lc),dt).to(1/self.t_orb)")
        else:
            self.freq = np.fft.rfftfreq(len(lc),dt).to(1/self.t_orb)

#        self.fft = np.fft.rfft(self.shockenergy.sum(axis=1))/l
#        self.freq = np.fft.rfftfreq(self.shockenergy.shape[0],dt).to(1/self.t_orb)
#        self.freq = frq[range(int(l/2) + 1)].to(1/self.t_orb)
#blobless_fft_opt = np.fft.rfft(blobless_lc_opt[i_steadystate:])/l

    def getArtViscData(self):
        data = fromfile(data_dir + self.name + '/monitor/gas/artvisc_1d_Y_raw.dat')
        self.avenergy = np.array(data).reshape(int(len(data)/self.ny), self.ny) * u.erg

    def getTempBinnedData(self):
        data = np.loadtxt(data_dir + self.name + '/e_tempbinned.dat',delimiter="\t") #fromfile(data_dir + self.name + '/e_tempbinned.dat')
        self.tbinnede = data #np.array(data)#.reshape(int(len(data)/self.ny), self.ny) * u.erg


############################
### DEFINE PLANET OBJECT ###
############################

class fargoPlanet():   

    def __init__(self,sim_name,grid,nghx=0,nghy=3):
        P = Parameters(data_dir + sim_name + "/" + "variables.par")
        #IN GENERAL: Things that are unique to a SPECIFIC fargo dataset should be defined/assigned here
        #Things that would be common to all of them (I think just methods for me) should be assigned outside of init
        self.name  = sim_name
        self.ny    = P.ny
        self.t_orb = grid.t_orb
        self.r_G   = grid.r_G
        self.mdot_edd = grid.mdot_edd
        self.L_edd = grid.L_edd

#        with open(data_dir + sim_name + "/" + "planet0.dat") as file:
#            for i, line in enumerate(file):
#                if line.split("\t")[0] == str(self.timestep):
#            self.time = np.float64(line.split("\t")[8])*u.s
#            self.omegaframe = np.float64(line.split("\t")[9])*(1/u.s)
#            self.mdot = np.float64(line.splot("\t")[-4])*u.g/u.s
        temp = np.loadtxt(data_dir + sim_name + "/" + "bigplanet0.dat")
        self.time_raw = (temp[:,8]*u.s).to(self.t_orb)
    
        self.time, uniq_ind = np.unique(self.time_raw,return_index=True)
    
        self.xpos = (temp[:,1]*u.cm)[uniq_ind].to(self.r_G)
        self.ypos = (temp[:,2]*u.cm)[uniq_ind].to(self.r_G)
        self.xvel = temp[:,4][uniq_ind]*u.cm/u.s
        self.yvel = temp[:,5][uniq_ind]*u.cm/u.s
#        self.time = (temp[:,8]*u.s).to(self.t_orb)
        self.omgf = temp[:,9][uniq_ind]*(1/u.s)
        self.mdot = temp[:,10][uniq_ind]*u.g/u.s
        self.xmom = temp[:,11][uniq_ind]#CHECK UNITS IN FARGO CODE
        self.ymom = temp[:,12][uniq_ind]#CHECK UNITS IN FARGO CODE
        self.angmom = temp[:,13][uniq_ind]#CHECK UNITS IN FARGO CODE
        self.mdisk = temp[:,14][uniq_ind]*u.g
        self.mdiskb = temp[:,15][uniq_ind]*u.g
        self.mdiskbb = temp[:,16][uniq_ind]*u.g

    def calcFFT(self,which=None,fftname=None,frqname=None):
        if which is not None:
            lc = which
        else:
            lc = self.mdot

        dt  = self.time[1] - self.time[0]
        Fs  = 1/dt #Sampling 
        l   = len(lc)
        T   = l/Fs
        k   = np.arange(l)
        frq = k/T

        if (fftname is not None) ^ (frqname is not None): #if only one of the attribute name variables is being customized
            print("WARNING: You only specified a custom attribute name for one of the two calcFFT() products. An existing attribute may have been overwritten.")
        if fftname is not None:
            exec(f"self.{fftname} = np.fft.rfft(lc)/l")
        else:
            self.fft = np.fft.rfft(lc)/l
        if frqname is not None:
            exec(f"self.{frqname} = np.fft.rfftfreq(len(lc),dt).to(1/self.t_orb)")
        else:
            self.freq = np.fft.rfftfreq(len(lc),dt).to(1/self.t_orb)




########################
### VIDEO GENERATION ###
########################

def gatherFiles(field,sim_name,pltfiledir,startval=0):
    os.chdir(pltfiledir)

    plt_files = glob.glob(sim_name+"/*"+field+"*_*"+ str(startval) +"*.png")

#     print(plt_files)
    
    return plt_files

def makeMovie(field,sim_name,pltfiledir="/mnt/storage/fargoData/Images",startval="0000",moviedir="/mnt/storage/fargoData/Movies",framerate=30):
    plt_files = gatherFiles(field,sim_name,pltfiledir,startval)
    if plt_files == []:
        print("ERROR: No images found for the specified field.")
        return
    os.chdir(moviedir)
    timestamp   = str(dt.datetime.now().isoformat())
    for plt_file in sort(plt_files):
#       print("1")
      ind = plt_file.find(str(startval))
#       print(ind)
#       print("2")
      if ind:
        base_name = plt_file[:ind]
        ext_name = plt_file[ind+len(str(startval)):]
        

#         print("base_name: ", base_name)
#         print("ext_name: ", ext_name)
#         print(base_name + ext_name[1:-4] +timestamp[0:10] + ".mp4")
    
        os.system(f"mkdir {base_name.split('/')[0]}")
        os.system("ffmpeg -start_number "+ str(startval) + " -y -framerate " + str(framerate) + " -i " + pltfiledir + '/' + base_name +
                  "%0" + str(len(startval)) + "d" + ext_name + " -c:v libx264 " + '-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" ' + "-preset slow -crf 15 -pix_fmt yuv420p " + base_name + ext_name[1:-4] +
                  timestamp[0:10] + ".mp4")

    



def isTorqueSaturated(alpha=0.1,M_prim=1e8,f=1e-2,q=1e-2,r_sec=1e2):
    a = 116.959 * (alpha/0.1) * (M_prim/1e7)**(-0.25) * (f/1e-2)**(-1.25) * (q/1e-3)**(-13/12) * (r_sec/1e2)**(5/8) # * (del_ri/r_Hill)**(17/4) #note: generally, del_ri/r_Hill=1.0
    return a,a<=1/np.exp(1)

def isDiskGapped(alpha=0.1,M_prim=1e8,mdot=0.1,f=1e-2,q=1e-2,r_sec=1e2,lam=1.0):
    a,is_saturated = isTorqueSaturated(alpha,M_prim,f,q,r_sec)
    T_neu_max = 3.4387e49 * (alpha/0.1)**(-2) * (M_prim/1e7)**(2.5) * (f/1e-2)**(2.5) * (q/1e-3)**(2.5) * (r_sec/1e2)**(0.25) * (1.0 + (q/3)**(1/3))**(-115/24)
    
    if is_saturated:
        T_nes_at_r_nes = 2.154e49 * (alpha/0.1)**(-2/11) * (M_prim/1e7)**(45/22) * (f/1e-2)**(5/22) * (q/1e-3)**(5/11) * (r_sec/1e2)**(61/44) * (0.4*(-4/17)*(np.log(a) - np.log(-np.log(a))))**(13/11)
        torque_mo = np.min([T_nes_at_r_nes,T_neu_max])
    else:
        torque_mo = T_neu_max
    torque_mg = 1.386e49 * (alpha/0.1)**(0.5) * (M_prim/1e7)**(1.25) * (mdot/0.1)**(5/8) * (q/1e-3)**(5/8) * (r_sec/1e2)**(-3/8) * lam**(-11/16)
    return torque_mo,torque_mg,torque_mg<torque_mo
