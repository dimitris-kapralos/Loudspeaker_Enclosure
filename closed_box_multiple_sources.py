import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import mpmath
from scipy.special import gamma, j1, jv, spherical_jn, spherical_yn


# Speed of sound propagation in air, m/s
SOUND_CELERITY = 343
# Air density, kg/m^3
R_0 = 1.18
# Molecular mean free path length between collisions
LM = 6 * 10 ** (-8)


class Loudspeaker:
    """Loudspeaker parameters class.

    Re: voice coil resistance, Ω
    Le: voice coil inductance, H
    e_g: voice coil height, V
    Qes: electrical Q factor
    Qms: mechanical Q factor
    fs: resonant frequency, Hz
    Vas: equivalent volume of compliance, m^3
    Qts: total Q factor
    Vab: volume of the enclosure, m^3
    Cms: mechanical compliance
    Mms: mechanical mass
    Bl: force factor
    pmax: maximum linear excursion
    Vmax: maximum volume displacement
    a: Diaphragm radius, m
    Sd: Diaphragm surface area, m^2
    Rms: Mechanical resistance
    """

    def __init__(self, lsp_par):
        """Initialize the loudspeaker object."""
        self.Re = lsp_par["Re"]
        self.Le = lsp_par["Le"]
        self.e_g = lsp_par["e_g"]
        self.Qes = lsp_par["Qes"]
        self.Qms = lsp_par["Qms"]
        self.fs = lsp_par["fs"]
        self.Vas = lsp_par["Vas"]
        self.Qts = lsp_par["Qts"]
        self.Cms = lsp_par["Cms"]
        self.Mms = lsp_par["Mms"]
        self.Bl = lsp_par["Bl"]
        if "a" in lsp_par:
            self.a = lsp_par["a"]
            self.Sd = np.pi * self.a**2
        if "Sd" in lsp_par:
            self.Sd = lsp_par["Sd"]  # diaphragm area in m^2
            self.a = np.sqrt(self.Sd / np.pi)  # radius of the diaphragm
        self.Rms = 1 / self.Qms * np.sqrt(self.Mms / self.Cms)


class BassReflexEnclosure:
    """Calculates the loudspeaker system frequency response and impedance."""

    def __init__(self, lsp_par, number_of_speakers, port_shape):
        """Initialize a loudspeaker system object."""
        self.number_of_speakers = number_of_speakers
        self.port_shape = port_shape

        # Parameters
        self.lsp = Loudspeaker(lsp_par)
        
        self.Va1 = 9.15 # volume of the enclosure without the volume of # lining material in m^3

        
        self.Vm1 = 3.05  # volume of the lining material in m^3

        
        self.Vb1 = 12.2  # total volume of the enclosure in m^3

        self.lx1 = 0.15  # length of the enclosure in m
        self.ly1 = 0.32  # width of the enclosure in m
        self.lz1 = 0.192  # height of the enclosure in m
        
        self.porosity = 0.99  # porosity
        self.P_0 = 10**5  # atmospheric pressure
        self.u = 0.03  # flow velocity in the material in m/s
        self.m = 1.86 * 10 ** (-5)  # viscosity coefficient in N.s/m^2
        self.r = 50 * 10 ** (-6)  # fiber diameter
        self.truncation_limit = 10  # Truncation limit for the double summation
        self.d = 0.064  # the thickness of the lining material
        
        
    # Calculate simplified impedance of the box     
    def calculate_simplified_box_impedance_Zab(self, f, B, Va, Vm, lx, ly):
        Mab = B*R_0/(np.pi * self.lsp.a)

        CAA = (Va*10**(-3))/(1.4*self.P_0)
        CAM = (Vm*10**(-3))/self.P_0
        
        Xab = 2*np.pi*f*Mab - 1 /(2*np.pi*f*(CAA + CAM))
        
        Ram = R_0* SOUND_CELERITY / (lx * ly) 
        
        Rab = Ram/ ((1+ Va/(1.4*Vm))**2 + (2*np.pi*f)**2 *Ram**2 * CAA**2)
        #print(Rab)
        
        Zab = 1*(Rab + 1j*Xab)
    
        return Zab