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

        self.fb = 36  # tuning frequency
        self.Vab1 = 12.58
        self.Vab2 = 25.2
        self.Vab3 = 37.750
        self.Vab4 = 50.32
        self.t1 = 0.59  # length of the port
        self.t2 = 0.41
        self.t3 = 0.31
        self.t4 = 0.30
        self.Sp1 = 0.0035  # area of the port
        self.Sp2 = 0.005
        self.Sp3 = 0.00611
        self.Sp4 = 0.0071
        
        self.Va1 = 8.57 # volume of the enclosure without the volume of # lining material in m^3
        self.Va2 = 17.2
        self.Va3 = 25.74
        self.Va4 = 34.31
        
        self.Vm1 = 2.86  # volume of the lining material in m^3
        self.Vm2 = 6.1
        self.Vm3 = 8.58
        self.Vm4 = 11.44
        
        self.Vb1 = 11.43  # total volume of the enclosure in m^3
        self.Vb2 = 22.9
        self.Vb3 = 34.32
        self.Vb4 = 45.75
        
        self.a_p1 = np.sqrt(self.Sp1 / np.pi)  # radius of the port
        self.a_p2 = np.sqrt(self.Sp2 / np.pi)
        self.a_p3 = np.sqrt(self.Sp3 / np.pi)
        self.a_p4 = np.sqrt(self.Sp4 / np.pi)

        self.porosity = 0.99  # porosity
        self.P_0 = 10**5  # atmospheric pressure
        self.u = 0.03  # flow velocity in the material in m/s
        self.m = 1.86 * 10 ** (-5)  # viscosity coefficient in N.s/m^2
        self.r = 50 * 10 ** (-6)  # fiber diameter
        self.truncation_limit = 10  # Truncation limit for the double summation
        self.d = 0.064  # the thickness of the lining material

        self.lx1 = 0.15  # width of the enclosure
        self.ly1 = 0.401  # height of the enclosure
        self.lz1 = 0.210  # depth of the enclosure
        
        self.lx2 = 0.15  # width of the enclosure
        self.ly2 = 0.63
        self.lz2 = 0.192
        
        self.lx3 = 0.2  # width of the enclosure
        self.ly3 = 0.671
        self.lz3 = 0.192
        
        self.lx4 = 0.2  # width of the enclosure
        self.ly4 = 0.894
        self.lz4 = 0.192

        self.x1 = 0.075  # distance from the center of the diaphragm to the box wall
        self.x2 = 0.1  # distance from the center of the diaphragm to the box wall
        self.x3 = 0.1  # distance from the center of the diaphragm to the box wall
        self.x4 = 0.1
        self.y1 = 0.15  # distance from the center of the diaphragm to the box wall
        self.yp1 = 0.3  # distance from the center of the diaphragm to the box wall
        self.y2 = 0.095
        self.yp2 = 0.52
        self.y3 = 0.32
        self.yp3 = 0.6
        self.y4 = 0.32
        self.yp4 = 0.8

        self.a2 = 0.15  # width of the port
        self.b2 = 0.034  # height of the port
        self.d1 = 0.2  # distance between the diaphragms

        # Values of coefficients (Table 7.1)
        self.a3 = 0.0858
        self.a4 = 0.175
        self.b3 = 0.7
        self.b4 = 0.59
        self.calculate_R_f()
        
    def calculate_R_f(self):
        """Calculate flow resistance of lining material, Eq. 7.8."""
        self.R_f = (
            (4 * self.m * (1 - self.porosity)) / (self.porosity * self.r**2)
        ) * (
            (1 - 4 / np.pi * (1 - self.porosity))
            / (2 + np.log((self.m * self.porosity) / (2 * self.r * R_0 * self.u)))
            + (6 / np.pi) * (1 - self.porosity)
        )
        
    # Calculate the simplified box impedance based on equation 7.2-7.7         
    def calculate_box_impedance_Zab(self, f):
        if self.number_of_speakers == "one" :
            Mab = 0.53*R_0/(np.pi * self.lsp.a)
                
            CAA = (self.Va1*10**(-3))/(1.4*self.P_0)
            CAM = (self.Vm1*10**(-3))/self.P_0    
            
            Xab = 2*np.pi*f*Mab - 1 /(2*np.pi*f*(CAA + CAM))
            
            Rab = 7111/ ((1+ self.Va1/(1.4*self.Vm1))**2 + (2*np.pi*f)**2 *7111**2 * CAA**2)
                
        elif self.number_of_speakers == "two" :
            Mab = 0.53*R_0/(np.pi * self.lsp.a)

            CAA = (self.Va2*10**(-3))/(1.4*self.P_0)
            CAM = (self.Vm2*10**(-3))/self.P_0
            
            Xab = 2*np.pi*f*Mab - 1 /(2*np.pi*f*(CAA + CAM))
            
            Rab = 4480/ ((1+ self.Va2/(1.4*self.Vm2))**2 + (2*np.pi*f)**2 *4480**2 * CAA**2)
            #print(Rab)
            
        elif self.number_of_speakers == "three" :
            Mab = 0.53*R_0/(np.pi * self.lsp.a)
            CAA = (self.Va3*10**(-3))/(1.4*self.P_0)
            CAM = (self.Vm3*10**(-3))/self.P_0
            
            Xab = 2*np.pi*f*Mab - 1 /(2*np.pi*f*(CAA + CAM))
            
            Rab = 3179/ ((1+ self.Va3/(1.4*self.Vm3))**2 + (2*np.pi*f)**2 *3179**2 * CAA**2)
            
        elif self.number_of_speakers == "four" :
            Mab = 0.53*R_0/(np.pi * self.lsp.a)
            CAA = (self.Va4*10**(-3))/(1.4*self.P_0)
            CAM = (self.Vm4*10**(-3))/self.P_0
            
            Xab = 2*np.pi*f*Mab - 1 /(2*np.pi*f*(CAA + CAM))
            
            Rab = 2386/ ((1+ self.Va4/(1.4*self.Vm4))**2 + (2*np.pi*f)**2 *2386**2 * CAA**2)
        
        Zab = (Rab + 1j*Xab)
        
        
        return Zab
        
    # Calculate the radiation impedance of the diaphragm based on equation 13.339 - 13.345 with Mellow's method    
    def calculate_Za1(self, f,):
        k = (2 * np.pi * f / SOUND_CELERITY) * ((1 + self.a3 *(self.R_f/f)**self.b3) -1j * self.a4 * (self.R_f/f)**self.b4)

        # Calculate the Bessel and Struve functions
        J1 = jv(1, k * self.lsp.a) 
        H1 = mpmath.struveh(1, k * self.lsp.a) 

        z11 = R_0 * SOUND_CELERITY * ((1 - (J1**2/ (k * self.lsp.a))) + 1j* (H1/ (k * self.lsp.a))) 

        # Calculate Z12
        z12 =  (2 * R_0 * SOUND_CELERITY) / np.sqrt(np.pi)
        z13  = (2 * R_0 * SOUND_CELERITY) / np.sqrt(np.pi)
        z14 = (2 * R_0 * SOUND_CELERITY) / np.sqrt(np.pi)
        sum_mn = 0
        for m in range(self.truncation_limit+1):
            for n in range(self.truncation_limit+1):
                term1 = ((k * self.lsp.a) / (k * 0.2)) ** m
                term2 = ((k * self.lsp.a) / (k * 0.2)) ** n
                term3 = gamma(m + n + 0.5)
                term4 = jv(m + 1, k * self.lsp.a) * jv(n + 1, k * self.lsp.a)
                term5 = 1 / (np.math.factorial(m) * np.math.factorial(n))
                term6 = spherical_jn(m + n, k * 0.2) + 1j * spherical_yn(m + n, k * 0.2)
                sum_mn += term1 * term2 * term3 * term4 * term5 * term6
        z12 *= sum_mn 
        z13 *= sum_mn  
        z14 *= sum_mn
        
        if self.number_of_speakers == "one" :
            Za1 =  self.lsp.a**2 * z11 /  self.lsp.a**2
            
        elif self.number_of_speakers == "two" :
            Za1 =  (self.lsp.a**2 * z11 + 2* self.lsp.a * self.lsp.a * z12 ) / (self.lsp.a**2 + self.lsp.a**2)

        elif self.number_of_speakers == "three":    
            Za1 =  (self.lsp.a**2 * z11 + 2* self.lsp.a * self.lsp.a * z12 + 2* self.lsp.a * self.lsp.a * z13 ) / (self.lsp.a**2 + self.lsp.a**2+ self.lsp.a**2)
    
        elif self.number_of_speakers == "four" :
            Za1 =  (self.lsp.a**2 * z11 + 2* self.lsp.a * self.lsp.a * z12 + 2* self.lsp.a * self.lsp.a * z13+ 2* self.lsp.a * self.lsp.a * z14) / (self.lsp.a**2 + self.lsp.a**2+ self.lsp.a**2+ self.lsp.a**2)
            
                
        return Za1
        
        
        
                
    def calculate_box_impedance_for_circular_piston_Zxy(self, f , a, ap, x1, x2, y1, y2, lx, ly, lz, Sd, Sp):
        "Calculate the box impedance based on equation 7.131 for circular loudspeaker."

        # wave number k equation 7.11
        k = (2 * np.pi * f / SOUND_CELERITY) * (
            (1 + self.a3 * (self.R_f / f) ** self.b3)
            - 1j * self.a4 * (self.R_f / f) ** self.b4
        )

        Zs = R_0 * SOUND_CELERITY + self.P_0 / (1j * 2 * np.pi * f * self.d)

        sum_mn = 0
        for m in range(self.truncation_limit + 1):
            for n in range(self.truncation_limit + 1):
                kmn = np.sqrt(
                    k**2 - (m * np.pi / lx) ** 2 - (n * np.pi / ly) ** 2
                )
                delta_m0 = 1 if m == 0 else 0
                delta_n0 = 1 if n == 0 else 0
                term1 = (
                    (kmn * Zs) / (k * R_0 * SOUND_CELERITY) + 1j * np.tan(kmn * lz)
                ) / (
                    1
                    + 1j
                    * ((kmn * Zs) / (k * R_0 * SOUND_CELERITY))
                    * np.tan(kmn * lz)
                )
                term2 = (
                    (2 - delta_m0)
                    * (2 - delta_n0)
                    / (
                        kmn * (n**2 * lx**2 + m**2 * ly**2)
                        + delta_m0 * delta_n0
                    )
                )
                term3 = (
                    np.cos((m * np.pi * x1) / lx)
                    * np.cos((n * np.pi * y1) / ly)
                    * j1(
                        (
                            np.pi
                            * a
                            * np.sqrt(n**2 * lx**2 + m**2 * ly**2)
                        )
                        / (lx * ly)
                    )
                )
                term4 = (
                    np.cos((m * np.pi * x2) / lx)
                    * np.cos((n * np.pi * y2) / ly)
                    * j1(
                        (
                            np.pi
                            * ap
                            * np.sqrt(n**2 * lx**2 + m**2 * ly**2)
                        )
                        / (lx * ly)
                    )
                )
                sum_mn += term1 * term2 * term3 * term4

        Zxy = (
            R_0
            * SOUND_CELERITY
            * (
                (Sd * Sp)
                / (lx * ly)
                * (
                    ((Zs / (R_0 * SOUND_CELERITY)) + 1j * np.tan(k * lz))
                    / (1 + 1j * ((Zs / (R_0 * SOUND_CELERITY)) * np.tan(k * lz)))
                )
                + 4 * k * a * ap * lx * ly * sum_mn
            )
        ) / (Sd * Sp)

        return Zxy

    

    def calculate_port_impedance_Za2(self, f, r_d, lx, ly, Sp ):
        "Calculate the rectangular port impedance based on equation 13.336 and 13.337."
        q = lx / ly
        
        k = (2 * np.pi * f / SOUND_CELERITY) * (
            (1 + self.a3 * (self.R_f / f) ** self.b3)
            - 1j * self.a4 * (self.R_f / f) ** self.b4
        )

        Rs_a2 = (R_0 * SOUND_CELERITY) / (np.sqrt(np.pi))

        sum_Rs = 0
        sum_Xs = 0

        for m in range(self.truncation_limit + 1):
            for n in range(self.truncation_limit + 1):
                term1 = (-1) ** (m + n)
                term2 = (
                    (2 * m + 1)
                    * (2 * n + 1)
                    * math.factorial(m + 1)
                    * math.factorial(n + 1)
                    * gamma(m + n + 3 / 2)
                )
                term3 = (k * lx / 2) ** (2 * m + 1)
                term4 = (k * ly / 2) ** (2 * n + 1)
                sum_Rs += (term1 / term2) * term3 * term4

        Rs_a2 *= sum_Rs

        for m in range(self.truncation_limit + 1):
            term1 = (-1) ** m * self.fm(q, m, n)
            term2 = (2 * m + 1) * math.factorial(m) * math.factorial(m + 1)
            term3 = (k * lx / 2) ** (2 * m + 1)
            sum_Xs += (term1 / term2) * term3

        Xs_a2 = ((2 * r_d * SOUND_CELERITY) / (np.sqrt(np.pi))) * (
            (1 - np.sinc(k * lx)) / (q * k * lx)
            + (1 - np.sinc(q * k * lx)) / (k * lx)
            + sum_Xs
        )

        Z_a2 = (Rs_a2 + 1j * Xs_a2) / Sp

        return Z_a2

    def fm(self, q, m, n):
        "Helper function for the calculation of the rectangular port impedance based on equation 13.337."
        result1 = scipy.special.hyp2f1(1, m + 0.5, m + 1.5, 1 / (1 + q**2))
        result2 = scipy.special.hyp2f1(1, m + 0.5, m + 1.5, 1 / (1 + q ** (-2)))

        sum_fm = 0
        for n in range(m + 1):
            sum_fm = self.gmn(m, n, q)

        return (result1 + result2) / ((2 * m + 1) * (1 + q ** (-2)) ** (m + 0.5)) + (
            1 / (2 * m + 3)
        ) * sum_fm

    def gmn(self, m, n, q):
        "Helper function for the calculation of the rectangular port impedance based on equation 13.337."
        first_sum = 0

        for p in range(n, m + 1):
            first_sum += (
                ((-1) ** (p - n) * (q) ** (2 * n - 1))
                / ((2 * p - 1) * (1 + q**2) ** (p - 1 / 2))
                * scipy.special.binom((m - n), p - n)
            )

        first_term = scipy.special.binom((2 * m + 3), (2 * n)) * first_sum

        second_sum = 0

        for p in range(m - n, m + 1):
            second_sum += (
                scipy.special.binom((p - m + n), n)
                * (-1) ** (p - m + n)
                * (q) ** (2 * n + 2)
            ) / ((2 * p - 1) * (1 + q ** (-2)) ** (p - 1 / 2))

        second_term = scipy.special.binom((2 * m + 3), (2 * n + 3)) * second_sum

        return first_term + second_term
        
        
    def calculate_impedance_response(self, frequencies):
        # Preallocate memory
        response = np.zeros_like(frequencies)
        Ze = np.zeros_like(frequencies)

        for i in range(len(frequencies)):
            # calculate the electrical impedance
            Z_e = self.lsp.Re + 1j * 2 * np.pi * frequencies[i] * self.lsp.Le

            # calculate the mechanical impedance
            Mmd = self.lsp.Mms - 16 * R_0 * self.lsp.a**3 / 3
            Z_md = (
                1j * 2 * np.pi * frequencies[i] * Mmd
                + self.lsp.Rms
                + 1 / (1j * 2 * np.pi * frequencies[i] * self.lsp.Cms)
            )

            # wave number k equation 7.11
            k = (2 * np.pi * frequencies[i] / SOUND_CELERITY) * (
                (1 + self.a3 * (self.R_f / frequencies[i]) ** self.b3)
                - 1j * self.a4 * (self.R_f / frequencies[i]) ** self.b4
            )

            # diaphrams radiation impedance based on equation 13.116 and 13.118
            R_s = R_0 * SOUND_CELERITY * k**2 * self.lsp.a**2
            X_s = (R_0 * SOUND_CELERITY * 8 * k * self.lsp.a)/(3 * np.pi)
            Z_a1 =  (R_s + 1j * X_s)
 
            if self.number_of_speakers == "one" and self.port_shape == "rectangular":
                # calculate the radiation impedance of the diaphragm based on equation 13.339 - 13.345 with Mellow's method
                #Z_a1 = self.calculate_Za1(frequencies[i])
                
                # calculate the leakage resistance
                CAA = (self.Va1 * 10 ** (-3)) / (
                    1.4 * self.P_0
                )  # compliance of the air in the box
                CAM = (
                    self.Vm1 * 10 ** (-3)
                ) / self.P_0  # compliance of the air in the lining material
                Cab = CAA + 1.4 * CAM  # apparent compliance of the air in the box
                Ral = 7 / (2 * np.pi * self.fb * Cab)
                
                Z11 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i], self.lsp.a, self.lsp.a, self.x1, self.x1, self.y1, self.y1 , self.lx1, self.ly1, self.lz1 , self.lsp.Sd, self.lsp.Sd
                )
                Z12 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i],self.lsp.a, self.a_p1, self.x1, self.x1, self.y1, self.yp1 , self.lx1, self.ly1, self.lz1 , self.lsp.Sd, self.Sp1
                )
                Z22 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i], self.a_p1, self.a_p1 ,self.x1, self.x1, self.yp1, self.yp1 , self.lx1, self.ly1, self.lz1 , self.Sp1, self.Sp1
                )
                Z21 = Z12

                Kn = LM / self.a2
                Bu = (2 * 0.9 ** (-1) - 1) * Kn
                
                kv = np.sqrt((-1j * np.pi * 2 * frequencies[i] * R_0) / self.m)
                ξ = 0.998 + 0.001j
                kp = (2 * np.pi * frequencies[i] * ξ) / SOUND_CELERITY
                Zp = (R_0 * SOUND_CELERITY * ξ) / (self.Sp1)
                
                # dynamic density based on equation 4.233
                r_d = (-8 * R_0) / ((1 + 4 * Bu) * kv**2 * self.a2**2)
                Z_a2 = self.calculate_port_impedance_Za2(frequencies[i], r_d, self.lx1, self.ly1, self.Sp1)
                
                
                # 2-port network parameters
                b11 = Z11 / Z21
                b12 = (Z11 * Z22 - Z12 * Z21) / Z21
                b21 = 1 / Z21
                b22 = Z22 / Z21
                
                Z_ab = self.calculate_box_impedance_Zab(frequencies[i])

                C = np.array([[1, 1 * Z_e], [0, 1]])
                E = np.array([[0, 1 * self.lsp.Bl], [1 / self.lsp.Bl, 0]])
                D = np.array([[1,  1* Z_md], [0, 1]])
                M = np.array([[ self.lsp.Sd, 0], [0, 1 / (self.lsp.Sd)]])
                F = np.array([[1, 1 * Z_a1], [0, 1]])
                L = np.array([[1, 0], [1 / Ral, 1]])
                B = np.array([[1, 0], [1 / Z_ab, 1]])
                #B = np.array([[b11, b12], [b21, b22]])
                P = np.array(
                    [
                        [np.cos(kp * self.t1), 1j * Zp * np.sin(kp * self.t1)],
                        [1j * (1 / Zp) * np.sin(kp * self.t1), np.cos(kp * self.t1)],
                    ]
                )
                R = np.array([[1, 0], [1 / Z_a2, 1]])

                A = C @ E @ D @ M @ F @ L @ B @ P @ R

                a11 = A[0, 0]
                # a12 = A[0, 1]
                a21 = A[1, 0]
                # a22 = A[1, 1]

                p9 = self.lsp.e_g / a11
                # Up = p9 / Z_a2

                N = np.dot(np.dot(B, P), R)

                # n11 = N[0, 0]
                # n12 = N[0, 1]
                n21 = N[1, 0]
                # n22 = N[1, 1]

                # U6 = n21 * p9
                UB = (n21 - 1 / Z_a2) * p9
                # Up = self.lsp.e_g / (a11 * Z_a2)

                # ig = a21 * p9

                U_ref = (1 * self.lsp.e_g * self.lsp.Bl * self.lsp.Sd) / (
                    2 * np.pi * frequencies[i] * self.lsp.Mms * self.lsp.Re
                )

                # response[i] = 20 * np.log10(float(abs(U6))
                #                             / float(abs(U_ref)))

                response[i] = 20 * np.log10(float(abs(UB)) / float(abs(U_ref)))
                Ze[i] = abs(((a11) / (a21)))

                self.response = response
                self.Ze = Ze

                
            elif self.number_of_speakers == "one" and self.port_shape == "circular":
                # calculate the radiation impedance of the diaphragm based on equation 13.339 - 13.345 with Mellow's method
                #Z_a1 = self.calculate_Za1(frequencies[i])
                 
                # calculate the leakage resistance
                CAA = (self.Va1 * 10 ** (-3)) / (
                    1.4 * self.P_0
                )  # compliance of the air in the box
                CAM = (
                    self.Vm1 * 10 ** (-3)
                ) / self.P_0  # compliance of the air in the lining material
                Cab = CAA + 1.4 * CAM  # apparent compliance of the air in the box
                Ral = 7 / (2 * np.pi * self.fb * Cab)
                
                Z11 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i],self.lsp.a, self.lsp.a ,self.x1, self.x1, self.y1, self.y1 , self.lx1, self.ly1, self.lz1 , self.lsp.Sd, self.lsp.Sd
                )
                Z12 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i],self.lsp.a, self.a_p1, self.x1, self.x1, self.y1, self.yp1 , self.lx1, self.ly1, self.lz1, self.lsp.Sd, self.Sp1
                )
                Z22 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i],self.a_p1 , self.a_p1 ,self.x1, self.x1, self.yp1, self.yp1 , self.lx1, self.ly1, self.lz1 , self.Sp1, self.Sp1
                )
                Z21 = Z12
                
                # 2-port network parameters
                b11 = Z11 / Z21
                b12 = (Z11 * Z22 - Z12 * Z21) / Z21
                b21 = 1 / Z21
                b22 = Z22 / Z21
                
                kv = np.sqrt((-1j * np.pi * 2 * frequencies[i] * R_0) / self.m)
                ξ = 0.998 + 0.001j
                kp = (2 * np.pi * frequencies[i] * ξ) / SOUND_CELERITY
                Zp = (R_0 * SOUND_CELERITY * ξ) / (self.Sp1)                
                
                # calculate the impedance for circular port
                H1 = mpmath.struveh(1, 2*k * self.a_p1) 
                R_sp = R_0 * SOUND_CELERITY * (1-jv(1, 2*k*self.a_p1)/(k*self.a_p1))
                X_sp = R_0 * SOUND_CELERITY * (H1/(k*self.a_p1))
                Z_a2 = (R_sp + 1j * X_sp)/ self.Sp4

                Z_ab = self.calculate_box_impedance_Zab(frequencies[i])

                C = np.array([[1, 1 * Z_e], [0, 1]])
                E = np.array([[0, 1 * self.lsp.Bl], [1 / self.lsp.Bl, 0]])
                D = np.array([[1, 1 * Z_md], [0, 1]])
                M = np.array([[ self.lsp.Sd, 0], [0, 1 / (self.lsp.Sd)]])
                F = np.array([[1, 1 * Z_a1], [0, 1]])
                L = np.array([[1, 0], [1 / Ral, 1]])
                B = np.array([[1, 0], [1 / Z_ab, 1]])
                #B = np.array([[b11, b12], [b21, b22]])
                P = np.array(
                    [
                        [np.cos(kp * self.t1), 1j * Zp * np.sin(kp * self.t1)],
                        [1j * (1 / Zp) * np.sin(kp * self.t1), np.cos(kp * self.t1)],
                    ]
                )
                R = np.array([[1, 0], [1 / Z_a2, 1]])
                
                A = C @ E @ D @ M @ F @ L @ B @ P @ R

                a11 = A[0, 0]
                # a12 = A[0, 1]
                a21 = A[1, 0]
                # a22 = A[1, 1]

                p9 = self.lsp.e_g / a11
                # Up = p9 / Z_a2

                N = np.dot(np.dot(B, P), R)

                # n11 = N[0, 0]
                # n12 = N[0, 1]
                n21 = N[1, 0]
                # n22 = N[1, 1]

                # U6 = n21 * p9
                UB = (n21 - 1 / Z_a2) * p9
                # Up = self.lsp.e_g / (a11 * Z_a2)

                # ig = a21 * p9

                U_ref = (1 * self.lsp.e_g * self.lsp.Bl * self.lsp.Sd) / (
                    2 * np.pi * frequencies[i] * self.lsp.Mms * self.lsp.Re
                )

                # response[i] = 20 * np.log10(float(abs(U6))
                #                             / float(abs(U_ref)))

                response[i] = 20 * np.log10(float(abs(UB)) / float(abs(U_ref)))
                Ze[i] = abs(((a11) / (a21)))

                self.response = response
                self.Ze = Ze                



            elif self.number_of_speakers == "two" and self.port_shape == "rectangular":
                # calculate the radiation impedance of the diaphragm based on equation 13.339 - 13.345 with Mellow's method
                #Z_a1 = self.calculate_Za1(frequencies[i])
                
                # calculate the leakage resistance
                CAA = (self.Va2 * 10 ** (-3)) / (
                    1.4 * self.P_0
                )  # compliance of the air in the box
                CAM = (
                    self.Vm2 * 10 ** (-3)
                ) / self.P_0  # compliance of the air in the lining material
                Cab = CAA + 1.4 * CAM  # apparent compliance of the air in the box
                Ral = 7 / (2 * np.pi * self.fb * Cab)
                
                Z11 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i],self.lsp.a, self.lsp.a, self.x2, self.x2, self.y2, self.y2 , self.lx2, self.ly2, self.lz2 , self.lsp.Sd, self.lsp.Sd
                )
                Z12 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i] , self.lsp.a, self.a_p2, self.x2, self.x2, self.y2, self.yp2 , self.lx2, self.ly2, self.lz2 , self.lsp.Sd, self.Sp2
                )
                Z22 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i] ,self.a_p2, self.a_p2, self.x2, self.x2, self.yp2, self.yp2 , self.lx2, self.ly2, self.lz2, self.Sp2, self.Sp2
                )
                Z21 = Z12

                Kn = LM / self.a2
                Bu = (2 * 0.9 ** (-1) - 1) * Kn

                kv = np.sqrt((-1j * np.pi * 2 * frequencies[i] * R_0) / self.m)
                ξ = 0.998 + 0.001j
                kp = (2 * np.pi * frequencies[i] * ξ) / SOUND_CELERITY
                Zp = (R_0 * SOUND_CELERITY * ξ) / (self.Sp2)                

                # dynamic density based on equation 4.233
                r_d = (-8 * R_0) / ((1 + 4 * Bu) * kv**2 * self.a2**2)
                Z_a2 = self.calculate_port_impedance_Za2(frequencies[i], r_d, self.lx2, self.ly2, self.Sp2)
                
                
                # 2-port network parameters
                b11 = Z11 / Z21
                b12 = (Z11 * Z22 - Z12 * Z21) / Z21
                b21 = 1 / Z21
                b22 = Z22 / Z21
                
                Z_ab = self.calculate_box_impedance_Zab(frequencies[i])

                C = np.array([[1, 1/2 * Z_e], [0, 1]])
                E = np.array([[0, 1 * self.lsp.Bl], [1 / self.lsp.Bl, 0]])
                D = np.array([[1,  2* Z_md], [0, 1]])
                M = np.array([[ 2* self.lsp.Sd, 0], [0, 1 / (2* self.lsp.Sd)]])
                F = np.array([[1, 2 * Z_a1], [0, 1]])
                L = np.array([[1, 0], [1 / Ral, 1]])
                B = np.array([[1, 0], [1 / Z_ab, 1]])
                #B = np.array([[b11, b12], [b21, b22]])
                P = np.array(
                    [
                        [np.cos(kp * self.t2), 1j * Zp * np.sin(kp * self.t2)],
                        [1j * (1 / Zp) * np.sin(kp * self.t2), np.cos(kp * self.t2)],
                    ]
                )
                R = np.array([[1, 0], [1 / Z_a2, 1]])
                
                A = C @ E @ D @ M @ F @ L @ B @ P @ R

                a11 = A[0, 0]
                # a12 = A[0, 1]
                a21 = A[1, 0]
                # a22 = A[1, 1]

                p9 = self.lsp.e_g / a11
                # Up = p9 / Z_a2

                N = np.dot(np.dot(B, P), R)

                # n11 = N[0, 0]
                # n12 = N[0, 1]
                n21 = N[1, 0]
                # n22 = N[1, 1]

                # U6 = n21 * p9
                UB = (n21 - 1 / Z_a2) * p9
                # Up = self.lsp.e_g / (a11 * Z_a2)

                # ig = a21 * p9

                U_ref = (2 * self.lsp.e_g * self.lsp.Bl * self.lsp.Sd) / (
                    2 * np.pi * frequencies[i] * self.lsp.Mms * self.lsp.Re
                )

                # response[i] = 20 * np.log10(float(abs(U6))
                #                             / float(abs(U_ref)))

                response[i] = 20 * np.log10(float(abs(UB)) / float(abs(U_ref)))
                Ze[i] = abs(((a11) / (a21)))

                self.response = response
                self.Ze = Ze                

            
            elif self.number_of_speakers == "two" and self.port_shape == "circular":
                # calculate the radiation impedance of the diaphragm based on equation 13.339 - 13.345 with Mellow's method
                #Z_a1 = self.calculate_Za1(frequencies[i])
                
                # calculate the leakage resistance
                CAA = (self.Va2 * 10 ** (-3)) / (
                    1.4 * self.P_0
                )  # compliance of the air in the box
                CAM = (
                    self.Vm2 * 10 ** (-3)
                ) / self.P_0  # compliance of the air in the lining material
                Cab = CAA + 1.4 * CAM  # apparent compliance of the air in the box
                Ral = 7 / (2 * np.pi * self.fb * Cab)
                
                Z11 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i],self.lsp.a, self.lsp.a, self.x2, self.x2, self.y2, self.y2 , self.lx2, self.ly2, self.lz2 , self.lsp.Sd, self.lsp.Sd
                )
                Z12 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i],self.lsp.a, self.a_p2 ,self.x2, self.x2, self.y2, self.yp2 , self.lx2, self.ly2, self.lz2 , self.lsp.Sd, self.Sp2
                )
                Z22 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i], self.a_p2, self.a_p2,self.x2, self.x2, self.yp2, self.yp2 , self.lx2, self.ly2, self.lz2 , self.Sp2, self.Sp2
                )
                Z21 = Z12
                
                # 2-port network parameters
                b11 = Z11 / Z21
                b12 = (Z11 * Z22 - Z12 * Z21) / Z21
                b21 = 1 / Z21
                b22 = Z22 / Z21
                
                Z_ab = self.calculate_box_impedance_Zab(frequencies[i])
                
                kv = np.sqrt((-1j * np.pi * 2 * frequencies[i] * R_0) / self.m)
                ξ = 0.998 + 0.001j
                kp = (2 * np.pi * frequencies[i] * ξ) / SOUND_CELERITY
                Zp = (R_0 * SOUND_CELERITY * ξ) / (self.Sp2)                
                
                # calculate the impedance for circular port
                H1 = mpmath.struveh(1, 2*k * self.a_p2) 
                R_sp = R_0 * SOUND_CELERITY * (1-jv(1, 2*k*self.a_p2)/(k*self.a_p2))
                X_sp = R_0 * SOUND_CELERITY * (H1/(k*self.a_p2))
                Z_a2 = (R_sp + 1j * X_sp)/ self.Sp4

                C = np.array([[1, 1/2 * Z_e], [0, 1]])
                E = np.array([[0, 1 * self.lsp.Bl], [1 / self.lsp.Bl, 0]])
                D = np.array([[1, 2 * Z_md], [0, 1]])
                M = np.array([[ 2* self.lsp.Sd, 0], [0, 1 / (2 * self.lsp.Sd)]])
                F = np.array([[1, 2 * Z_a1], [0, 1]])
                L = np.array([[1, 0], [1 / Ral, 1]])
                B = np.array([[1, 0], [1 / Z_ab, 1]])
                #B = np.array([[b11, b12], [b21, b22]])
                P = np.array(
                    [
                        [np.cos(kp * self.t2), 1j * Zp * np.sin(kp * self.t2)],
                        [1j * (1 / Zp) * np.sin(kp * self.t2), np.cos(kp * self.t2)],
                    ]
                )
                R = np.array([[1, 0], [1 / Z_a2, 1]])
                
                A = C @ E @ D @ M @ F @ L @ B @ P @ R

                a11 = A[0, 0]
                # a12 = A[0, 1]
                a21 = A[1, 0]
                # a22 = A[1, 1]

                p9 = self.lsp.e_g / a11
                # Up = p9 / Z_a2

                N = np.dot(np.dot(B, P), R)

                # n11 = N[0, 0]
                # n12 = N[0, 1]
                n21 = N[1, 0]
                # n22 = N[1, 1]

                # U6 = n21 * p9
                UB = (n21 - 1 / Z_a2) * p9
                # Up = self.lsp.e_g / (a11 * Z_a2)

                # ig = a21 * p9

                U_ref = (2 * self.lsp.e_g * self.lsp.Bl * self.lsp.Sd) / (
                    2 * np.pi * frequencies[i] * self.lsp.Mms * self.lsp.Re
                )

                # response[i] = 20 * np.log10(float(abs(U6))
                #                             / float(abs(U_ref)))

                response[i] = 20 * np.log10(float(abs(UB)) / float(abs(U_ref)))
                Ze[i] = abs(((a11) / (a21)))

                self.response = response
                self.Ze = Ze                



            elif self.number_of_speakers == "three" and self.port_shape == "rectangular":
                # calculate the radiation impedance of the diaphragm based on equation 13.339 - 13.345 with Mellow's method
                #Z_a1 = self.calculate_Za1(frequencies[i])
                # calculate the leakage resistance
                CAA = (self.Va3 * 10 ** (-3)) / (
                    1.4 * self.P_0
                )  # compliance of the air in the box
                CAM = (
                    self.Vm3 * 10 ** (-3)
                ) / self.P_0  # compliance of the air in the lining material
                Cab = CAA + 1.4 * CAM  # apparent compliance of the air in the box
                Ral = 7 / (2 * np.pi * self.fb * Cab)                
                
                Z11 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i], self.lsp.a, self.lsp.a, self.x3, self.x3, self.y3, self.y3 , self.lx3, self.ly3, self.lz3 , self.lsp.Sd, self.lsp.Sd
                )
                Z12 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i] ,self.lsp.a, self.a_p3 ,self.x3, self.x3, self.y3, self.yp3 , self.lx3, self.ly3, self.lz3 , self.lsp.Sd, self.Sp3
                )
                Z22 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i] , self.a_p3, self.a_p3, self.x3, self.x3, self.yp3, self.yp3 , self.lx3, self.ly3, self.lz3 , self.Sp3, self.Sp3
                )
                Z21 = Z12

                Kn = LM / self.a2
                Bu = (2 * 0.9 ** (-1) - 1) * Kn
                
                kv = np.sqrt((-1j * np.pi * 2 * frequencies[i] * R_0) / self.m)
                ξ = 0.998 + 0.001j
                kp = (2 * np.pi * frequencies[i] * ξ) / SOUND_CELERITY
                Zp = (R_0 * SOUND_CELERITY * ξ) / (self.Sp3)                

                # dynamic density based on equation 4.233
                r_d = (-8 * R_0) / ((1 + 4 * Bu) * kv**2 * self.a2**2)
                Z_a2 = self.calculate_port_impedance_Za2(frequencies[i], r_d, self.lx3, self.ly3, self.Sp3)
                
                
                # 2-port network parameters
                b11 = Z11 / Z21
                b12 = (Z11 * Z22 - Z12 * Z21) / Z21
                b21 = 1 / Z21
                b22 = Z22 / Z21
                
                Z_ab = self.calculate_box_impedance_Zab(frequencies[i])

                C = np.array([[1, 1/3 * Z_e], [0, 1]])
                E = np.array([[0, 1 * self.lsp.Bl], [1 / self.lsp.Bl, 0]])
                D = np.array([[1,  3* Z_md], [0, 1]])
                M = np.array([[3* self.lsp.Sd, 0], [0, 1 / (3*self.lsp.Sd)]])
                F = np.array([[1, 3 * Z_a1], [0, 1]])
                L = np.array([[1, 0], [1 / Ral, 1]])
                B = np.array([[1, 0], [1 / Z_ab, 1]])
                #B = np.array([[b11, b12], [b21, b22]])
                P = np.array(
                    [
                        [np.cos(kp * self.t3), 1j * Zp * np.sin(kp * self.t3)],
                        [1j * (1 / Zp) * np.sin(kp * self.t3), np.cos(kp * self.t3)],
                    ]
                )
                R = np.array([[1, 0], [1 / Z_a2, 1]])
                
                A = C @ E @ D @ M @ F @ L @ B @ P @ R

                a11 = A[0, 0]
                # a12 = A[0, 1]
                a21 = A[1, 0]
                # a22 = A[1, 1]

                p9 = self.lsp.e_g / a11
                # Up = p9 / Z_a2

                N = np.dot(np.dot(B, P), R)

                # n11 = N[0, 0]
                # n12 = N[0, 1]
                n21 = N[1, 0]
                # n22 = N[1, 1]

                # U6 = n21 * p9
                UB = (n21 - 1 / Z_a2) * p9
                # Up = self.lsp.e_g / (a11 * Z_a2)

                # ig = a21 * p9

                U_ref = (3 * self.lsp.e_g * self.lsp.Bl * self.lsp.Sd) / (
                    2 * np.pi * frequencies[i] * self.lsp.Mms * self.lsp.Re
                )

                # response[i] = 20 * np.log10(float(abs(U6))
                #                             / float(abs(U_ref)))

                response[i] = 20 * np.log10(float(abs(UB)) / float(abs(U_ref)))
                Ze[i] = abs(((a11) / (a21)))

                self.response = response
                self.Ze = Ze                

            
            elif self.number_of_speakers == "three" and self.port_shape == "circular":
                # calculate the radiation impedance of the diaphragm based on equation 13.339 - 13.345 with Mellow's method
                #Z_a1 = self.calculate_Za1(frequencies[i])
                # calculate the leakage resistance
                CAA = (self.Va3 * 10 ** (-3)) / (
                    1.4 * self.P_0
                )  # compliance of the air in the box
                CAM = (
                    self.Vm3 * 10 ** (-3)
                ) / self.P_0  # compliance of the air in the lining material
                Cab = CAA + 1.4 * CAM  # apparent compliance of the air in the box
                Ral = 7 / (2 * np.pi * self.fb * Cab)  
                                
                Z11 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i], self.lsp.a, self.lsp.a, self.x3, self.x3, self.y3, self.y3 , self.lx3, self.ly3, self.lz3 , self.lsp.Sd, self.lsp.Sd
                )
                Z12 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i] ,self.lsp.a, self.a_p3 ,self.x3, self.x3, self.y3, self.yp3 , self.lx3, self.ly3, self.lz3 , self.lsp.Sd, self.Sp3
                )
                Z22 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i] , self.a_p3, self.a_p3, self.x3, self.x3, self.yp3, self.yp3 , self.lx3, self.ly3, self.lz3 , self.Sp3, self.Sp3
                )
                Z21 = Z12

                Kn = LM / self.a2
                Bu = (2 * 0.9 ** (-1) - 1) * Kn
                
                kv = np.sqrt((-1j * np.pi * 2 * frequencies[i] * R_0) / self.m)
                ξ = 0.998 + 0.001j
                kp = (2 * np.pi * frequencies[i] * ξ) / SOUND_CELERITY
                Zp = (R_0 * SOUND_CELERITY * ξ) / (self.Sp3)                

                H1 = mpmath.struveh(1, 2*k * self.a_p3) 
                R_sp = R_0 * SOUND_CELERITY * (1-jv(1, 2*k*self.a_p3)/(k*self.a_p3))
                X_sp = R_0 * SOUND_CELERITY * (H1/(k*self.a_p3))
                Z_a2 = (R_sp + 1j * X_sp)/ self.Sp3
                
                # 2-port network parameters
                b11 = Z11 / Z21
                b12 = (Z11 * Z22 - Z12 * Z21) / Z21
                b21 = 1 / Z21
                b22 = Z22 / Z21
                
                Z_ab= self.calculate_box_impedance_Zab(frequencies[i])

                C = np.array([[1, 1/3 * Z_e], [0, 1]])
                E = np.array([[0, 1 * self.lsp.Bl], [1 / self.lsp.Bl, 0]])
                D = np.array([[1,  3* Z_md], [0, 1]])
                M = np.array([[3* self.lsp.Sd, 0], [0, 1 / (3*self.lsp.Sd)]])
                F = np.array([[1, 3 * Z_a1], [0, 1]])
                L = np.array([[1, 0], [1 / Ral, 1]])
                B = np.array([[1, 0], [1 / Z_ab, 1]])
                #B = np.array([[b11, b12], [b21, b22]])
                P = np.array(
                    [
                        [np.cos(kp * self.t3), 1j * Zp * np.sin(kp * self.t3)],
                        [1j * (1 / Zp) * np.sin(kp * self.t3), np.cos(kp * self.t3)],
                    ]
                )
                R = np.array([[1, 0], [1 / Z_a2, 1]])
                
                A = C @ E @ D @ M @ F @ L @ B @ P @ R

                a11 = A[0, 0]
                # a12 = A[0, 1]
                a21 = A[1, 0]
                # a22 = A[1, 1]

                p9 = self.lsp.e_g / a11
                # Up = p9 / Z_a2

                N = np.dot(np.dot(B, P), R)

                # n11 = N[0, 0]
                # n12 = N[0, 1]
                n21 = N[1, 0]
                # n22 = N[1, 1]

                # U6 = n21 * p9
                UB = (n21 - 1 / Z_a2) * p9
                # Up = self.lsp.e_g / (a11 * Z_a2)

                # ig = a21 * p9

                U_ref = (3 * self.lsp.e_g * self.lsp.Bl * self.lsp.Sd) / (
                    2 * np.pi * frequencies[i] * self.lsp.Mms * self.lsp.Re
                )

                # response[i] = 20 * np.log10(float(abs(U6))
                #                             / float(abs(U_ref)))

                response[i] = 20 * np.log10(float(abs(UB)) / float(abs(U_ref)))
                Ze[i] = abs(((a11) / (a21)))

                self.response = response
                self.Ze = Ze                


            elif self.number_of_speakers == "four" and self.port_shape == "rectangular":
                # calculate the radiation impedance of the diaphragm based on equation 13.339 - 13.345 with Mellow's method
                #Z_a1 = self.calculate_Za1(frequencies[i])
                # calculate the leakage resistance
                CAA = (self.Va4 * 10 ** (-3)) / (
                    1.4 * self.P_0
                )  # compliance of the air in the box
                CAM = (
                    self.Vm4 * 10 ** (-3)
                ) / self.P_0  # compliance of the air in the lining material
                Cab = CAA + 1.4 * CAM  # apparent compliance of the air in the box
                Ral = 7 / (2 * np.pi * self.fb * Cab)
                                  
                Z11 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i], self.lsp.a, self.lsp.a, self.x4, self.x4, self.y4, self.y4 , self.lx4, self.ly4, self.lz4 , self.lsp.Sd, self.lsp.Sd
                )
                Z12 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i] , self.lsp.a, self.a_p4, self.x4, self.x4, self.y4, self.yp4 , self.lx4, self.ly4, self.lz4 , self.lsp.Sd, self.Sp4
                )
                Z22 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i] , self.a_p4, self.a_p4, self.x4, self.x4, self.yp4, self.yp4 , self.lx4, self.ly4, self.lz4 , self.Sp4, self.Sp4
                )
                Z21 = Z12

                Kn = LM / self.a2
                Bu = (2 * 0.9 ** (-1) - 1) * Kn
                
                kv = np.sqrt((-1j * np.pi * 2 * frequencies[i] * R_0) / self.m)
                ξ = 0.998 + 0.001j
                kp = (2 * np.pi * frequencies[i] * ξ) / SOUND_CELERITY
                Zp = (R_0 * SOUND_CELERITY * ξ) / (self.Sp4)                

                # dynamic density based on equation 4.233
                r_d = (-8 * R_0) / ((1 + 4 * Bu) * kv**2 * self.a2**2)
                Z_a2 = self.calculate_port_impedance_Za2(frequencies[i], r_d, self.lx4, self.ly4, self.Sp4)
                
                
                # 2-port network parameters
                b11 = Z11 / Z21
                b12 = (Z11 * Z22 - Z12 * Z21) / Z21
                b21 = 1 / Z21
                b22 = Z22 / Z21
                
                Z_ab = self.calculate_box_impedance_Zab(frequencies[i])

                C = np.array([[1, 1/4 * Z_e], [0, 1]])
                E = np.array([[0, 1 * self.lsp.Bl], [1 / self.lsp.Bl, 0]])
                D = np.array([[1,  4* Z_md], [0, 1]])
                M = np.array([[4* self.lsp.Sd, 0], [0, 1 / (4*self.lsp.Sd)]])
                F = np.array([[1, 4 * Z_a1], [0, 1]])
                L = np.array([[1, 0], [1 / Ral, 1]])
                B = np.array([[1, 0], [1 / Z_ab, 1]])
                #B = np.array([[b11, b12], [b21, b22]])
                P = np.array(
                    [
                        [np.cos(kp * self.t4), 1j * Zp * np.sin(kp * self.t4)],
                        [1j * (1 / Zp) * np.sin(kp * self.t4), np.cos(kp * self.t4)],
                    ]
                )
                R = np.array([[1, 0], [1 / Z_a2, 1]])
                
                A = C @ E @ D @ M @ F @ L @ B @ P @ R

                a11 = A[0, 0]
                # a12 = A[0, 1]
                a21 = A[1, 0]
                # a22 = A[1, 1]

                p9 = self.lsp.e_g / a11
                # Up = p9 / Z_a2

                N = np.dot(np.dot(B, P), R)

                # n11 = N[0, 0]
                # n12 = N[0, 1]
                n21 = N[1, 0]
                # n22 = N[1, 1]

                # U6 = n21 * p9
                UB = (n21 - 1 / Z_a2) * p9
                # Up = self.lsp.e_g / (a11 * Z_a2)

                # ig = a21 * p9

                U_ref = (4 * self.lsp.e_g * self.lsp.Bl * self.lsp.Sd) / (
                    2 * np.pi * frequencies[i] * self.lsp.Mms * self.lsp.Re
                )

                # response[i] = 20 * np.log10(float(abs(U6))
                #                             / float(abs(U_ref)))

                response[i] = 20 * np.log10(float(abs(UB)) / float(abs(U_ref)))
                Ze[i] = abs(((a11) / (a21)))

                self.response = response
                self.Ze = Ze                
     
        
            elif self.number_of_speakers == "four" and self.port_shape == "circular":
                # calculate the radiation impedance of the diaphragm based on equation 13.339 - 13.345 with Mellow's method
                #Z_a1 = self.calculate_Za1(frequencies[i])
                # calculate the leakage resistance
                CAA = (self.Va4 * 10 ** (-3)) / (
                    1.4 * self.P_0
                )  # compliance of the air in the box
                CAM = (
                    self.Vm4 * 10 ** (-3)
                ) / self.P_0  # compliance of the air in the lining material
                Cab = CAA + 1.4 * CAM  # apparent compliance of the air in the box
                Ral = 7 / (2 * np.pi * self.fb * Cab)
                                
                Z11 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i], self.lsp.a, self.lsp.a, self.x4, self.x4, self.y4, self.y4 , self.lx4, self.ly4, self.lz4 , self.lsp.Sd, self.lsp.Sd
                )
                Z12 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i] , self.lsp.a, self.a_p4, self.x4, self.x4, self.y4, self.yp4 , self.lx4, self.ly4, self.lz4 , self.lsp.Sd, self.Sp4
                )
                Z22 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i] , self.a_p4, self.a_p4, self.x4, self.x4, self.yp4, self.yp4 , self.lx4, self.ly4, self.lz4 , self.Sp4, self.Sp4
                )
                Z21 = Z12

                Kn = LM / self.a2
                Bu = (2 * 0.9 ** (-1) - 1) * Kn

                H1 = mpmath.struveh(1, 2*k * self.a_p4) 
                R_sp = R_0 * SOUND_CELERITY * (1-jv(1, 2*k*self.a_p4)/(k*self.a_p4))
                X_sp = R_0 * SOUND_CELERITY * (H1/(k*self.a_p4))
                Z_a2 = (R_sp + 1j * X_sp)/ self.Sp4
                
                kv = np.sqrt((-1j * np.pi * 2 * frequencies[i] * R_0) / self.m)
                ξ = 0.998 + 0.001j
                kp = (2 * np.pi * frequencies[i] * ξ) / SOUND_CELERITY
                Zp = (R_0 * SOUND_CELERITY * ξ) / (self.Sp4)                
                
                
                # 2-port network parameters
                b11 = Z11 / Z21
                b12 = (Z11 * Z22 - Z12 * Z21) / Z21
                b21 = 1 / Z21
                b22 = Z22 / Z21
                
                Z_ab = self.calculate_box_impedance_Zab(frequencies[i])

                C = np.array([[1, 1/4 * Z_e], [0, 1]])
                E = np.array([[0, 1 * self.lsp.Bl], [1 / self.lsp.Bl, 0]])
                D = np.array([[1,  4* Z_md], [0, 1]])
                M = np.array([[4* self.lsp.Sd, 0], [0, 1 / (4*self.lsp.Sd)]])
                F = np.array([[1, 4 * Z_a1], [0, 1]])
                L = np.array([[1, 0], [1 / Ral, 1]])
                B = np.array([[1, 0], [1 / Z_ab, 1]])
                #B = np.array([[b11, b12], [b21, b22]])
                P = np.array(
                    [
                        [np.cos(kp * self.t4), 1j * Zp * np.sin(kp * self.t4)],
                        [1j * (1 / Zp) * np.sin(kp * self.t4), np.cos(kp * self.t4)],
                    ]
                )
                R = np.array([[1, 0], [1 / Z_a2, 1]])

                A = C @ E @ D @ M @ F @ L @ B @ P @ R

                a11 = A[0, 0]
                # a12 = A[0, 1]
                a21 = A[1, 0]
                # a22 = A[1, 1]

                p9 = self.lsp.e_g / a11
                # Up = p9 / Z_a2

                N = np.dot(np.dot(B, P), R)

                # n11 = N[0, 0]
                # n12 = N[0, 1]
                n21 = N[1, 0]
                # n22 = N[1, 1]

                # U6 = n21 * p9
                UB = (n21 - 1 / Z_a2) * p9
                # Up = self.lsp.e_g / (a11 * Z_a2)

                # ig = a21 * p9

                U_ref = (4 * self.lsp.e_g * self.lsp.Bl * self.lsp.Sd) / (
                    2 * np.pi * frequencies[i] * self.lsp.Mms * self.lsp.Re
                )

                # response[i] = 20 * np.log10(float(abs(U6))
                #                             / float(abs(U_ref)))

                response[i] = 20 * np.log10(float(abs(UB)) / float(abs(U_ref)))
                Ze[i] = abs(((a11) / (a21)))

                self.response = response
                self.Ze = Ze
            
        return response, Ze
            
            
octave_steps = 24
frequencies_per_octave = octave_steps * np.log(2)
min_frequency = 10
max_frequency = 10000
num_points = int(octave_steps * np.log2(max_frequency / min_frequency)) + 1
frequencies = np.logspace(
    np.log2(min_frequency), np.log2(max_frequency), num=num_points, base=2
)

lsp_parameters = {
    "Re": 6.72,
    "Le": 0.00067,
    "e_g": 2.83,
    "Qes": 0.522,
    "Qms": 1.9,
    "fs": 37,
    "Sd": 0.012,
    "Vas": 24,
    "Qts": 0.41,
    "Cms": 0.00119,
    "Mms": 0.0156,
    "Bl": 6.59,
}

# Create an instance of the BassReflexEnclosure class
lsp_sys_1 = BassReflexEnclosure(lsp_parameters, "one", "circular")
lsp_sys_2 = BassReflexEnclosure(lsp_parameters, "two", "circular")
lsp_sys_3 = BassReflexEnclosure(lsp_parameters, "three", "circular")      
lsp_sys_4 = BassReflexEnclosure(lsp_parameters, "four", "circular")

response_1, Ze_1 = lsp_sys_1.calculate_impedance_response(frequencies)
response_2, Ze_2 = lsp_sys_2.calculate_impedance_response(frequencies)
response_3, Ze_3 = lsp_sys_3.calculate_impedance_response(frequencies)
response_4 , Ze_4 = lsp_sys_4.calculate_impedance_response(frequencies)


fig1, ax1 = plt.subplots()
ax1.semilogx(frequencies, response_1, label="1 Loudspeaker with Circ. port")
ax1.semilogx(frequencies, response_2, label="2 Loudspeakers with Circ. port")
ax1.semilogx(frequencies, response_3, label="3 Loudspeakers with Circ. port")
ax1.semilogx(frequencies, response_4, label="4 Loudspeakers with Circ. port")
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Response (dB)")
ax1.set_title("System Response")
ax1.grid(which="both")
ax1.legend()
fig1.show()
fig1.savefig("1_to_4_pistons_responses_Circ_7.7.png")

fig2, ax2 = plt.subplots()
ax2.semilogx(frequencies, Ze_1, label="1 Loudspeaker with Circ. port")
ax2.semilogx(frequencies, Ze_2, label="2 Loudspeakers with Circ. port")
ax2.semilogx(frequencies, Ze_3, label="3 Loudspeakers with Circ. port")
ax2.semilogx(frequencies, Ze_4, label="4 Loudspeakers with Circ. port")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Impedance (Ohm)")
ax2.set_title("System Impedance")
ax2.grid(which="both")
ax2.legend()
fig2.show()
fig2.savefig("1_to_4_pistons_impedances_Circ_7.7.png")

  