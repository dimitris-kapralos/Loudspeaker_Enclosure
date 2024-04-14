import numpy as np
import mpmath
from scipy.special import (
    j1,
    lpmv,
    spherical_jn,
    spherical_yn,
    hankel2,
    jv,
    gamma,
    hyp2f1,
    binom,
)
import math
from scipy.integrate import quad


# Speed of sound propagation in air, m/s
SOUND_CELERITY = 343
# Air density, kg/m^3
R_0 = 1.18
# Molecular mean free path length between collisions
LM = 6 * 10 ** (-8)
# porosity
POROSITY = 0.99
# atmospheric pressure, Pa
P_0 = 10**5
# flow velocity in the material, m/s
U = 0.03
# viscosity coefficient, N.s/m^2
m = 1.86 * 10 ** (-5)
# fiber diameter, m
R = 60 * 10 ** (-6)
# adiabatic index
GAMMA = 1.4

# Values of coefficients for characteristic impedance and wave number, respectively, of a homogenous absorbent material.
A3 = 0.0858
A4 = 0.175
B3 = 0.7
B4 = 0.59


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
        self.Mmd = self.Mms - 16 * R_0 * self.a**3 / 3
        self.radius = lsp_par["radius"]
        self.height = lsp_par["height"]
        self.volume = np.pi * self.radius**2 * self.height

    def calculate_R_f(self):
        """Calculate flow resistance of lining material, Eq. 7.8."""
        R_f = ((4 * m * (1 - POROSITY)) / (POROSITY * R**2)) * (
            (1 - 4 / np.pi * (1 - POROSITY))
            / (2 + np.log((m * POROSITY) / (2 * R * R_0 * U)))
            + (6 / np.pi) * (1 - POROSITY)
        )

        return R_f

    def calculate_wave_number(self, f):
        """Calculate the wave number k."""
        R_f = self.calculate_R_f()
        k = (2 * np.pi * f / SOUND_CELERITY) * (
            (1 + A3 * (R_f / f) ** B3) - 1j * A4 * (R_f / f) ** B4
        )
        return k

    def calculate_spl(self, f):
        """Calculate the sound pressure level."""
        k = self.calculate_wave_number(f)
        H1 = mpmath.struveh(1, 2 * k * self.a)
        Zmt = (
            self.Bl**2 / (self.Re + 1j * 2 * np.pi * f * self.Le)
            + 1j * 2 * np.pi * f * self.Mmd
            + self.Rms
            + 1 / (1j * 2 * np.pi * f * self.Cms)
            + 2
            * self.Sd
            * R_0
            * SOUND_CELERITY
            * (1 - jv(1, 2 * k * self.a) / (k * self.a) + 1j * H1 / (k * self.a))
        )
        u_c = self.e_g * self.Bl / ((self.Re + 1j * 2 * np.pi * f * self.Le) * Zmt)
        p_rms = R_0 * f * self.Sd * u_c
        pref = 20e-6
        SPL = 20 * np.log10(float(np.abs(p_rms)) / float(np.abs(pref)))

        return SPL

    def calculate_sound_power(self, f):
        """Calculate the sound power."""
        Rmr = ((2 * np.pi * f) ** 2 * self.Sd**2 * R_0) / (2 * np.pi * SOUND_CELERITY)
        Rm = self.Bl**2 / self.Re + self.Rms + 2 * Rmr
        Mm1 = 2.67 * self.a**3 * R_0
        Xm = 2 * np.pi * f * (self.Mmd + 2 * Mm1) - 1 / (2 * np.pi * f * self.Cms)
        W = (
            np.abs(self.e_g / np.sqrt(2)) ** 2
            * (2 * (self.Bl**2) * Rmr)
            / (self.Re**2 * (Rm**2 + Xm**2))
        )
        Wref = 10 ** (-12)
        power = 10 * np.log10(np.abs(W) / np.abs(Wref))
        
        return power
    
    
    def calculate_loudspeaker_response(self, frequencies):
        """Calculate the system response."""
        # Preallocate memory
        power = np.zeros_like(frequencies)
        spl = np.zeros_like(frequencies)
        for i in range(len(frequencies)):
            # calculate the sound power
            W = self.calculate_sound_power(frequencies[i])
            power[i] = W

            # calculate the sound pressure level
            SPL = self.calculate_spl(frequencies[i])
            spl[i] = SPL
        return power, spl

class ClosedBoxEnclosure:
    """Loudspeaker enclosure parameters class."""

    def __init__(self, clb_par, lsp_par):
        """Initialize the loudspeaker enclosure object."""

        self.lsp = Loudspeaker(lsp_par)

        self.calculate_box_volume(clb_par["number_of_speakers"])
        self.lx = clb_par["lx"]
        self.ly = clb_par["ly"]
        self.lz = clb_par["lz"]
        self.r = clb_par["r"]
        self.d = clb_par["d"]
        self.Wmax = clb_par["Wmax"]
        self.truncation_limit = clb_par["truncation_limit"]

    def calculate_box_volume(self, number_of_speakers):
        """Calculate the volume of the enclosure and the lining material."""
        self.Vab = number_of_speakers * self.lsp.Vas * 0.5242  # (/ 1.5819)
        self.Va = self.Vab / (1 + GAMMA / 3)
        self.Vm = self.Va / 3
        self.Vb = self.Va + self.Vm

    def calculate_diaphragm_radiation_impedance(self, f):
        k = self.lsp.calculate_wave_number(f)
        H1 = mpmath.struveh(1, 2 * k * self.lsp.a)
        R_sp = R_0 * SOUND_CELERITY * (1 - jv(1, 2 * k * self.lsp.a) / (k * self.lsp.a))
        X_sp = R_0 * SOUND_CELERITY * (H1 / (k * self.lsp.a))
        Z_a2 = R_sp + 1j * X_sp

        return Z_a2

    def calculate_simplified_diaphragm_radiation_impedance(self, f):
        Rar = 0.01076 * f**2
        Xar = 1.5 * f / self.lsp.a
        Z_a1 = 1 * (Rar + 1j * Xar)

        return Z_a1

    def calculate_simplified_box_impedance_Zab(self, f, Va, Vm, B):
        """Calculate the simplified box impedance for circular loudspeaker."""
        Mab = float(B) * R_0 / (np.pi * self.lsp.a)

        CAA = (Va * 10 ** (-3)) / (1.4 * P_0)
        CAM = (Vm * 10 ** (-3)) / P_0

        Xab = 2 * np.pi * f * Mab - 1 / (2 * np.pi * f * (CAA + CAM))

        Ram = R_0 * SOUND_CELERITY / (self.lx * self.ly)

        Rab = Ram / (
            (1 + Va / (1.4 * Vm)) ** 2
            + (2 * np.pi * f) ** 2 * Ram**2 * CAA**2
        )
        # print(Rab)

        Zab = 1 * (Rab + 1j * Xab)

        return Zab

    def calculate_box_impedance_for_circular_piston_Zxy(self, f, x1, y1):
        "Calculate the complex box impedance for circular loudspeaker."
        k = self.lsp.calculate_wave_number(f)

        Zs = R_0 * SOUND_CELERITY + P_0 / (1j * 2 * np.pi * f * self.d)

        sum_mn = 0
        for m in range(self.truncation_limit + 1):
            for n in range(self.truncation_limit + 1):
                kmn = np.sqrt(
                    k**2 - (m * np.pi / self.lx) ** 2 - (n * np.pi / self.ly) ** 2
                )
                delta_m0 = 1 if m == 0 else 0
                delta_n0 = 1 if n == 0 else 0
                term1 = (
                    (kmn * Zs) / (k * R_0 * SOUND_CELERITY) + 1j * np.tan(kmn * self.lz)
                ) / (
                    1
                    + 1j
                    * ((kmn * Zs) / (k * R_0 * SOUND_CELERITY))
                    * np.tan(kmn * self.lz)
                )
                term2 = (
                    (2 - delta_m0)
                    * (2 - delta_n0)
                    / (
                        kmn * (n**2 * self.lx**2 + m**2 * self.ly**2)
                        + delta_m0 * delta_n0
                    )
                )
                term3 = (
                    np.cos((m * np.pi * x1) / self.lx)
                    * np.cos((n * np.pi * y1) / self.ly)
                    * j1(
                        (
                            np.pi
                            * self.lsp.a
                            * np.sqrt(n**2 * self.lx**2 + m**2 * self.ly**2)
                        )
                        / (self.lx * self.ly)
                    )
                )
                term4 = (
                    np.cos((m * np.pi * x1) / self.lx)
                    * np.cos((n * np.pi * y1) / self.ly)
                    * j1(
                        (
                            np.pi
                            * self.lsp.a
                            * np.sqrt(n**2 * self.lx**2 + m**2 * self.ly**2)
                        )
                        / (self.lx * self.ly)
                    )
                )
                sum_mn += term1 * term2 * term3 * term4

        Zxy = (
            R_0
            * SOUND_CELERITY
            * (
                (self.lsp.Sd * self.lsp.Sd)
                / (self.lx * self.ly)
                * (
                    ((Zs / (R_0 * SOUND_CELERITY)) + 1j * np.tan(k * self.lz))
                    / (1 + 1j * ((Zs / (R_0 * SOUND_CELERITY)) * np.tan(k * self.lz)))
                )
                + 4 * k * self.lsp.a * self.lsp.a * self.lx * self.ly * sum_mn
            )
        ) / (self.lsp.Sd * self.lsp.Sd)

        return Zxy

    def integrand(self, thita, n, k):
        """Function to calculate the integrand"""
        r1 = (self.r * np.cos(self.lsp.a)) / np.cos(thita)
        hankel = spherical_jn(n, k * r1) - 1j * spherical_yn(n, k * r1)
        return (hankel) * lpmv(0, n, np.cos(thita)) * r1**2 * np.tan(thita)

    def calculate_circular_Za1(self, f):
        """Calculate the radiation impedance of a piston in a cap."""
        k = self.lsp.calculate_wave_number(f)
        alpha = np.arcsin(self.lsp.a / self.r)
        Z_a1 = (2 * R_0 * SOUND_CELERITY) / (self.r**2 * np.sin(alpha) ** 2)
        sum_n = 0
        for n in range(0, self.truncation_limit):
            if n == 0:
                An = (
                    1j
                    * k
                    * (-1)
                    * np.sin(alpha)
                    / (k * (-(1) * hankel2(1, k * self.r)))
                )
            elif n == 1:
                An = (
                    1j
                    * k
                    * (np.cos(alpha) ** 3 - 1)
                    / (
                        k
                        / (2 * 1 + 1)
                        * (
                            1 * hankel2(1 - 1, k * self.r)
                            - (1 + 1) * hankel2(1 + 1, k * self.r)
                        )
                    )
                )
            else:
                An = (
                    1j
                    * k
                    * (2 * n + 1)
                    * np.sin(alpha)
                    * (
                        np.sin(alpha) * lpmv(0, n, np.cos(alpha))
                        + np.cos(alpha) * lpmv(1, n, np.cos(alpha))
                    )
                    / (
                        2
                        * (n - 1)
                        * (n + 2)
                        * (
                            k
                            / (2 * n + 1)
                            * (
                                n * hankel2(n - 1, k * self.r)
                                - (n + 1) * hankel2(n + 1, k * self.r)
                            )
                        )
                    )
                )
            term3, _ = quad(self.integrand, 0, alpha, args=(n, k))
            sum_n += An * term3
        Z_a1 *= sum_n
        return Z_a1

    def calculate_closed_box_response(self, frequencies, number_of_speakers):
        """Calculate the system response."""
        # Preallocate memory
        response = np.zeros_like(frequencies)
        Ze = np.zeros_like(frequencies)
        power = np.zeros_like(frequencies)
        spl = np.zeros_like(frequencies)

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
            # calculate the wave number k
            self.lsp.calculate_wave_number(frequencies[i])

            # calculate the simplified diaphragm radiation impedance
            Z_a1 = self.calculate_simplified_diaphragm_radiation_impedance(
                frequencies[i]
            )

            # calculate the complex diaphragm radiation impedance
            Z_a2 = self.calculate_diaphragm_radiation_impedance(frequencies[i])

            # calculate the box impedance for piston in a cap
            Z_a3 = self.calculate_circular_Za1(frequencies[i])

            # calculate the simplified box impedance for circular loudspeaker
            Zab = self.calculate_simplified_box_impedance_Zab(frequencies[i], self.Va, self.Vm, B=0.46)

            # calculate the complex box impedance for circular loudspeaker
            Zxy = self.calculate_box_impedance_for_circular_piston_Zxy(
                frequencies[i], x1=0.1, y1=0.2
            )

            # transmission line matrices method
            C = np.array([[1, 1 / number_of_speakers * Z_e], [0, 1]])
            E = np.array([[0, 1 * self.lsp.Bl], [1 / (1 * self.lsp.Bl), 0]])
            D = np.array([[1, number_of_speakers * Z_md], [0, 1]])
            M = np.array(
                [
                    [number_of_speakers * self.lsp.Sd, 0],
                    [0, 1 / (number_of_speakers * self.lsp.Sd)],
                ]
            )
            F = np.array([[1, number_of_speakers * Z_a3], [0, 1]])
            B = np.array([[1, 0], [1 / Zab, 1]])

            A = np.dot(np.dot(np.dot(np.dot(np.dot(C, E), D), M), F), B)

            a11 = A[0, 0]
            # a12 = A[0, 1]
            a21 = A[1, 0]
            # a22 = A[1, 1]

            p_6 = self.lsp.e_g / a11
            U_c = p_6 / Zab

            # calculate the system response
            U_ref = (number_of_speakers * self.lsp.e_g * self.lsp.Bl * self.lsp.Sd) / (
                2 * np.pi * frequencies[i] * self.lsp.Mms * self.lsp.Re
            )
            response[i] = 20 * np.log10((np.abs(U_c)) / (np.abs(U_ref)))

            # calculate the system impedance
            Ze[i] = np.abs((a11) / (a21))

            # calculate the power Lw
            Rmr = (
                (2 * np.pi * frequencies[i]) ** 2
                * number_of_speakers
                * (self.lsp.Sd) ** 2
                * R_0
            ) / (2 * np.pi * SOUND_CELERITY)
            W = (
                np.abs((U_c) / (np.sqrt(2) * number_of_speakers * self.lsp.Sd)) ** 2
                * 1
                * Rmr
            )
            W_ref = 10 ** (-12)
            power[i] = 10 * np.log10(W / W_ref)

            # calculate the sound pressure level
            prms = R_0 * frequencies[i] * U_c 
            pref = 20e-6
            spl[i] = 20 * np.log10((prms) / (pref))

        return response, Ze, power, spl


class BassReflexEnclosure:
    """Loudspeaker enclosure parameters class."""

    def __init__(self, clb_par, lsp_par):
        """Initialize the loudspeaker enclosure object."""
        self.lsp = Loudspeaker(lsp_par)
        self.clb = ClosedBoxEnclosure(
            clb_par, lsp_par
        )  # Pass lsp_par to initialize ClosedBoxEnclosure

        # Bass reflex enclosure parameters
        # self.Sp = br_par["Sp"]
        # self.t = br_par["t"]
        # self.lm = br_par["lm"]

        # Access attributes from ClosedBoxEnclosure instance
        self.Vab = self.clb.Vab
        self.Vb = self.clb.Vb
        self.Va = self.clb.Va
        self.Vm = self.clb.Vm
        self.lx = self.clb.lx
        self.ly = self.clb.ly
        self.lz = self.clb.lz
        self.r = self.clb.r
        self.d = self.clb.d
        self.Wmax = self.clb.Wmax
        self.truncation_limit = self.clb.truncation_limit

        # Call methods from ClosedBoxEnclosure
        # self.clb.calculate_R_f()
        # self.clb.calculate_wave_number()
        # self.clb.calculate_box_impedance_for_circular_piston_Zxy
        # self.clb.calculate_diaphragm_radiation_impedance()
        # self.clb.calculate_simplified_diaphragm_radiation_impedance()
        # self.clb.calculate_simplified_box_impedance_Zab()

    def calculate_port(self):
        """Calculate the port parameters."""
        # maximum sound pressure level
        spl_max = 20 * np.log10(
            (np.sqrt(self.lsp.Re * self.Wmax) * self.lsp.Bl * self.lsp.Sd * R_0)
            / (2 * np.pi * self.lsp.Re * self.lsp.Mms * 20 * 10 ** (-6))
        )

        pmax = (
            2 * np.sqrt(2) * 10 ** ((spl_max - 7.4) / 20 - 5)
        )  # maximum pressure level

        Vmax = (pmax * 1000) / (
            2 * np.pi * self.lsp.fs**2 * R_0
        )  # maximum volume displacement

        Vp = 10 * Vmax  # volume of the port

        fb = 0.9735 * self.lsp.fs  # fixed value based on alignment

        t = 345 / (2 * np.pi * fb) * np.sqrt((Vp) / (self.Vab))  # length of the port

        Sp = (Vp * 0.001) / t  # area of the port

        return Sp, t, fb

    def calculate_leakage_resistance(self):
        # calculate the leakage resistance
        CAA = (self.Va * 10 ** (-3)) / (1.4 * P_0)  # compliance of the air in the box
        CAM = (
            self.Vm * 10 ** (-3)
        ) / P_0  # compliance of the air in the lining material
        Cab = CAA + 1.4 * CAM  # apparent compliance of the air in the box
        _, _, fb = self.calculate_port()
        Ral = 7 / (2 * np.pi * fb * Cab)
        return Ral

    def calculate_port_impedance_Za2(self, f, r_d, lx, ly, Sp):
        "Calculate the rectangular port impedance based on equation 13.336 and 13.337."
        q = lx / ly

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
                term3 = (self.clb.k * lx / 2) ** (2 * m + 1)
                term4 = (self.clb.k * ly / 2) ** (2 * n + 1)
                sum_Rs += (term1 / term2) * term3 * term4

        Rs_a2 *= sum_Rs

        for m in range(self.truncation_limit + 1):
            term1 = (-1) ** m * self.fm(q, m, n)
            term2 = (2 * m + 1) * math.factorial(m) * math.factorial(m + 1)
            term3 = (self.clb.k * lx / 2) ** (2 * m + 1)
            sum_Xs += (term1 / term2) * term3

        Xs_a2 = ((2 * r_d * SOUND_CELERITY) / (np.sqrt(np.pi))) * (
            (1 - np.sinc(self.clb.k * lx)) / (q * self.clb.k * lx)
            + (1 - np.sinc(q * self.clb.k * lx)) / (self.clb.k * lx)
            + sum_Xs
        )

        Z_a2 = (Rs_a2 + 1j * Xs_a2) / Sp

        return Z_a2

    def fm(self, q, m, n):
        "Helper function for the calculation of the rectangular port impedance based on equation 13.337."
        result1 = hyp2f1(1, m + 0.5, m + 1.5, 1 / (1 + q**2))
        result2 = hyp2f1(1, m + 0.5, m + 1.5, 1 / (1 + q ** (-2)))

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
                * binom((m - n), p - n)
            )

        first_term = binom((2 * m + 3), (2 * n)) * first_sum

        second_sum = 0

        for p in range(m - n, m + 1):
            second_sum += (
                binom((p - m + n), n) * (-1) ** (p - m + n) * (q) ** (2 * n + 2)
            ) / ((2 * p - 1) * (1 + q ** (-2)) ** (p - 1 / 2))

        second_term = binom((2 * m + 3), (2 * n + 3)) * second_sum

        return first_term + second_term

    def calculate_bass_reflex_response(self, frequencies, number_of_speakers):
        """Calculate the system response."""
        # Preallocate memory
        response = np.zeros_like(frequencies)
        Ze = np.zeros_like(frequencies)
        power = np.zeros_like(frequencies)

        for i in range(len(frequencies)):
            Sp, t, fb = self.calculate_port()

            # calculate the electrical impedance
            Z_e = self.lsp.Re + 1j * 2 * np.pi * frequencies[i] * self.lsp.Le

            # calculate the mechanical impedance
            Mmd = self.lsp.Mms - 16 * R_0 * self.lsp.a**3 / 3
            Z_md = (
                1j * 2 * np.pi * frequencies[i] * Mmd
                + self.lsp.Rms
                + 1 / (1j * 2 * np.pi * frequencies[i] * self.lsp.Cms)
            )
            # calculate the wave number k
            self.lsp.calculate_wave_number(frequencies[i])

            # calculate the simplified diaphragm radiation impedance
            Z_a2 = self.clb.calculate_simplified_diaphragm_radiation_impedance(
                frequencies[i]
            )
            # calculate the simplified box impedance for circular loudspeaker
            Zab = self.clb.calculate_simplified_box_impedance_Zab(
                frequencies[i], self.Va, self.Vm,  B=0.3
            )

            # calculate the complex diaphragm radiation impedance
            Z_a1 = self.clb.calculate_diaphragm_radiation_impedance(frequencies[i])

            # calculate the complex box impedance for circular loudspeaker
            Zxy = self.clb.calculate_box_impedance_for_circular_piston_Zxy(
                frequencies[i], x1=0.1, y1=0.2
            )

            # calculate the box impedance for piston in a cap
            Z_a3 = self.clb.calculate_circular_Za1(frequencies[i])

            Ral = self.calculate_leakage_resistance()

            kv = np.sqrt((-1j * np.pi * 2 * frequencies[i] * R_0) / m)
            ξ = 0.998 + 0.001j
            kp = (2 * np.pi * frequencies[i] * ξ) / SOUND_CELERITY
            Zp = (R_0 * SOUND_CELERITY * ξ) / (Sp)

            # transmission line matrices method
            C = np.array([[1, 1 / number_of_speakers * Z_e], [0, 1]])
            E = np.array([[0, 1 * self.lsp.Bl], [1 / (1 * self.lsp.Bl), 0]])
            D = np.array([[1, number_of_speakers * Z_md], [0, 1]])
            M = np.array(
                [
                    [number_of_speakers * self.lsp.Sd, 0],
                    [0, 1 / (number_of_speakers * self.lsp.Sd)],
                ]
            )
            F = np.array([[1, number_of_speakers * Z_a3], [0, 1]])
            L = np.array([[1, 0], [1 / Ral, 1]])
            B = np.array([[1, 0], [1 / Zab, 1]])
            P = np.array(
                [
                    [np.cos(kp * t), 1j * Zp * np.sin(kp * t)],
                    [1j * (1 / Zp) * np.sin(kp * t), np.cos(kp * t)],
                ]
            )
            R = np.array([[1, 0], [1 / Z_a2, 1]])

            A = np.dot(
                np.dot(
                    np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(C, E), D), M), F), L), B),
                    P,
                ),
                R,
            )

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

            U6 = n21 * p9
            UB = (n21 - 1 / Z_a2) * p9

            # calculate the system response
            U_ref = (number_of_speakers * self.lsp.e_g * self.lsp.Bl * self.lsp.Sd) / (
                2 * np.pi * frequencies[i] * self.lsp.Mms * self.lsp.Re
            )
            response[i] = 20 * np.log10((np.abs(UB)) / (np.abs(U_ref)))

            # calculate the system impedance
            Ze[i] = np.abs((a11) / (a21))

            # calculate the power Lw
            Rmr = (
                (2 * np.pi * frequencies[i]) ** 2
                * number_of_speakers
                * (self.lsp.Sd) ** 2
                * R_0
            ) / (2 * np.pi * SOUND_CELERITY)
            W = (
                np.abs((UB) / (np.sqrt(2) * number_of_speakers * self.lsp.Sd)) ** 2
                * 1
                * Rmr
            )
            W_ref = 10 ** (-12)
            power[i] = 10 * np.log10(W / W_ref)

        return response, Ze, power


class DodecahedronEnclosure:
    """Loudspeaker enclosure parameters class."""

    def __init__(self, clb_par, dod_par ,lsp_par):
        """Initialize the loudspeaker enclosure object."""
        self.lsp = Loudspeaker(lsp_par)
        self.clb = ClosedBoxEnclosure(
            clb_par, lsp_par
        )  # Pass lsp_par to initialize ClosedBoxEnclosure

        # Access attributes from ClosedBoxEnclosure instance
        self.Vab = dod_par["Vab"]

    def calculate_dodecahedron_volume(self):
        """Calculate the volume of the dodecahedron."""
        Vb = self.Vab - self.lsp.volume * 12
        Va = 3*Vb / 4
        Vm = Va / 3 
        
        return Va, Vm, Vb

    def calculate_dodecahedron_response(self, frequencies):
        """Calculate the system response."""
        # Preallocate memory
        response = np.zeros_like(frequencies)
        Ze = np.zeros_like(frequencies)
        power = np.zeros_like(frequencies)
        spl = np.zeros_like(frequencies)

        for i in range(len(frequencies)):
            # calculate the dodecahedron volume
            Va, Vm, Vb = self.calculate_dodecahedron_volume()
            
            # calculate the electrical impedance
            Z_e = self.lsp.Re + 1j * 2 * np.pi * frequencies[i] * self.lsp.Le

            # calculate the mechanical impedance
            Mmd = self.lsp.Mms - 16 * R_0 * self.lsp.a**3 / 3
            Z_md = (
                1j * 2 * np.pi * frequencies[i] * Mmd
                + self.lsp.Rms
                + 1 / (1j * 2 * np.pi * frequencies[i] * self.lsp.Cms)
            )
            # calculate the wave number k
            self.lsp.calculate_wave_number(frequencies[i])

            # calculate the box impedance for piston in a cap
            Z_a3 = self.clb.calculate_circular_Za1(frequencies[i])

            # calculate the simplified box impedance for circular loudspeaker
            Zab = self.clb.calculate_simplified_box_impedance_Zab(frequencies[i],Va, Vm, B=0.3)

            # transmission line matrices method
            C = np.array([[1, 4 / 3 * Z_e], [0, 1]])
            E = np.array([[0, 4 * self.lsp.Bl], [1 / (4 * self.lsp.Bl), 0]])
            D = np.array([[1, 12 * Z_md], [0, 1]])
            M = np.array(
                [
                    [12 * self.lsp.Sd, 0],
                    [0, 1 / (12 * self.lsp.Sd)],
                ]
            )
            F = np.array([[1, 12 * Z_a3], [0, 1]])
            B = np.array([[1, 0], [1 / Zab, 1]])

            A = np.dot(np.dot(np.dot(np.dot(np.dot(C, E), D), M), F), B)

            a11 = A[0, 0]
            # a12 = A[0, 1]
            a21 = A[1, 0]
            # a22 = A[1, 1]

            p_6 = self.lsp.e_g / a11
            U_c = p_6 / Zab

            # calculate the system response
            U_ref = (4 * self.lsp.e_g * self.lsp.Bl * self.lsp.Sd) / (
                2 * np.pi * frequencies[i] * self.lsp.Mms * self.lsp.Re
            )
            response[i] = 20 * np.log10((np.abs(U_c)) / (np.abs(U_ref)))

            # calculate the system impedance
            Ze[i] = np.abs((a11) / (a21))

            # calculate the power Lw
            Rmr = (
                (2 * np.pi * frequencies[i]) ** 2
                * 12
                * (self.lsp.Sd) ** 2
                * R_0
            ) / (2 * np.pi * SOUND_CELERITY)
            W = (
                np.abs((U_c) / (np.sqrt(2) * 12 * self.lsp.Sd)) ** 2
                * 1
                * Rmr
            )
            W_ref = 10 ** (-12)
            power[i] = 10 * np.log10(W / W_ref)

            # calculate the sound pressure level
            prms = R_0 * frequencies[i] * U_c 
            pref = 20e-6
            spl[i] = 20 * np.log10((prms) / (pref))

        return response, Ze, power, spl
    
    