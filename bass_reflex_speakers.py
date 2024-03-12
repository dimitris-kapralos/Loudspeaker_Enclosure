"""Loudspeaker system design calculations."""

import math
import matplotlib.pyplot as plt
import numpy as np
import mpmath
import scipy.special
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
        self.Vab = lsp_par["Vab"]
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

    def __init__(self, lsp_par, speaker_type, port_shape):
        """Initialize a loudspeaker system object."""
        self.speaker_type = speaker_type
        self.port_shape = port_shape

        # Parameters
        self.lsp = Loudspeaker(lsp_par)

        self.Vp = 2.3  # volume of the port
        self.fb = 36  # tuning frequency
        self.t = 0.41  # length of the port
        self.Sp = 0.005  # area of the port
        # volume of the enclosure without the volume of # lining
        # material in m^3
        self.Va = 17.2
        self.Vm = 6.1  # volume of the lining material in m^3
        self.Vb = 22.9  # total volume of the enclosure in m^3
        self.a_p = np.sqrt(self.Sp / np.pi)  # radius of the port

        self.porosity = 0.99  # porosity
        self.P_0 = 10**5  # atmospheric pressure
        self.u = 0.03  # flow velocity in the material in m/s
        self.m = 1.86 * 10 ** (-5)  # viscosity coefficient in N.s/m^2
        self.r = 50 * 10 ** (-6)  # fiber diameter
        self.truncation_limit = 10  # Truncation limit for the double summation
        self.d = 0.064  # the thickness of the lining material

        self.lx = 0.15  # width of the enclosure
        self.ly = 0.5  # height of the enclosure
        self.lz = 0.192  # depth of the enclosure

        self.x1 = 0.075  # distance from the center of the diaphragm to the box wall
        self.x2 = 0.075  # distance from the center of the diaphragm to the box wall
        self.y1 = 0.095  # distance from the center of the diaphragm to the box wall
        self.y2 = 0.32  # distance from the center of the diaphragm to the box wall
        self.q = self.ly / self.lx  # aspect ratio of the enclosure
        self.a1 = 0.15  # width of the diphragm
        self.b1 = 0.15  # height of the diaphragm
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

    def calculate_diaphragms_radiation_impedance_Za1(self, f):
        "Calculate the diaphragm radiation impedance based on equation 13.339 and 13.345."

        k = (2 * np.pi * f / SOUND_CELERITY) * (
            (1 + self.a3 * (self.R_f / f) ** self.b3)
            - 1j * self.a4 * (self.R_f / f) ** self.b4
        )

        # Calculate the Bessel and Struve functions
        J1 = jv(1, k * self.lsp.a)
        H1 = mpmath.struveh(1, k * self.lsp.a)

        z11 = (
            R_0
            * SOUND_CELERITY
            * ((1 - (J1**2 / (k * self.lsp.a))) + 1j * (H1 / (k * self.lsp.a)))
        )

        # Calculate Z12
        z12 = (2 * R_0 * SOUND_CELERITY) / np.sqrt(np.pi)
        sum_mn = 0
        for m in range(self.truncation_limit + 1):
            for n in range(self.truncation_limit + 1):
                term1 = ((k * self.lsp.a) / (k * self.d1)) ** m
                term2 = ((k * self.lsp.a) / (k * self.d1)) ** n
                term3 = gamma(m + n + 0.5)
                term4 = jv(m + 1, k * self.lsp.a) * jv(n + 1, k * self.lsp.a)
                term5 = 1 / (np.math.factorial(m) * np.math.factorial(n))
                term6 = spherical_jn(m + n, k * self.d1) + 1j * spherical_yn(
                    m + n, k * self.d1
                )
                sum_mn += term1 * term2 * term3 * term4 * term5 * term6
        z12 *= sum_mn

        Za1 = (self.lsp.a**2 * z11 + 2 * self.lsp.a * self.lsp.a * z12) / (
            self.lsp.a**2 + self.lsp.a**2
        )

        return Za1

    def calculate_box_impedance_for_rectangular_piston_Z11(self, f):
        "Calculate the box impedance based on equation 7.131 for rectangular loudspeaker."

        k = (2 * np.pi * f / SOUND_CELERITY) * (
            (1 + self.a3 * (self.R_f / f) ** self.b3)
            - 1j * self.a4 * (self.R_f / f) ** self.b4
        )
        Zs = R_0 * SOUND_CELERITY + self.P_0 / (1j * 2 * np.pi * f * self.d)

        sum1 = 0
        sum2 = 0
        sum3 = 0

        for n in range(1, self.truncation_limit + 1):
            k0n = np.sqrt(
                k**2 - (0 * np.pi / self.lx) ** 2 - (n * np.pi / self.ly) ** 2
            )
            term1 = (
                (k)
                / (k0n * n**2)
                * np.cos(n * np.pi * self.y1 / self.ly)
                * np.cos(n * np.pi * self.y1 / self.ly)
                * np.sin((n * np.pi * self.b1) / (2 * self.ly))
                * np.sin((n * np.pi * self.b1) / (2 * self.ly))
            )
            term2 = (
                (k0n * Zs) / (k * R_0 * SOUND_CELERITY) + 1j * np.tan(k0n * self.lz)
            ) / (
                1
                + 1j * ((k0n * Zs) / (k * R_0 * SOUND_CELERITY)) * np.tan(k0n * self.lz)
            )
            sum1 += term1 * term2

        for m in range(1, self.truncation_limit + 1):
            km0 = np.sqrt(
                k**2 - (2 * m * np.pi / self.lx) ** 2 - (0 * np.pi / self.ly) ** 2
            )
            term1 = (
                (k)
                / (km0 * m**2)
                * np.sin(m * np.pi * self.a1 / self.lx)
                * np.sin(m * np.pi * self.a1 / self.lx)
            )
            term2 = (
                (km0 * Zs) / (k * R_0 * SOUND_CELERITY) + 1j * np.tan(km0 * self.lz)
            ) / (
                1
                + 1j * ((km0 * Zs) / (k * R_0 * SOUND_CELERITY)) * np.tan(km0 * self.lz)
            )
            sum2 += term1 * term2

        for m in range(1, self.truncation_limit + 1):
            for n in range(1, self.truncation_limit + 1):
                kmn = np.sqrt(
                    k**2 - (2 * m * np.pi / self.lx) ** 2 - (n * np.pi / self.ly) ** 2
                )
                term1 = (
                    (k)
                    / (kmn * m**2 * n**2)
                    * np.sin(m * np.pi * self.a1 / self.lx)
                    * np.sin(n * np.pi * self.a1 / self.lx)
                    * np.cos((m * np.pi * self.y1) / (self.ly))
                )
                term2 = (
                    np.cos(n * np.pi * self.y1 / self.ly)
                    * np.sin((n * np.pi * self.b1) / (2 * self.ly))
                    * np.sin((n * np.pi * self.b1) / (2 * self.ly))
                )
                term3 = (
                    (kmn * Zs) / (k * R_0 * SOUND_CELERITY) + 1j * np.tan(kmn * self.lz)
                ) / (
                    1
                    + 1j
                    * ((kmn * Zs) / (k * R_0 * SOUND_CELERITY))
                    * np.tan(kmn * self.lz)
                )

            sum3 += term1 * term2 * term3

        Z11 = (
            R_0
            * SOUND_CELERITY
            * (
                1
                / (self.lx * self.ly)
                * (
                    ((Zs / (R_0 * SOUND_CELERITY)) + 1j * np.tan(k * self.lz))
                    / (1 + 1j * ((Zs / (R_0 * SOUND_CELERITY)) * np.tan(k * self.lz)))
                )
                + (8 * self.ly) / (np.pi**2 * self.b1 * self.b1 * self.lx) * sum1
                + (2 * self.lx) / (np.pi**2 * self.a1 * self.a1 * self.ly) * sum2
                + (16 * self.lx * self.ly)
                / (np.pi**4 * self.a1 * self.a1 * self.b1 * self.b1)
                * sum3
            )
        )
        return Z11

    def calculate_box_impedance_for_rectangular_piston_Z12(self, f):
        k = (2 * np.pi * f / SOUND_CELERITY) * (
            (1 + self.a3 * (self.R_f / f) ** self.b3)
            - 1j * self.a4 * (self.R_f / f) ** self.b4
        )
        Zs = R_0 * SOUND_CELERITY + self.P_0 / (1j * 2 * np.pi * f * self.d)

        sum1 = 0
        sum2 = 0
        sum3 = 0

        for n in range(1, self.truncation_limit + 1):
            k0n = np.sqrt(
                k**2 - (0 * np.pi / self.lx) ** 2 - (n * np.pi / self.ly) ** 2
            )
            term1 = (
                (k)
                / (k0n * n**2)
                * np.cos(n * np.pi * self.y1 / self.ly)
                * np.cos(n * np.pi * self.y2 / self.ly)
                * np.sin((n * np.pi * self.b1) / (2 * self.ly))
                * np.sin((n * np.pi * self.b1) / (2 * self.ly))
            )
            term2 = (
                (k0n * Zs) / (k * R_0 * SOUND_CELERITY) + 1j * np.tan(k0n * self.lz)
            ) / (
                1
                + 1j * ((k0n * Zs) / (k * R_0 * SOUND_CELERITY)) * np.tan(k0n * self.lz)
            )
            sum1 += term1 * term2

        for m in range(1, self.truncation_limit + 1):
            km0 = np.sqrt(
                k**2 - (2 * m * np.pi / self.lx) ** 2 - (0 * np.pi / self.ly) ** 2
            )
            term1 = (
                (k)
                / (km0 * m**2)
                * np.sin(m * np.pi * self.a1 / self.lx)
                * np.sin(m * np.pi * self.a1 / self.lx)
            )
            term2 = (
                (km0 * Zs) / (k * R_0 * SOUND_CELERITY) + 1j * np.tan(km0 * self.lz)
            ) / (
                1
                + 1j * ((km0 * Zs) / (k * R_0 * SOUND_CELERITY)) * np.tan(km0 * self.lz)
            )
            sum2 += term1 * term2

        for m in range(1, self.truncation_limit + 1):
            for n in range(1, self.truncation_limit + 1):
                kmn = np.sqrt(
                    k**2 - (2 * m * np.pi / self.lx) ** 2 - (n * np.pi / self.ly) ** 2
                )
                term1 = (
                    (k)
                    / (kmn * m**2 * n**2)
                    * np.sin(m * np.pi * self.a1 / self.lx)
                    * np.sin(n * np.pi * self.a1 / self.lx)
                    * np.cos((m * np.pi * self.y1) / (self.ly))
                )
                term2 = (
                    np.cos(n * np.pi * self.y2 / self.ly)
                    * np.sin((n * np.pi * self.b1) / (2 * self.ly))
                    * np.sin((n * np.pi * self.b1) / (2 * self.ly))
                )
                term3 = (
                    (kmn * Zs) / (k * R_0 * SOUND_CELERITY) + 1j * np.tan(kmn * self.lz)
                ) / (
                    1
                    + 1j
                    * ((kmn * Zs) / (k * R_0 * SOUND_CELERITY))
                    * np.tan(kmn * self.lz)
                )

            sum3 += term1 * term2 * term3

        Z12 = (
            R_0
            * SOUND_CELERITY
            * (
                1
                / (self.lx * self.ly)
                * (
                    ((Zs / (R_0 * SOUND_CELERITY)) + 1j * np.tan(k * self.lz))
                    / (1 + 1j * ((Zs / (R_0 * SOUND_CELERITY)) * np.tan(k * self.lz)))
                )
                + (8 * self.ly) / (np.pi**2 * self.b1 * self.b1 * self.lx) * sum1
                + (2 * self.lx) / (np.pi**2 * self.a1 * self.a1 * self.ly) * sum2
                + (16 * self.lx * self.ly)
                / (np.pi**4 * self.a1 * self.a1 * self.b1 * self.b1)
                * sum3
            )
        )
        return Z12

    def calculate_box_impedance_for_rectangular_piston_Z22(self, f):
        k = (2 * np.pi * f / SOUND_CELERITY) * (
            (1 + self.a3 * (self.R_f / f) ** self.b3)
            - 1j * self.a4 * (self.R_f / f) ** self.b4
        )
        Zs = R_0 * SOUND_CELERITY + self.P_0 / (1j * 2 * np.pi * f * self.d)

        sum1 = 0
        sum2 = 0
        sum3 = 0

        for n in range(1, self.truncation_limit + 1):
            k0n = np.sqrt(
                k**2 - (0 * np.pi / self.lx) ** 2 - (n * np.pi / self.ly) ** 2
            )
            term1 = (
                (k)
                / (k0n * n**2)
                * np.cos(n * np.pi * self.y2 / self.ly)
                * np.cos(n * np.pi * self.y2 / self.ly)
                * np.sin((n * np.pi * self.b1) / (2 * self.ly))
                * np.sin((n * np.pi * self.b1) / (2 * self.ly))
            )
            term2 = (
                (k0n * Zs) / (k * R_0 * SOUND_CELERITY) + 1j * np.tan(k0n * self.lz)
            ) / (
                1
                + 1j * ((k0n * Zs) / (k * R_0 * SOUND_CELERITY)) * np.tan(k0n * self.lz)
            )
            sum1 += term1 * term2

        for m in range(1, self.truncation_limit + 1):
            km0 = np.sqrt(
                k**2 - (2 * m * np.pi / self.lx) ** 2 - (0 * np.pi / self.ly) ** 2
            )
            term1 = (
                (k)
                / (km0 * m**2)
                * np.sin(m * np.pi * self.a1 / self.lx)
                * np.sin(m * np.pi * self.a1 / self.lx)
            )
            term2 = (
                (km0 * Zs) / (k * R_0 * SOUND_CELERITY) + 1j * np.tan(km0 * self.lz)
            ) / (
                1
                + 1j * ((km0 * Zs) / (k * R_0 * SOUND_CELERITY)) * np.tan(km0 * self.lz)
            )
            sum2 += term1 * term2

        for m in range(1, self.truncation_limit + 1):
            for n in range(1, self.truncation_limit + 1):
                kmn = np.sqrt(
                    k**2 - (2 * m * np.pi / self.lx) ** 2 - (n * np.pi / self.ly) ** 2
                )
                term1 = (
                    (k)
                    / (kmn * m**2 * n**2)
                    * np.sin(m * np.pi * self.a1 / self.lx)
                    * np.sin(n * np.pi * self.a1 / self.lx)
                    * np.cos((m * np.pi * self.y2) / (self.ly))
                )
                term2 = (
                    np.cos(n * np.pi * self.y2 / self.ly)
                    * np.sin((n * np.pi * self.b1) / (2 * self.ly))
                    * np.sin((n * np.pi * self.b1) / (2 * self.ly))
                )
                term3 = (
                    (kmn * Zs) / (k * R_0 * SOUND_CELERITY) + 1j * np.tan(kmn * self.lz)
                ) / (
                    1
                    + 1j
                    * ((kmn * Zs) / (k * R_0 * SOUND_CELERITY))
                    * np.tan(kmn * self.lz)
                )

            sum3 += term1 * term2 * term3

        Z22 = (
            R_0
            * SOUND_CELERITY
            * (
                1
                / (self.lx * self.ly)
                * (
                    ((Zs / (R_0 * SOUND_CELERITY)) + 1j * np.tan(k * self.lz))
                    / (1 + 1j * ((Zs / (R_0 * SOUND_CELERITY)) * np.tan(k * self.lz)))
                )
                + (8 * self.ly) / (np.pi**2 * self.b1 * self.b1 * self.lx) * sum1
                + (2 * self.lx) / (np.pi**2 * self.a1 * self.a1 * self.ly) * sum2
                + (16 * self.lx * self.ly)
                / (np.pi**4 * self.a1 * self.a1 * self.b1 * self.b1)
                * sum3
            )
        )
        return Z22

    def calculate_box_impedance_for_circular_piston_Zxy(self, f , x1, x2, y1, y2):
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
                    np.cos((m * np.pi * x2) / self.lx)
                    * np.cos((n * np.pi * y2) / self.ly)
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
        ) / self.lsp.Sd**2

        return Zxy

    

    def calculate_port_impedance_Za2(self, f, r_d):
        "Calculate the rectangular port impedance based on equation 13.336 and 13.337."

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
                term3 = (k * self.lx / 2) ** (2 * m + 1)
                term4 = (k * self.ly / 2) ** (2 * n + 1)
                sum_Rs += (term1 / term2) * term3 * term4

        Rs_a2 *= sum_Rs

        for m in range(self.truncation_limit + 1):
            term1 = (-1) ** m * self.fm(self.q, m, n)
            term2 = (2 * m + 1) * math.factorial(m) * math.factorial(m + 1)
            term3 = (k * self.lx / 2) ** (2 * m + 1)
            sum_Xs += (term1 / term2) * term3

        Xs_a2 = ((2 * r_d * SOUND_CELERITY) / (np.sqrt(np.pi))) * (
            (1 - np.sinc(k * self.lx)) / (self.q * k * self.lx)
            + (1 - np.sinc(self.q * k * self.lx)) / (k * self.lx)
            + sum_Xs
        )

        Z_a2 = (Rs_a2 + 1j * Xs_a2) / self.Sp

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

            # diaphrams radiation impedance based on equation 13.339 and 13.345
            Z_a1 = self.calculate_diaphragms_radiation_impedance_Za1(frequencies[i])

            # calculate the leakage resistance
            CAA = (self.Va * 10 ** (-3)) / (
                1.4 * self.P_0
            )  # compliance of the air in the box
            CAM = (
                self.Vm * 10 ** (-3)
            ) / self.P_0  # compliance of the air in the lining material
            Cab = CAA + 1.4 * CAM  # apparent compliance of the air in the box
            Ral = 7 / (2 * np.pi * self.fb * Cab)
            # print(Ral)

            kv = np.sqrt((-1j * np.pi * 2 * frequencies[i] * R_0) / self.m)
            ξ = 0.998 + 0.001j
            kp = (2 * np.pi * frequencies[i] * ξ) / SOUND_CELERITY
            Zp = (R_0 * SOUND_CELERITY * ξ) / (self.Sp)

            if self.speaker_type == "rectangular" and self.port_shape == "rectangular":
                Z11 = self.calculate_box_impedance_for_rectangular_piston_Z11(
                    frequencies[i]
                )
                Z12 = self.calculate_box_impedance_for_rectangular_piston_Z12(
                    frequencies[i]
                )
                Z22 = self.calculate_box_impedance_for_rectangular_piston_Z22(
                    frequencies[i]
                )
                Z21 = Z12

                Kn = LM / self.a2
                Bu = (2 * 0.9 ** (-1) - 1) * Kn

                # dynamic density based on equation 4.233
                r_d = (-8 * R_0) / ((1 + 4 * Bu) * kv**2 * self.a2**2)
                Z_a2 = self.calculate_port_impedance_Za2(frequencies[i], r_d)

            elif self.speaker_type == "circular" and self.port_shape == "rectangular":
                Z11 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i], self.x1, self.x1, self.y1, self.y1 
                )
                Z12 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i] , self.x1, self.x2, self.y1, self.y2
                )
                Z22 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i] , self.x2, self.x2, self.y2, self.y2
                )
                Z21 = Z12

                Kn = LM / self.a2
                Bu = (2 * 0.9 ** (-1) - 1) * Kn

                # dynamic density based on equation 4.233
                r_d = (-8 * R_0) / ((1 + 4 * Bu) * kv**2 * self.a2**2)
                Z_a2 = self.calculate_port_impedance_Za2(frequencies[i], r_d)

            elif self.speaker_type == "circular" and self.port_shape == "circular":
                Z11 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i], self.x1, self.x1, self.y1, self.y1 
                )
                Z12 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i] , self.x1, self.x2, self.y1, self.y2
                )
                Z22 = self.calculate_box_impedance_for_circular_piston_Zxy(
                    frequencies[i] , self.x2, self.x2, self.y2, self.y2
                )
                Z21 = Z12

                # calculate the impedance for circular port
                R_s = R_0 * SOUND_CELERITY * k**2 * self.a_p**2
                X_s = (R_0 * SOUND_CELERITY * 8 * k * self.a_p) / (3 * np.pi)
                Z_a2 = R_s + 1j * X_s

            # 2-port network parameters
            b11 = Z11 / Z21
            b12 = (Z11 * Z22 - Z12 * Z21) / Z21
            b21 = 1 / Z21
            b22 = Z22 / Z21

            C = np.array([[1, 1 * Z_e], [0, 1]])
            E = np.array([[0, 1 * self.lsp.Bl], [1 / self.lsp.Bl, 0]])
            D = np.array([[1, 1 * Z_md], [0, 1]])
            M = np.array([[1 * self.lsp.Sd, 0], [0, 1 / self.lsp.Sd]])
            F = np.array([[1, 1 * Z_a1], [0, 1]])
            L = np.array([[1, 0], [1 / Ral, 1]])
            B = np.array([[b11, b12], [b21, b22]])
            P = np.array(
                [
                    [np.cos(kp * self.t), 1j * Zp * np.sin(kp * self.t)],
                    [1j * (1 / Zp) * np.sin(kp * self.t), np.cos(kp * self.t)],
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

        return response, Ze

    # Define a range of frequencies


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
    "Vab": 17.6,
    "Cms": 0.00119,
    "Mms": 0.0156,
    "Bl": 6.59,
}

# Create an instance of the BassReflexEnclosure class
lsp_sys_1 = BassReflexEnclosure(lsp_parameters, "rectangular", "rectangular")
lsp_sys_2 = BassReflexEnclosure(lsp_parameters, "circular", "rectangular")
lsp_sys_3 = BassReflexEnclosure(lsp_parameters, "circular", "circular")

# Calculate the impedance response
response_1, Ze_1 = lsp_sys_1.calculate_impedance_response(frequencies)
response_2, Ze_2 = lsp_sys_2.calculate_impedance_response(frequencies)
response_3, Ze_3 = lsp_sys_3.calculate_impedance_response(frequencies)

# plot on the same graph the response of the two cases
fig1, ax1 = plt.subplots()
ax1.semilogx(frequencies, response_1, label="1 Rect. Loudspeaker with Rect. port")
ax1.semilogx(frequencies, response_2, label="1 Circ. Loudspeaker with Rect. port")
ax1.semilogx(frequencies, response_3, label="1 Circ. Loudspeaker with Circ. port")
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Response (dB)")
ax1.set_title("System Response")
ax1.grid(which="both")
ax1.legend()
fig1.show()
fig1.savefig("Ex7.3-2_pistons_responses.png")

# plot on the same graph the impedance of the two cases
fig2, ax2 = plt.subplots()
ax2.semilogx(frequencies, Ze_1, label="2 Rect. Loudspeakers with Rect. port")
ax2.semilogx(frequencies, Ze_2, label="2 Circ. Loudspeakers with Rect. port")
ax2.semilogx(frequencies, Ze_3, label="2 Circ. Loudspeakers with Circ. port")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Response (dB)")
ax2.set_title("System Response")
ax2.grid(which="both")
ax2.legend()
fig2.show()
fig2.savefig("Ex7.3-2_pistons_impedances.png")
