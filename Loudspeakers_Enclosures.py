import numpy as np
import mpmath
from scipy.special import j1, lpmv, spherical_jn, spherical_yn, hankel2, jv
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
M = 1.86 * 10 ** (-5)
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
        self.truncation_limit = clb_par["truncation_limit"]
        self.calculate_R_f()

    def calculate_box_volume(self, number_of_speakers):
        """Calculate the volume of the enclosure and the lining material."""
        self.Vab = number_of_speakers * self.lsp.Vas / 1.5819
        self.Va = self.Vab / (1 + GAMMA / 3)
        self.Vm = self.Va / 3
        self.Vb = self.Va + self.Vm

    def calculate_R_f(self):
        """Calculate flow resistance of lining material, Eq. 7.8."""
        self.R_f = ((4 * M * (1 - POROSITY)) / (POROSITY * R**2)) * (
            (1 - 4 / np.pi * (1 - POROSITY))
            / (2 + np.log((M * POROSITY) / (2 * R * R_0 * U)))
            + (6 / np.pi) * (1 - POROSITY)
        )

    def calculate_wave_number(self, f):
        """Calculate the wave number k."""
        self.k = (2 * np.pi * f / SOUND_CELERITY) * (
            (1 + A3 * (self.R_f / f) ** B3) - 1j * A4 * (self.R_f / f) ** B4
        )

    def calculate_diaphragm_radiation_impedance(self):
        H1 = mpmath.struveh(1, 2 * self.k * self.lsp.a)
        R_sp = (
            R_0
            * SOUND_CELERITY
            * (1 - jv(1, 2 * self.k * self.lsp.a) / (self.k * self.lsp.a))
        )
        X_sp = R_0 * SOUND_CELERITY * (H1 / (self.k * self.lsp.a))
        Z_a2 = R_sp + 1j * X_sp

        return Z_a2

    def calculate_simplified_diaphragm_radiation_impedance(self, f):
        Rar = 0.01076 * f**2
        Xar = 1.5 * f / self.lsp.a
        Z_a1 = 1 * (Rar + 1j * Xar)

        return Z_a1

    def calculate_simplified_box_impedance_Zab(self, f, B):
        """Calculate the simplified box impedance for circular loudspeaker."""
        Mab = float(B) * R_0 / (np.pi * self.lsp.a)

        CAA = (self.Va * 10 ** (-3)) / (1.4 * P_0)
        CAM = (self.Vm * 10 ** (-3)) / P_0

        Xab = 2 * np.pi * f * Mab - 1 / (2 * np.pi * f * (CAA + CAM))

        Ram = R_0 * SOUND_CELERITY / (self.lx * self.ly)

        Rab = Ram / (
            (1 + self.Va / (1.4 * self.Vm)) ** 2
            + (2 * np.pi * f) ** 2 * Ram**2 * CAA**2
        )
        # print(Rab)

        Zab = 1 * (Rab + 1j * Xab)

        return Zab

    def calculate_box_impedance_for_circular_piston_Zxy(self, f, x1, y1):
        "Calculate the complex box impedance for circular loudspeaker."

        Zs = R_0 * SOUND_CELERITY + P_0 / (1j * 2 * np.pi * f * self.d)

        sum_mn = 0
        for m in range(self.truncation_limit + 1):
            for n in range(self.truncation_limit + 1):
                kmn = np.sqrt(
                    self.k**2 - (m * np.pi / self.lx) ** 2 - (n * np.pi / self.ly) ** 2
                )
                delta_m0 = 1 if m == 0 else 0
                delta_n0 = 1 if n == 0 else 0
                term1 = (
                    (kmn * Zs) / (self.k * R_0 * SOUND_CELERITY)
                    + 1j * np.tan(kmn * self.lz)
                ) / (
                    1
                    + 1j
                    * ((kmn * Zs) / (self.k * R_0 * SOUND_CELERITY))
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
                    ((Zs / (R_0 * SOUND_CELERITY)) + 1j * np.tan(self.k * self.lz))
                    / (
                        1
                        + 1j
                        * ((Zs / (R_0 * SOUND_CELERITY)) * np.tan(self.k * self.lz))
                    )
                )
                + 4 * self.k * self.lsp.a * self.lsp.a * self.lx * self.ly * sum_mn
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

        alpha = np.arcsin(self.lsp.a / self.R)
        Z_a1 = (2 * R_0 * SOUND_CELERITY) / (self.r**2 * np.sin(alpha) ** 2)
        sum_n = 0
        for n in range(0, self.truncation_limit):
            if n == 0:
                An = (
                    1j
                    * self.k
                    * (-1)
                    * np.sin(alpha)
                    / (self.k * (-(1) * hankel2(1, self.k * self.r)))
                )
            elif n == 1:
                An = (
                    1j
                    * self.k
                    * (np.cos(alpha) ** 3 - 1)
                    / (
                        self.k
                        / (2 * 1 + 1)
                        * (
                            1 * hankel2(1 - 1, self.k * self.r)
                            - (1 + 1) * hankel2(1 + 1, self.k * self.r)
                        )
                    )
                )
            else:
                An = (
                    1j
                    * self.k
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
                            self.k
                            / (2 * n + 1)
                            * (
                                n * hankel2(n - 1, self.k * self.r)
                                - (n + 1) * hankel2(n + 1, self.k * self.r)
                            )
                        )
                    )
                )
            term3, _ = quad(self.integrand, 0, alpha, args=(n, self.k))
            sum_n += An * term3
        Z_a1 *= sum_n
        return Z_a1

    def calculate_system_response(self, frequencies):
        """Calculate the system response."""
        # Preallocate memory
        response = np.zeros_like(frequencies)
        Ze = np.zeros_like(frequencies)
        power = np.zeros_like(frequencies)

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
            self.calculate_wave_number(frequencies[i])

            # calculate the simplified diaphragm radiation impedance
            Z_a1 = self.calculate_simplified_diaphragm_radiation_impedance(
                frequencies[i]
            )

            # calculate the complex diaphragm radiation impedance
            Z_a2 = self.calculate_diaphragm_radiation_impedance()

            # calculate the simplified box impedance for circular loudspeaker
            Zab = self.calculate_simplified_box_impedance_Zab(frequencies[i], B=0.46)

            # calculate the complex box impedance for circular loudspeaker
            Zxy = self.calculate_box_impedance_for_circular_piston_Zxy(
                frequencies[i], x1=0.1, y1=0.2
            )

            # transmission line matrices method
            C = np.array([[1, 1 / 1 * Z_e], [0, 1]])
            E = np.array([[0, 1 * self.lsp.Bl], [1 / (1 * self.lsp.Bl), 0]])
            D = np.array([[1, 1 * Z_md], [0, 1]])
            M = np.array([[1 * self.lsp.Sd, 0], [0, 1 / (1 * self.lsp.Sd)]])
            F = np.array([[1, 1 * Z_a1], [0, 1]])
            B = np.array([[1, 0], [1 / Zab, 1]])

            A = np.dot(np.dot(np.dot(np.dot(np.dot(C, E), D), M), F), B)

            a11 = A[0, 0]
            # a12 = A[0, 1]
            a21 = A[1, 0]
            # a22 = A[1, 1]

            p_6 = self.lsp.e_g / a11
            U_c = p_6 / Zab

            # calculate the system response
            U_ref = (1 * self.lsp.e_g * self.lsp.Bl * self.lsp.Sd) / (
                2 * np.pi * frequencies[i] * self.lsp.Mms * self.lsp.Re
            )
            response[i] = 20 * np.log10((np.abs(U_c)) / (np.abs(U_ref)))

            # calculate the system impedance
            Ze[i] = np.abs((a11) / (a21))

            # calculate the power Lw
            Rmr = ((2 * np.pi * frequencies[i]) ** 2 * (self.lsp.Sd) ** 2 * R_0) / (
                2 * np.pi * SOUND_CELERITY
            )
            W = np.abs((U_c) / (np.sqrt(2) * self.lsp.Sd)) ** 2 * 1 * Rmr
            W_ref = 10 ** (-12)
            power[i] = 10 * np.log10(W / W_ref)
            
        return response, Ze, power


