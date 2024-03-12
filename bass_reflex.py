"""Loudspeaker system design calculations."""

import math
import matplotlib.pyplot as plt
import numpy as np
# import mpmath
import scipy.special
from scipy.special import gamma,  j1  


# Speed of sound propagation in air, m/s
SOUND_CELERITY = 343
# Air density, kg/m^3
R_0 = 1.18
# Molecular mean free path length between collisions
LM = 6 * 10**(-8)


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
        self.pmax = lsp_par["pmax"]
        self.Vmax = lsp_par["Vmax"]
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

        self.Vp = 0.3  # volume of the port
        self.fb = 36  # tuning frequency
        self.t = 0.2  # length of the port
        self.Sp = 0.0015  # area of the port
        # volume of the enclosure without the volume of # lining
        # material in m^3
        self.Va = 12
        self.Vm = 4   # volume of the lining material in m^3
        self.Vb = 16  # total volume of the enclosure in m^3
        self.a_p = np.sqrt(self.Sp / np.pi)  # radius of the port

        self.porosity = 0.99  # porosity
        self.P_0 = 10 ** 5  # atmospheric pressure
        self.u = 0.03  # flow velocity in the material in m/s
        self.m = 1.86 * 10**(-5)  # viscosity coefficient in N.s/m^2
        self.r = 50 * 10 ** (-6)  # fiber diameter
        self.truncation_limit = 10  # Truncation limit for the double summation
        self.d = 0.064  # the thickness of the lining material

        self.lx = 0.15  # width of the enclosure
        self.ly = 0.401  # height of the enclosure
        self.lz = 0.228  # depth of the enclosure
        # distance from the center of the diaphragm to the box wall
        self.x1 = 0.075
        # distance from the center of the diaphragm to the box wall
        self.y1 = 0.25
        self.q = self.ly / self.lx  # aspect ratio of the enclosure
        self.a2 = 0.1  # width of the port
        self.b2 = 0.015  # height of the port

        # Values of coefficients (Table 7.1)
        self.a3 = 0.0858
        self.a4 = 0.175
        self.b3 = 0.7
        self.b4 = 0.59
        self.calculate_R_f()

    def calculate_R_f(self):
        """Calculate flow resistance of lining material, Eq. 7.8."""
        self.R_f = (((4*self.m*(1 - self.porosity))
                     / (self.porosity * self.r**2))
                    * ((1 - 4/np.pi * (1 - self.porosity))
                       / (2 + np.log((self.m*self.porosity)
                                     / (2*self.r*R_0*self.u)))
                       + (6/np.pi) * (1 - self.porosity)))
        

    def calculate_box_impedance_ZAB(self, f, lz, a, r_0, lx, ly, truncation_limit):
        " Calculate the box impedance based on equation 7.131."
        # wave number k equation 7.11
        k = ((2 * np.pi * f / SOUND_CELERITY)
                * ((1 + self.a3 * (self.R_f/f)**self.b3)
                - 1j * self.a4 * (self.R_f/f)**self.b4))

        Zs = r_0 * SOUND_CELERITY + self.P_0/(1j * 2 * np.pi * f
                                                * self.d)

        sum_mn = 0
        for m in range(truncation_limit+1):
            for n in range(truncation_limit+1):
                kmn = np.sqrt(k**2 - (m*np.pi/lx)**2 - (n*np.pi/ly)**2)
                delta_m0 = 1 if m == 0 else 0
                delta_n0 = 1 if n == 0 else 0
                term1 = (((kmn*Zs)/(k*r_0*SOUND_CELERITY)
                            + 1j * np.tan(kmn*lz))
                            / (1 + 1j * ((kmn*Zs)/(k*r_0*SOUND_CELERITY))
                            * np.tan(kmn*lz)))
                term2 = ((2 - delta_m0) * (2 - delta_n0)
                            / (kmn * (n**2 * lx**2 + m**2 * ly**2)
                            + delta_m0 * delta_n0))
                term3 = (np.cos((m*np.pi*self.x1)/lx)
                            * np.cos((n*np.pi*self.y1)/ly)
                            * j1((np.pi * a * np.sqrt(n**2 * lx**2
                                + m**2 * ly**2))/(lx*ly)))
                term4 = (np.cos((m*np.pi*self.x1)/lx)
                            * np.cos((n*np.pi*self.y1)/ly)
                            * j1((np.pi * a
                                * np.sqrt(n**2 * lx**2 + m**2 * ly**2))
                                / (lx*ly)))
                sum_mn += term1 * term2 * term3 * term4

        ZAB = ((r_0 * SOUND_CELERITY * ((self.lsp.Sd * self.lsp.Sd)
                                        / (lx*ly)
                * (((Zs/(r_0*SOUND_CELERITY)) + 1j * np.tan(k*lz))
                / (1 + 1j * ((Zs/(r_0*SOUND_CELERITY))
                                * np.tan(k*lz))))
                + 4 * k*a*a*lx*ly * sum_mn)) / self.lsp.Sd**2)

        return ZAB
        

    def calculate_impedance_response(self, frequencies):
        """Perform calculations based on speaker type and port shape."""
        if (self.speaker_type == 'circular'
                and self.port_shape == 'rectangular'):


            # calculate the rectangular port impedance based on
            # equation 13.336, 13.327
            def calculate_port_impedance_Za2(f, r_d, lx, ly, r_0, N):
                k = ((2 * np.pi * f / SOUND_CELERITY)
                     * ((1 + self.a3 * (self.R_f / f)**self.b3)
                        - 1j * self.a4 * (self.R_f / f)**self.b4))

                Rs_a2 = (r_0 * SOUND_CELERITY) / (np.sqrt(np.pi))

                sum_Rs = 0
                sum_Xs = 0

                for m in range(N+1):
                    for n in range(N+1):
                        term1 = (-1) ** (m + n)
                        term2 = ((2*m + 1) * (2*n + 1) * math.factorial(m+1)
                                 * math.factorial(n+1) * gamma(m + n + 3/2))
                        term3 = (k*lx/2) ** (2*m + 1)
                        term4 = (k*ly/2) ** (2*n + 1)
                        sum_Rs += (term1 / term2) * term3 * term4

                Rs_a2 *= sum_Rs

                for m in range(N+1):
                    term1 = (-1) ** m * fm(self.q, m, n)
                    term2 = ((2*m + 1) * math.factorial(m)
                             * math.factorial(m+1))
                    term3 = (k*lx/2) ** (2*m + 1)
                    sum_Xs += (term1 / term2) * term3

                Xs_a2 = (((2 * r_d * SOUND_CELERITY) / (np.sqrt(np.pi))) *
                         ((1 - np.sinc(k*lx))/(self.q*k*lx) +
                          (1 - np.sinc(self.q * k*lx))/(k*lx) + sum_Xs))

                Z_a2 = (Rs_a2 + 1j * Xs_a2) / self.Sp

                return Z_a2

            def fm(q, m, n):
                result1 = scipy.special.hyp2f1(1, m + 0.5, m + 1.5, 1
                                               / (1 + q**2))
                result2 = scipy.special.hyp2f1(1, m + 0.5, m + 1.5, 1
                                               / (1 + q**(-2)))

                sum_fm = 0
                for n in range(m+1):
                    sum_fm = gmn(m, n, q)

                return ((result1 + result2) / ((2*m+1)*(1+q**(-2))**(m+0.5))
                        + (1/(2*m + 3)) * sum_fm)

            def gmn(m, n, q):
                first_sum = 0

                for p in range(n, m + 1):
                    first_sum += (((-1) ** (p - n) * (q) ** (2 * n - 1))
                                  / ((2 * p - 1) * (1 + q**2)**((p - 1/2)))
                                  * scipy.special.binom((m - n), p - n))

                first_term = scipy.special.binom((2 * m + 3),
                                                 (2 * n)) * first_sum

                second_sum = 0

                for p in range(m - n, m + 1):
                    second_sum += ((scipy.special.binom((p - m + n), n)
                                   * (-1)**(p - m + n) * (q)**(2 * n + 2))
                                   / ((2 * p - 1)
                                      * (1 + q**(-2))**((p - 1/2))))

                second_term = scipy.special.binom((2 * m + 3),
                                                  (2 * n + 3)) * second_sum

                return first_term + second_term

            # Preallocate memory
            response = np.zeros_like(frequencies)
            Ze = np.zeros_like(frequencies)

            for i in range(len(frequencies)):

                # Calculate the electrical impedance
                Z_e = (self.lsp.Re + 1j * 2 * np.pi * frequencies[i]
                       * self.lsp.Le)

                # Calculate the mechanical impedance
                Mmd = self.lsp.Mms - 16 * R_0 * self.lsp.a**3 / 3
                Z_md = (1j * 2 * np.pi * frequencies[i] * Mmd + self.lsp.Rms
                        + 1 / (1j * 2 * np.pi * frequencies[i] * self.lsp.Cms))

                # wave number k equation 7.11
                k = ((2 * np.pi * frequencies[i] / SOUND_CELERITY)
                     * ((1 + self.a3 * (self.R_f/frequencies[i])**self.b3) - 1j
                        * self.a4 * (self.R_f/frequencies[i])**self.b4))

                # diaphragm radiation impedance based on equations
                # 13.116 - 13.118
                R_s = R_0 * SOUND_CELERITY * k**2 * self.lsp.a**2
                X_s = (R_0 * SOUND_CELERITY * 8 * k * self.lsp.a)/(3 * np.pi)
                Z_a1 = (R_s + 1j * X_s)

                # calculate box impedance
                Z_ab = self.calculate_box_impedance_ZAB(frequencies[i], self.lz, self.lsp.a, R_0, self.lx, self.ly, self.truncation_limit)

                # calculate the leakage resistance

                # compliance of the air in the box
                CAA = (self.Va*10**(-3))/(1.4*self.P_0)
                # compliance of the air in the lining material
                CAM = (self.Vm*10**(-3))/self.P_0
                # apparent compliance of the air in the box
                Cab = CAA + 1.4 * CAM
                Ral = 7 / (2 * np.pi * self.fb * Cab)
                # print(Ral)

                kv = np.sqrt((-1j * np.pi * 2 * frequencies[i] * R_0)
                             / self.m)
                # ap = np.sqrt((self.Sp) / np.pi)
                # kvap = kv * ap
                # ξ = np.sqrt(1 - 1j * (2 * jv(1, kvap))
                #             / (kvap * jv(0, kvap)))
                ξ = 0.998 + 0.001j
                kp = (2 * np.pi * frequencies[i] * ξ) / SOUND_CELERITY
                Zp = (R_0 * SOUND_CELERITY * ξ) / (self.Sp)
                Kn = LM / self.a2
                Bu = (2 * 0.9**(-1) - 1) * Kn

                # dynamic density based on equation 4.233
                r_d = (-8*R_0) / ((1 + 4*Bu) * kv**2 * self.a2**2)

                # calculate the port impedance
                Z_a2 = calculate_port_impedance_Za2(frequencies[i], r_d,  self.lx, self.ly,
                                     R_0, self.truncation_limit)

                C = np.array([[1, 1 * Z_e], [0, 1]])
                E = np.array([[0, 1 * self.lsp.Bl], [1 / self.lsp.Bl, 0]])
                D = np.array([[1, 1 * Z_md], [0, 1]])
                M = np.array([[1 * self.lsp.Sd, 0], [0, 1 / self.lsp.Sd]])
                F = np.array([[1, 1 * Z_a1], [0, 1]])
                L = np.array([[1, 0], [1 / Ral, 1]])
                B = np.array([[1, 0], [1 / Z_ab, 1]])
                P = np.array([[np.cos(kp*self.t), 1j*Zp*np.sin(kp*self.t)],
                              [1j*(1/Zp)*np.sin(kp*self.t),
                               np.cos(kp*self.t)]])
                R = np.array([[1, 0], [1/Z_a2, 1]])

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
                UB = (n21 - 1/Z_a2) * p9
                # Up = self.lsp.e_g / (a11 * Z_a2)

                # ig = a21 * p9

                U_ref = ((1 * self.lsp.e_g * self.lsp.Bl * self.lsp.Sd)
                         / (2 * np.pi * frequencies[i] * self.lsp.Mms
                            * self.lsp.Re))

                # response[i] = 20 * np.log10(float(abs(U6))
                #                             / float(abs(U_ref)))
                response[i] = 20 * np.log10(float(abs(UB))
                                            / float(abs(U_ref)))
                Ze[i] = abs(((a11) / (a21)))

            return response, Ze

        elif self.speaker_type == 'circular' and self.port_shape == 'circular':


            # Preallocate memory
            response = np.zeros_like(frequencies)
            Ze = np.zeros_like(frequencies)

            for i in range(len(frequencies)):

                # calculate the electrical impedance
                Z_e = (self.lsp.Re + 1j * 2 * np.pi * frequencies[i]
                       * self.lsp.Le)

                # calculate the mechanical impedance
                Mmd = self.lsp.Mms - 16 * R_0 * self.lsp.a**3 / 3
                Z_md = (1j * 2 * np.pi * frequencies[i] * Mmd + self.lsp.Rms
                        + 1 / (1j * 2 * np.pi * frequencies[i] * self.lsp.Cms))

                # wave number k equation 7.11
                k = ((2 * np.pi * frequencies[i] / SOUND_CELERITY)
                     * ((1 + self.a3 * (self.R_f/frequencies[i])**self.b3)
                        - 1j * self.a4 * (self.R_f/frequencies[i])**self.b4))

                # port radiation impedance based on equations 13.116 - 13.118
                R_s = R_0 * SOUND_CELERITY * k**2 * self.lsp.a**2
                X_s = (R_0 * SOUND_CELERITY * 8 * k * self.lsp.a)/(3 * np.pi)
                Z_a1 = (R_s + 1j * X_s)

                # calculate box impedance
                Z_ab = self.calculate_box_impedance_ZAB(frequencies[i], self.lz, self.lsp.a, R_0, self.lx, self.ly, self.truncation_limit)

                # calculate the leakage resistance

                # compliance of the air in the box
                CAA = (self.Va*10**(-3))/(1.4*self.P_0)
                # compliance of the air in the lining material
                CAM = (self.Vm*10**(-3))/self.P_0
                # apparent compliance of the air in the box
                Cab = CAA + 1.4 * CAM
                Ral = 7 / (2 * np.pi * self.fb * Cab)
                # print(Ral)

                kv = np.sqrt((-1j * np.pi * 2 * frequencies[i] * R_0)
                             / self.m)
                # ap = np.sqrt((self.Sp) / np.pi)
                # kvap = kv * ap

                # ξ = np.sqrt(1 - 1j*(2*jv(1, kvap))/(kvap*jv(0, kvap)))
                ξ = 0.998 + 0.001j
                kp = (2 * np.pi * frequencies[i] * ξ) / SOUND_CELERITY
                Zp = (R_0 * SOUND_CELERITY * ξ) / (self.Sp)

                # calculate the port impedance
                R_s = R_0 * SOUND_CELERITY * k**2 * self.a_p**2
                X_s = (R_0 * SOUND_CELERITY * 8 * k * self.a_p)/(3 * np.pi)
                Z_a2 = (R_s + 1j * X_s)

                C = np.array([[1, 1 * Z_e], [0, 1]])
                E = np.array([[0, 1 * self.lsp.Bl], [1 / self.lsp.Bl, 0]])
                D = np.array([[1, 1 * Z_md], [0, 1]])
                M = np.array([[1 * self.lsp.Sd, 0], [0, 1 / self.lsp.Sd]])
                F = np.array([[1, 1 * Z_a1], [0, 1]])
                L = np.array([[1, 0], [1 / Ral, 1]])
                B = np.array([[1, 0], [1 / Z_ab, 1]])
                P = np.array([[np.cos(kp*self.t), 1j*Zp*np.sin(kp*self.t)],
                              [1j*(1/Zp)*np.sin(kp*self.t),
                               np.cos(kp*self.t)]])
                R = np.array([[1, 0], [1/Z_a2, 1]])

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
                UB = (n21 - 1/Z_a2) * p9
                # Up = self.lsp.e_g / (a11 * Z_a2)

                # ig = a21 * p9

                U_ref = ((1 * self.lsp.e_g * self.lsp.Bl * self.lsp.Sd)
                         / (2 * np.pi * frequencies[i] * self.lsp.Mms
                            * self.lsp.Re))

                # response[i] = 20 * np.log10(float(abs(U6))
                #                             / float(abs(U_ref)))

                response[i] = 20 * np.log10(float(abs(UB))
                                            / float(abs(U_ref)))
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
frequencies = np.logspace(np.log2(min_frequency),
                          np.log2(max_frequency),
                          num=num_points, base=2)

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
    "pmax": 0.28,
    "Vmax": 0.03
    }

# Create an instance of the BassReflexEnclosure class
lsp_sys_1 = BassReflexEnclosure(lsp_parameters, 'circular', 'rectangular')
lsp_sys_2 = BassReflexEnclosure(lsp_parameters, 'circular', 'circular')

# Calculate the impedance response
response_1, Ze_1 = lsp_sys_1.calculate_impedance_response(frequencies)
response_2, Ze_2 = lsp_sys_2.calculate_impedance_response(frequencies)

# plot on the same graph the response of the two cases
fig1, ax1 = plt.subplots()
ax1.semilogx(frequencies, response_1,
             label='1 Circ. Loudspeaker with Rect. port')
ax1.semilogx(frequencies, response_2,
             label='1 Circ. Loudspeaker with Circ. port')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Response (dB)')
ax1.set_title('System Response')
ax1.grid(which='both')
ax1.legend()
fig1.show()
fig1.savefig("Ex7.3-2_responses.png")

# plot on the same graph the impedance of the two cases
fig2, ax2 = plt.subplots()
ax2.semilogx(frequencies, Ze_1,
             label='2 Rect. Loudspeakers with Rect. port')
ax2.semilogx(frequencies, Ze_2,
             label='2 Circ. Loudspeakers with Rect. port')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Response (dB)')
ax2.set_title('System Response')
ax2.grid(which='both')
ax2.legend()
fig2.show()
fig2.savefig("Ex7.3-2_impedances.png")
