import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from scipy.special import j1, gamma


Re = 6.27  # electrical resistance in ohms
Le = 0.0006  # electrical inductance in H
e_g = 2.83  # source voltage in V
Qes = 0.522  # electrical Q factor
Qms = 1.9  # mechanical Q factor
fs = 37  # resonance frequency in Hz
fb = 36  # tuning frequency in Hz
Sd = 0.012  # diaphragm area in m^2
Vas = 24  # equivalent volume of compliance in m^3
Qts = 0.41  # total Q factor
Vab = 50.32  # volume of the enclosure in m^3
Cms = 0.00119  # mechanical compliance in m/N
Mms = 0.0156  # mechanical mass in kg
Bl = 6.59  # force factor in Tm
Va = 34.31  # volume of the enclosure without the volume of lining material in m^3
Vm = 11.44  # volume of the lining material in m^3
Vb = 45.75  # volume of the box in m^3
Ram = 4480  # acoustic resistance of the material in N.s/m^5
Rab = 492  # acoustic resistance of the box in N.s/m^5
Qa = 49  # acoustic Q factor of the box
Rap = 238  # acoustic resistance of the port in N.s/m^5
Qp = 101  # acoustic Q factor of the port
r_0 = 1.18  # air density in kg/m^3
t = 0.32  # length of the port in m
Sp = 0.0071  # area of the port in m^2
Rms = 1 / Qms * np.sqrt(Mms / Cms)  # mechanical resistance in N.s/m

c = 343  # speed of sound in air in m/s
porosity = 0.99  # porosity
P_0 = 10**5  # atmospheric pressure in N/m^2
u = 0.03  # flow velocity in the material in m/s
m = 1.86 * 10 ** (-5)  # viscosity coefficient in N.s/m^2
r = 80 * 10 ** (-6)  # fiber diameter
truncation_limit = 10  # Truncation limit for the double summation


x1 = 0.1  # distance from the center of the diaphragm to the box wall
x2 = 0.1  # distance from the center of the diaphragm to the box wall
x3 = 0.1  # distance from the center of the diaphragm to the box wall
x4 = 0.1  # distance from the center of the diaphragm to the box wall
y1 = 0.1  # distance from the center of the diaphragm to the box wall
y2 = 0.32  # distance from the center of the diaphragm to the box wall
y3 = 0.50  # distance from the center of the diaphragm to the box wall
y4 = 0.65  # distance from the center of the diaphragm to the box wall
yp = 0.8  # distance from the center of the port to the box wall
lz = 0.192  # length of the box
lx = 0.2  # width of the box
ly = 0.894  # height of the box
q = ly / lx  # aspect ratio of the box
a = np.sqrt(Sd / np.pi)  # radius of the diaphragm
Mmd = Mms - 16 * r_0 * a**3 / 3  # dynamic mass of the diaphragm
b = 0.064  # Acoustical lining thick
lm = 6 * 10 ** (-8)  # dynamic density of the lining material

# Values of coefficients (Table 7.1)
a3 = 0.0858
a4 = 0.175
b3 = 0.7
b4 = 0.59
a_2 = 0.2
b_2 = 0.035
ap = np.sqrt(Sp / np.pi)  # radius of the port


# calculate the flow resistance of the material based on equation 7.8
R_f = ((4 * m * (1 - porosity)) / (porosity * r**2)) * (
    (1 - 4 / np.pi * (1 - porosity)) / (2 + np.log((m * porosity) / (2 * r * r_0 * u)))
    + (6 / np.pi) * (1 - porosity)
)


def calculate_box_impedance_for_circular_piston_Zxy(f, a1, a2, x1, x2, y1, y2, S1, S2):
    "Calculate the impedance of the box based on Eq. 7.131."

    # wave number k equation 7.11
    k = (2 * np.pi * f / c) * ((1 + a3 * (R_f / f) ** b3) - 1j * a4 * (R_f / f) ** b4)

    Zs = r_0 * c + P_0 / (1j * 2 * np.pi * f * b)

    sum_mn = 0
    for m in range(truncation_limit + 1):
        for n in range(truncation_limit + 1):
            kmn = np.sqrt(k**2 - (m * np.pi / lx) ** 2 - (n * np.pi / ly) ** 2)
            delta_m0 = 1 if m == 0 else 0
            delta_n0 = 1 if n == 0 else 0
            term1 = ((kmn * Zs) / (k * r_0 * c) + 1j * np.tan(kmn * lz)) / (
                1 + 1j * ((kmn * Zs) / (k * r_0 * c)) * np.tan(kmn * lz)
            )
            term2 = (
                (2 - delta_m0)
                * (2 - delta_n0)
                / (kmn * (n**2 * lx**2 + m**2 * ly**2) + delta_m0 * delta_n0)
            )
            term3 = (
                np.cos((m * np.pi * x1) / lx)
                * np.cos((n * np.pi * y1) / ly)
                * j1((np.pi * a1 * np.sqrt(n**2 * lx**2 + m**2 * ly**2)) / (lx * ly))
            )
            term4 = (
                np.cos((m * np.pi * x2) / lx)
                * np.cos((n * np.pi * y2) / ly)
                * j1((np.pi * a2 * np.sqrt(n**2 * lx**2 + m**2 * ly**2)) / (lx * ly))
            )
            sum_mn += term1 * term2 * term3 * term4

    Zxy = (
        r_0
        * c
        * (
            (S1 * S2)
            / (lx * ly)
            * (
                ((Zs / (r_0 * c)) + 1j * np.tan(k * lz))
                / (1 + 1j * ((Zs / (r_0 * c)) * np.tan(k * lz)))
            )
            + 4 * k * a1 * a2 * lx * ly * sum_mn
        )
    ) / (S1*S2)

    return Zxy


# calculate the rectangular box impedance based on equation 13.336, 13.327
def calculate_Za2(f, r_d, lx, ly, r_0, c, N):
    "Calculate the impedance of rectangular port based on Eq. 13.336, 13.327."
    k = (2 * np.pi * f / c) * ((1 + a3 * (R_f / f) ** b3) - 1j * a4 * (R_f / f) ** b4)

    Rs_a2 = (r_0 * c) / (np.sqrt(np.pi))

    sum_Rs = 0
    sum_Xs = 0

    for m in range(N + 1):
        for n in range(N + 1):
            term1 = (-1) ** (m + n)
            term2 = (
                (2 * m + 1)
                * (2 * n + 1)
                * np.math.factorial(m + 1)
                * np.math.factorial(n + 1)
                * gamma(m + n + 3 / 2)
            )
            term3 = (k * lx / 2) ** (2 * m + 1)
            term4 = (k * ly / 2) ** (2 * n + 1)
            sum_Rs += (term1 / term2) * term3 * term4

    Rs_a2 *= sum_Rs

    for m in range(N + 1):
        term1 = (-1) ** m * fm(q, m, n)
        term2 = (2 * m + 1) * np.math.factorial(m) * np.math.factorial(m + 1)
        term3 = (k * lx / 2) ** (2 * m + 1)
        sum_Xs += (term1 / term2) * term3

    Xs_a2 = ((2 * r_d * c) / (np.sqrt(np.pi))) * (
        (1 - np.sinc(k * lx)) / (q * k * lx)
        + (1 - np.sinc(q * k * lx)) / (k * lx)
        + sum_Xs
    )

    Z_a2 = (Rs_a2 + 1j * Xs_a2) / (a_2 * b_2)

    return Z_a2


def fm(q, m, n):
    "Helper function for the calculation of the rectangular port impedance."
    result1 = scipy.special.hyp2f1(1, m + 0.5, m + 1.5, 1 / (1 + q**2))
    result2 = scipy.special.hyp2f1(1, m + 0.5, m + 1.5, 1 / (1 + q ** (-2)))

    sum_fm = 0
    for n in range(m + 1):
        sum_fm = gmn(m, n, q)

    return (result1 + result2) / ((2 * m + 1) * (1 + q ** (-2)) ** (m + 0.5)) + (
        1 / (2 * m + 3)
    ) * sum_fm


def gmn(m, n, q):
    "Helper function for the calculation of the rectangular port impedance."
    first_sum = 0

    for p in range(n, m + 1):
        first_sum += (
            scipy.special.binom((m - n), p - n) * (-1) ** (p - n) * (q) ** (2 * n - 1)
        ) / ((2 * p - 1) * (1 + q**2) ** (p - 1 / 2))

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

# Calculate simplified box impedance based on equations 7.2-7.7
def calculate_box_impedance_Zab(f):
    Mab = 0.4*r_0/(np.pi * a)

    CAA = (Va*10**(-3))/(1.4*P_0)
    CAM = (Vm*10**(-3))/P_0
    
    Xab = 2*np.pi*f*Mab - 1 /(2*np.pi*f*(CAA + CAM))
    
    Rab = 2386/ ((1+ Va/(1.4*Vm))**2 + (2*np.pi*f)**2 *2386**2 * CAA**2)
    
    Zab = (Rab + 1j*Xab)
    
    return Zab


# Define a range of frequencies
octave_steps = 24
frequencies_per_octave = octave_steps * np.log(2)
min_frequency = 10
max_frequency = 10000
num_points = int(octave_steps * np.log2(max_frequency / min_frequency)) + 1
frequencies = np.logspace(
    np.log2(min_frequency), np.log2(max_frequency), num=num_points, base=2
)

# Preallocate memory
response = np.zeros_like(frequencies)
Ze = np.zeros_like(frequencies)

for i in range(len(frequencies)):
    # Calculate the electrical impedance
    Z_e = Re + 1j * 2 * np.pi * frequencies[i] * Le

    # Calculate the mechanical impedance
    Z_md = (
        1j * 2 * np.pi * frequencies[i] * Mmd
        + Rms
        + 1 / (1j * 2 * np.pi * frequencies[i] * Cms)
    )

    # Calculate the wave number
    k = (2 * np.pi * frequencies[i] / c) * (
        (1 + a3 * (R_f / frequencies[i]) ** b3) - 1j * a4 * (R_f / frequencies[i]) ** b4
    )

    # Calculate the diaphragm radiation impedance
    R_s = r_0 * c * k**2 * a**2
    X_s = (r_0 * c * 8 * k * a) / (3 * np.pi)
    Z_a1 = (R_s + 1j * X_s) 
    
    # Calculate the simplified box impedance
    Zab = calculate_box_impedance_Zab(frequencies[i])

    # Calculate the box impedance 
    Z11 = calculate_box_impedance_for_circular_piston_Zxy(
        frequencies[i], a, a, x1, x1, y1, y1, Sd, Sd
    )
    Z22 = calculate_box_impedance_for_circular_piston_Zxy(
        frequencies[i], ap, ap, x1, x1, yp, yp, Sp, Sp
    )
    Z12 = calculate_box_impedance_for_circular_piston_Zxy(
        frequencies[i], a, ap, x1, x1, y1, yp, Sd, Sp
    )
    Z21 = Z12

    # # 2-port network parameters
    b11 = Z11 / Z21
    b12 = (Z11 * Z22 - Z12 * Z21) / Z21
    b21 = 1 / Z21
    b22 = Z22 / Z21

    # leak resistance
    CAA = (Va * 10 ** (-3)) / (1.4 * P_0)
    CAM = (Vm * 10 ** (-3)) / P_0
    Cab = CAA + 1.4 * CAM
    Ral = 7 / (2 * np.pi * fb * Cab)

    kv = np.sqrt((-1j * np.pi * 2 * frequencies[i] * r_0) / m)
    ap = np.sqrt((a_2 * b_2) / np.pi)
    kvap = kv * ap

    ξ = 0.998 + 0.001j

    kp = (2 * np.pi * frequencies[i] * ξ) / c
    Zp = (r_0 * c * ξ) / (a_2 * b_2)

    Kn = lm / a_2

    Bu = (2 * 0.9 ** (-1) - 1) * Kn

    # dynamic density based on equation 4.233
    r_d = (-8 * r_0) / ((1 + 4 * Bu) * kv**2 * ap**2)
    # r_d = r_0 * (1 - (Q( kv * ap )) / (1 - 0.5 * Bu *kv**2 * ap**2 * Q(kv * ap) )) ** (-1)
    # r_d = (8*m)/(1j * 2 * np.pi * frequencies[i] * (1 + 4 * Bu) * a_2*b_2)

    # Calculate the impedance of a rectangular port
    Za2 = calculate_Za2(frequencies[i], r_d, lx, ly, r_0, c, truncation_limit)

    C = np.array([[1, 1 / 4 * Z_e], [0, 1]])
    E = np.array([[0, 1 * Bl], [1 / (1 * Bl), 0]])
    D = np.array([[1, 4 * Z_md], [0, 1]])
    M = np.array([[4 * Sd, 0], [0, 1 / (4 * Sd)]])
    F = np.array([[1, 4*Z_a1], [0, 1]])
    L = np.array([[1, 0], [1 / Ral, 1]])
    #B = np.array([[1, 0], [1 / Zab, 1]])
    B = np.array([[b11, b12], [b21, b22]])
    P = np.array(
        [
            [np.cos(kp * t), 1j * Zp * np.sin(kp * t)],
            [1j * (1 / Zp) * np.sin(kp * t), np.cos(kp * t)],
        ]
    )
    R = np.array([[1, 0], [1 / Za2, 1]])

    A = C @ E @ D @ M @ F @ L @ B @ P @ R

    a11 = A[0, 0]
    a12 = A[0, 1]
    a21 = A[1, 0]
    a22 = A[1, 1]

    p9 = e_g / a11
    Up = p9 / Za2

    N = np.dot(np.dot(B, P), R)

    n11 = N[0, 0]
    n12 = N[0, 1]
    n21 = N[1, 0]
    n22 = N[1, 1]

    U6 = n21 * p9
    UB = (n21 - 1 / Za2) * p9
    Up = e_g / (a11 * Za2)

    ig = a21 * p9

    U_ref = (4 * e_g * Bl * Sd) / (2 * np.pi * frequencies[i] * Mms * Re)

    # response[i] = 20 * np.log10(float(abs(U6)) / float(abs(U_ref)))

    response[i] = 20 * np.log10(float(abs(UB)) / float(abs(U_ref)))
    Ze[i] = abs(((a11) / (a21)))


# Plotting the response
plt.figure()
plt.semilogx(frequencies, response)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Response (dB)")
plt.title("System Response")
plt.grid(which="both")
plt.show()


# Plotting the impedance
plt.figure()
plt.semilogx(frequencies, Ze)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Response (dB)")
plt.title("System Impedance")
plt.grid(which="both")
plt.show()
