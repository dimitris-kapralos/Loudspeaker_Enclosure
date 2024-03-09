# Description: This script calculates the impedance and response of 2 circular loudspeakers with a circular bass reflex port


import numpy as np
import matplotlib.pyplot as plt
import mpmath
import scipy.special
from scipy.special import  j1, jv, gamma, spherical_jn, spherical_yn

# Parameters
Re = 6.27
Le = 0.0006
e_g = 2.83
Qes = 0.522
Qms = 1.9
fs = 37
Sd = 0.012
Vas = 24
Qts = 0.41
Vab = 25.2
Cms = 0.00119
Mms = 0.0156
Bl = 6.59
t = 0.411
Va = 17.2
Ram = 4480
Rab = 492
Qa = 49
Rap = 238
Qp = 101
r_0 = 1.18
Sp = 0.005
Rms = 1 / Qms * np.sqrt(Mms / Cms)

c = 343
porosity = 0.99 
P_0 = 10 ** 5 # atmospheric pressure in N/m^2
u = 0.03  # flow velocity in the material in m/s
m = 1.86 * 10**(-5)  # viscosity coefficient in N.s/m^2
r = 40 * 10 ** (-6) # fiber diameter 
truncation_limit = 10 # Truncation limit for the double summation
d1 = 0.15

x1 = x2 = 0.075
y1 = 0.095
y2 = 0.325
lz = 0.192
lx = 0.15  
ly = 0.635  
q = ly / lx
a1 = np.sqrt(Sd) # side of the diaphragm1
a2 = np.sqrt(Sd) # side of the diaphragm2
b1 = np.sqrt(Sd) # side of the diaphragm1
b2 = np.sqrt(Sd) # side of the diaphragm2
a = np.sqrt(Sd / np.pi)
Mmd = Mms - 16 * r_0 * a**3 / 3
b = 0.064 # Acoustical lining thick
lm = 6 * 10 ** (-8)

# Values of coefficients (Table 7.1)
a3 = 0.0858
a4 = 0.175
b3 = 0.7
b4 = 0.59
a_2 = 0.15
b_2 = 0.034

S1 = np.pi * a**2
S2 = np.pi * a**2
a_p = np.sqrt(Sp / np.pi)


# flow resistance of material equation 7.8
R_f = ((4*m*(1- porosity))/(porosity * r**2)) * ((1 - 4/np.pi * (1 - porosity))/(2 + np.log((m*porosity)/(2*r*r_0*u))) + (6/np.pi) * (1 - porosity))
print(R_f)

# 2 diaphragms radiation impedance based on equation 13.339 and 13.345
def calculate_Za1(f, a, r_0, c, d1, truncation_limit):
    k = (2 * np.pi * f / c) * ((1 + a3 *(R_f/f)**b3) -1j * a4 * (R_f/f)**b4)

    # Calculate the Bessel and Struve functions
    J1 = jv(1, k * a) 
    H1 = mpmath.struveh(1, k * a) 

    z11 = z22 = r_0 * c * ((1 - (J1**2/ (k * a))) + 1j* (H1/ (k * a))) 

    # Calculate Z12
    z12 = (2 * r_0 * c) / np.sqrt(np.pi)
    sum_mn = 0
    for m in range(truncation_limit+1):
        for n in range(truncation_limit+1):
            term1 = ((k * a) / (k * d1)) ** m
            term2 = ((k * a) / (k * d1)) ** n
            term3 = gamma(m + n + 0.5)
            term4 = jv(m + 1, k * a) * jv(n + 1, k * a)
            term5 = 1 / (np.math.factorial(m) * np.math.factorial(n))
            term6 = spherical_jn(m + n, k * d1) + 1j * spherical_yn(m + n, k * d1)
            sum_mn += term1 * term2 * term3 * term4 * term5 * term6
    z12 *= sum_mn

    Za1 =  (a**2 * z11 + 2* a * a * z12 ) / (a**2 + a**2)

    return Za1


# box impedance based on equation 7.131 for rectangular loudspeaker
# box impedance based on equation 7.131 
def calculate_Z11(f, lz, a, r_0, c, lx, ly, truncation_limit):

    # wave number k equation 7.11
    k = (2 * np.pi * f / c) * ((1 + a3 *(R_f/f)**b3) -1j * a4 * (R_f/f)**b4)

    Zs = r_0 * c + P_0/( 1j * 2 * np.pi * f * b)
    
    sum_mn = 0
    for m in range(truncation_limit+1):
        for n in range(truncation_limit+1):
            kmn = np.sqrt(k**2 - (m*np.pi/lx)**2 - (n*np.pi/ly)**2)
            delta_m0 = 1 if m == 0 else 0
            delta_n0 = 1 if n == 0 else 0
            term1 = ((kmn*Zs)/(k*r_0*c) + 1j * np.tan(kmn*lz)) / (1 + 1j * (( kmn*Zs)/(k*r_0*c)) * np.tan(kmn*lz))
            term2 = (2 - delta_m0) * (2 - delta_n0) / (kmn * (n**2 * lx**2 + m**2 * ly**2) + delta_m0 * delta_n0)
            term3 = np.cos((m*np.pi*x1)/lx) * np.cos((n*np.pi*y1)/ly) * j1((np.pi * a * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
            term4 = np.cos((m*np.pi*x1)/lx) * np.cos((n*np.pi*y1)/ly) * j1((np.pi * a * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
            sum_mn += term1 * term2 * term3 * term4 

    Z11 = (r_0 * c * ( (S1*S1)/(lx*ly) * (((Zs/(r_0*c)) + 1j * np.tan(k*lz)) / (1 + 1j * ((Zs/(r_0*c)) * np.tan(k*lz)))) + 4 * k*a*a*lx*ly* sum_mn)) / Sd**2   
    
    return  Z11



# box impedance based on equation 7.131 
def calculate_Z22(f, lz, a, r_0, c, lx, ly, truncation_limit):

    # wave number k equation 7.11
    k = (2 * np.pi * f / c) * ((1 + a3 *(R_f/f)**b3) -1j * a4 * (R_f/f)**b4)

    Zs = r_0 * c + P_0/( 1j * 2 * np.pi * f * b)
    
    sum_mn = 0
    for m in range(truncation_limit+1):
        for n in range(truncation_limit+1):
            kmn = np.sqrt(k**2 - (m*np.pi/lx)**2 - (n*np.pi/ly)**2)
            delta_m0 = 1 if m == 0 else 0
            delta_n0 = 1 if n == 0 else 0
            term1 = ((kmn*Zs)/(k*r_0*c) + 1j * np.tan(kmn*lz)) / (1 + 1j * (( kmn*Zs)/(k*r_0*c)) * np.tan(kmn*lz))
            term2 = (2 - delta_m0) * (2 - delta_n0) / (kmn * (n**2 * lx**2 + m**2 * ly**2) + delta_m0 * delta_n0)
            term3 = np.cos((m*np.pi*x2)/lx) * np.cos((n*np.pi*y2)/ly) * j1((np.pi * a * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
            term4 = np.cos((m*np.pi*x2)/lx) * np.cos((n*np.pi*y2)/ly) * j1((np.pi * a * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
            sum_mn += term1 * term2 * term3 * term4 

    Z22 = (r_0 * c * ( (S2*S2)/(lx*ly) * (((Zs/(r_0*c)) + 1j * np.tan(k*lz)) / (1 + 1j * ((Zs/(r_0*c)) * np.tan(k*lz)))) + 4 * k*a*a*lx*ly* sum_mn)) / Sd**2   
    
    return  Z22


# box impedance based on equation 7.131 
def calculate_Z12(f, lz, a, r_0, c, lx, ly, truncation_limit):

    # wave number k equation 7.11
    k = (2 * np.pi * f / c) * ((1 + a3 *(R_f/f)**b3) -1j * a4 * (R_f/f)**b4)

    Zs = r_0 * c + P_0/( 1j * 2 * np.pi * f * b)
    
    sum_mn = 0
    for m in range(truncation_limit+1):
        for n in range(truncation_limit+1):
            kmn = np.sqrt(k**2 - (m*np.pi/lx)**2 - (n*np.pi/ly)**2)
            delta_m0 = 1 if m == 0 else 0
            delta_n0 = 1 if n == 0 else 0
            term1 = ((kmn*Zs)/(k*r_0*c) + 1j * np.tan(kmn*lz)) / (1 + 1j * (( kmn*Zs)/(k*r_0*c)) * np.tan(kmn*lz))
            term2 = (2 - delta_m0) * (2 - delta_n0) / (kmn * (n**2 * lx**2 + m**2 * ly**2) + delta_m0 * delta_n0)
            term3 = np.cos((m*np.pi*x1)/lx) * np.cos((n*np.pi*y1)/ly) * j1((np.pi * a * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
            term4 = np.cos((m*np.pi*x2)/lx) * np.cos((n*np.pi*y2)/ly) * j1((np.pi * a * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
            sum_mn += term1 * term2 * term3 * term4 

    Z12 = (r_0 * c * ( (S1*S2)/(lx*ly) * (((Zs/(r_0*c)) + 1j * np.tan(k*lz)) / (1 + 1j * ((Zs/(r_0*c)) * np.tan(k*lz)))) + 4 * k*a*a*lx*ly* sum_mn)) / Sd**2   
    
    return  Z12





# Define a range of frequencies
octave_steps = 24
frequencies_per_octave = octave_steps * np.log(2)
min_frequency = 10
max_frequency = 10000
num_points = int(octave_steps * np.log2(max_frequency / min_frequency)) + 1
frequencies = np.logspace(np.log2(min_frequency), np.log2(max_frequency), num=num_points, base=2)

# Preallocate memory 
response = np.zeros_like(frequencies)
Ze = np.zeros_like(frequencies)

for i in range(len(frequencies)):
   
    Z_e = Re + 1j * 2 * np.pi * frequencies[i] * Le
    Z_md = 1j *  2 * np.pi * frequencies[i]  * Mmd + Rms + 1 / (1j * 2 * np.pi * frequencies[i] * Cms)

    k = (2 * np.pi * frequencies[i] / c) * ((1 + a3 *(R_f/frequencies[i])**b3) - 1j * a4 * (R_f/frequencies[i])**b4)
    
    Z_a1 =  calculate_Za1(frequencies[i], a , r_0, c, d1, truncation_limit)

    Z11 = calculate_Z11(frequencies[i], lz, a, r_0, c, lx, ly, truncation_limit)
    Z12 = Z21 = calculate_Z12(frequencies[i], lz, a, r_0, c, lx, ly, truncation_limit)
    Z22 = calculate_Z22(frequencies[i], lz, a, r_0, c, lx, ly, truncation_limit)

    b11 = (Z11 / Z21)
    b12 = ((Z11 * Z22 - Z12 * Z21) / Z21)
    b21 = (1 / Z21)
    b22 = (Z22 / Z21)

    Ral = 168200
    

    kv = np.sqrt((-1j * np.pi *2 * frequencies[i] * r_0) / m)
    ap = np.sqrt((Sp) / np.pi)
    ξ = 1
    kp = (2 * np.pi * frequencies[i]  * ξ) / c
    Zp = (r_0 * c * ξ) / (Sp)
    
 
    
    # Calculate the impedance of the port
    R_s = r_0 * c * (1- jv(1, k*a_p)/ (k*a_p)) 
    X_s = r_0 * c * (1 +  mpmath.struveh(1, k * a_p) / (k*a_p))
    Za2 =  (R_s + 1j * X_s)

    C = np.array([[1,  1/2 * Z_e], [0, 1]])
    E = np.array([[0, 1*Bl], [1 /  Bl, 0]])
    D = np.array([[1, 2 * Z_md], [0, 1]])
    M = np.array([[2 * Sd, 0], [0,  2 /  Sd]])
    F = np.array([[1, Z_a1], [0, 1]])
    L = np.array([[1, 0], [1/ Ral, 1]])
    B = np.array([[b11, b12], [b21, b22]])
    P = np.array([[np.cos(kp*t), 1j*Zp*np.sin(kp*t)], [1j*(1/Zp)*np.sin(kp*t), np.cos(kp*t)]])
    R = np.array([[1, 0], [1/Za2, 1]])

    A = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(C, E), D), M), F), L), B), P) , R)

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
    UB = (n21 - 1/Za2) * p9
    Up = e_g / (a11 * Za2)

    ig = a21 * p9

    U_ref = (2 * e_g * Bl * Sd) / ( 2 * np.pi * frequencies[i] * Mms * Re)

    #response[i] = 20 * np.log10(float(abs(U6)) / float(abs(U_ref)))

    response[i] = 20 * np.log10(float(abs(UB)) / float(abs(U_ref)))
    Ze[i] = abs(((a11) / (a21)))


# Plotting the response
plt.figure()
plt.semilogx( frequencies, response)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Response (dB)')
plt.title('System Response')
plt.grid(which='both')
plt.show()


# Plotting the impedance
plt.figure()
plt.semilogx( frequencies, Ze)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Response (dB)')
plt.title('System Impedance')
plt.grid(which='both')
plt.show()
