import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1

# Parameters
f_s = 39
f_c = 63
Bl = 6.59
e_g = 2.83
Cms = 0.00107
Mms = 0.0156
R_e = 6.72
L_e = 0.0005
Sd = 0.012
r_0 = 1.18
c = 343
a = np.sqrt(Sd / np.pi)
Qms = 2.2
Rms = 1 / Qms * np.sqrt(Mms / Cms)
Mmd = Mms - 16 * r_0 * a**3 / 3

lx = 0.15  
ly = 0.32  
L = 0.32 
truncation_limit = 10 # Truncation limit for the double summation

porosity = 0.99  # porosity
P_0 = 10 ** 5
u = 0.03  # flow velocity in the material in m/s
m = 1.86 * 10**(-5)  # viscosity coefficient in N.s/m^2
r = 60 * 10 ** (-6) # fiber diameter 

# Values of coefficients (Table 7.1)
a3 = 0.0858
a4 = 0.175
b3 = 0.7
b4 = 0.59

S1 = np.pi * a**2
S2 = np.pi * a**2
d = 0.07 # the thickness of the lining material

# flow resistance of material equation 7.8
R_f = ((4*m*(1- porosity))/(porosity * r**2)) * ((1 - 4/np.pi * (1 - porosity))/(2 + np.log((m*porosity)/(2*r*r_0*u))) + (6/np.pi) * (1 - porosity))
#print(R_f)

# box impedance based on equation 7.12 
def calculate_ZAB(f, L, a, r_0, c, lx, ly, truncation_limit):
    
    # wave number k equation 7.11
    k = (2 * np.pi * f/ c) * ((1 + a3 *(R_f/f)**b3) - 1j * a4 * (R_f/f)**b4)
    
    Zs = r_0 * c + P_0/( 1j * 2 * np.pi * f * d)
    
    sum_term = 0

    for m in range(truncation_limit + 1):
        for n in range(truncation_limit + 1):
            kmn = np.sqrt(k**2 - (2*m*np.pi/lx)**2 - (2*n*np.pi/ly)**2)
            sum_term = (2 - (m == 0)) * (2 - (n == 0)) / ((kmn * (n**2 + m**2)) + (m == 0) * (n == 0))
            bessel_term = j1(2 * np.pi * a * np.sqrt(m**2 + n**2) / L)
            sum_term *= bessel_term**2 * ((((Zs*kmn)/(k*r_0*c)) + 1j * np.tan(kmn*L/2)) / (1 + 1j *(((Zs*kmn)/(k*r_0*c)) * np.tan(kmn*L/2))))
            
    Z_ab =  r_0 * c * (1 / (L**2) * (((Zs/(r_0*c)) + 1j * np.tan(k*L/2)) / (1 + 1j * ((Zs/(r_0*c)) * np.tan(k*L/2))))  + k / (np.pi**2 * a**2) * sum_term )
    
    return Z_ab


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
    Z_e = R_e + 1j * 2 * np.pi * frequencies[i] * L_e
    Z_md = 1j *  2 * np.pi * frequencies[i] * Mmd + Rms + 1 / (1j *2 * np.pi * frequencies[i] * Cms)
    
    k = (2 * np.pi * frequencies[i] / c) * ((1 + a3 *(R_f/frequencies[i])**b3) - 1j * a4 * (R_f/frequencies[i])**b4)
    
    # diaphragm radiation impedance based on equations 13.116 - 13.118
    R_s = r_0 * c * k**2 * a**2
    X_s = (r_0 * c * 8 * k * a)/(3 * np.pi)
    Z_a1 =  (R_s + 1j * X_s)
    
    # Calculate Z_ab using the function
    Z_ab = calculate_ZAB(frequencies[i], L, a, r_0, c, lx, ly, truncation_limit)

    C = np.array([[1,  (4/2) * Z_e], [0, 1]])
    E = np.array([[0, 1 * Bl], [1 / (1 * Bl), 0]])
    D = np.array([[1, 2 * Z_md], [0, 1]])
    M = np.array([[2* Sd, 0], [0, 1 /(2 * Sd)]])
    F = np.array([[1, 2 * Z_a1], [0, 1]])
    B = np.array([[1, 0], [1 / (2 * Z_ab), 1]])

    A = np.dot(np.dot(np.dot(np.dot(np.dot(C, E), D), M), F), B)

    a11 = A[0, 0]
    a12 = A[0, 1]
    a21 = A[1, 0]
    a22 = A[1, 1]

    U_ref = (e_g * Bl * Sd) / ( 2 * np.pi * frequencies[i] * Mms * R_e)
    p_6 = e_g / a11
    U_c = p_6 / Z_ab
    response[i] = 20 * np.log10(float(abs(U_c)) / float(abs(U_ref)))
    Ze[i] = abs((a11) / (a21))
    

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




