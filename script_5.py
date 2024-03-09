# Description: This script calculates the impedance and response of a 1 circular loudspeaker with a rectangular bass reflex port


import numpy as np
import matplotlib.pyplot as plt
import mpmath
import scipy.special
from scipy.special import  j1, jv, gamma, spherical_jn, spherical_yn, j0

# Parameters
Re = 6.72
Le = 0.0005
e_g = 2.83
Qes = 0.522
Qms = 1.9
fs = 37
Sd = 0.012
Vas = 24
Qts = 0.41
Vab = 37.75
Cms = 0.00119
Mms = 0.0156
Bl = 6.59
t = 0.411
Va = 25.74
Ram = 4480
Rab = 492
Qa = 49
Rap = 238
Qp = 101
r_0 = 1.18
Sp = 0.0061
a_2 = np.sqrt(Sp / np.pi)
Rms = 1 / Qms * np.sqrt(Mms / Cms)

c = 343
porosity = 0.99  # porosity
P_0 = 10 ** 5
u = 0.03  # flow velocity in the material in m/s
m = 1.86 * 10**(-5)  # viscosity coefficient in N.s/m^2
r = 50 * 10 ** (-6) # fiber diameter 
truncation_limit = 10 # Truncation limit for the double summation
d1 = 0.15

x1 = x2 = 0.1
y1 = 0.23
y2 = 0.501
lz = 0.282
lx = 0.2  
ly = 0.670  
q = ly / lx
a = np.sqrt(Sd / np.pi)
a1 = a
a2 = a
b1 = 0.15
a_2 = 0.15
b_2 = 0.034
Mmd = Mms - 16 * r_0 * a**3 / 3
b2 = 0.15
b = 0.064
lm = 6 * 10 ** (-8)

# Values of coefficients (Table 7.1)
a3 = 0.0858
a4 = 0.175
b3 = 0.7
b4 = 0.59

S1 = S2 = Sd

R_f = ((4*m*(1- porosity))/(porosity * r**2)) * ((1 - 4/np.pi * (1 - porosity))/(2 + np.log((m*porosity)/(2*r*r_0*u))) + (6/np.pi) * (1 - porosity))
#print(R_f)

# box impedance based on equation 7.131
def calculate_ZAB(f, lz, a, r_0, c, lx, ly, truncation_limit):

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

    ZAB = (r_0 * c * ( (Sd*Sd)/(lx*ly) * (((Zs/(r_0*c)) + 1j * np.tan(k*lz)) / (1 + 1j * ((Zs/(r_0*c)) * np.tan(k*lz)))) + 4 * k*a*a*lx*ly* sum_mn)) / (Sd*Sd)   
    
    return  ZAB





# box impedance based on equation 7.131 
# def calculate_Z11(f, lz, a, a_2, r_0, c, lx, ly, truncation_limit):

#     # wave number k equation 7.11
#     k = (2 * np.pi * f / c) * ((1 + a3 *(R_f/f)**b3) -1j * a4 * (R_f/f)**b4)

#     Zs = r_0 * c + P_0/( 1j * 2 * np.pi * f * b)
    
#     sum_mn = 0
#     for m in range(truncation_limit+1):
#         for n in range(truncation_limit+1):
#             kmn = np.sqrt(k**2 - (m*np.pi/lx)**2 - (n*np.pi/ly)**2)
#             delta_m0 = 1 if m == 0 else 0
#             delta_n0 = 1 if n == 0 else 0
#             term1 = ((kmn*Zs)/(k*r_0*c) + 1j * np.tan(kmn*lz)) / (1 + 1j * (( kmn*Zs)/(k*r_0*c)) * np.tan(kmn*lz))
#             term2 = (2 - delta_m0) * (2 - delta_n0) / (kmn * (n**2 * lx**2 + m**2 * ly**2) + delta_m0 * delta_n0)
#             term3 = np.cos((m*np.pi*x1)/lx) * np.cos((n*np.pi*y1)/ly) * j1((np.pi * a * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
#             term4 = np.cos((m*np.pi*x1)/lx) * np.cos((n*np.pi*y1)/ly) * j1((np.pi * a * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
#             sum_mn += term1 * term2 * term3 * term4 

#     Z11 = (r_0 * c * ( (Sd*Sd)/(lx*ly) * (((Zs/(r_0*c)) + 1j * np.tan(k*lz)) / (1 + 1j * ((Zs/(r_0*c)) * np.tan(k*lz)))) + 4 * k*a*a*lx*ly* sum_mn)) / Sd**2   
    
#     return  Z11



# # box impedance based on equation 7.131 
# def calculate_Z22(f, lz, a, a_2, r_0, c, lx, ly, truncation_limit):

#     # wave number k equation 7.11
#     k = (2 * np.pi * f / c) * ((1 + a3 *(R_f/f)**b3) -1j * a4 * (R_f/f)**b4)

#     Zs = r_0 * c + P_0/( 1j * 2 * np.pi * f * b)
    
#     sum_mn = 0
#     for m in range(truncation_limit+1):
#         for n in range(truncation_limit+1):
#             kmn = np.sqrt(k**2 - (m*np.pi/lx)**2 - (n*np.pi/ly)**2)
#             delta_m0 = 1 if m == 0 else 0
#             delta_n0 = 1 if n == 0 else 0
#             term1 = ((kmn*Zs)/(k*r_0*c) + 1j * np.tan(kmn*lz)) / (1 + 1j * (( kmn*Zs)/(k*r_0*c)) * np.tan(kmn*lz))
#             term2 = (2 - delta_m0) * (2 - delta_n0) / (kmn * (n**2 * lx**2 + m**2 * ly**2) + delta_m0 * delta_n0)
#             term3 = np.cos((m*np.pi*x2)/lx) * np.cos((n*np.pi*y2)/ly) * j1((np.pi * a_2 * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
#             term4 = np.cos((m*np.pi*x2)/lx) * np.cos((n*np.pi*y2)/ly) * j1((np.pi * a_2 * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
#             sum_mn += term1 * term2 * term3 * term4 

#     Z22 = (r_0 * c * ( (Sp*Sp)/(lx*ly) * (((Zs/(r_0*c)) + 1j * np.tan(k*lz)) / (1 + 1j * ((Zs/(r_0*c)) * np.tan(k*lz)))) + 4 * k*a_2*a_2*lx*ly* sum_mn)) / Sp**2   
    
#     return  Z22


# # box impedance based on equation 7.131 
# def calculate_Z12(f, lz, a,a_2, r_0, c, lx, ly, truncation_limit):

#     # wave number k equation 7.11
#     k = (2 * np.pi * f / c) * ((1 + a3 *(R_f/f)**b3) -1j * a4 * (R_f/f)**b4)

#     Zs = r_0 * c + P_0/( 1j * 2 * np.pi * f * b)
    
#     sum_mn = 0
#     for m in range(truncation_limit+1):
#         for n in range(truncation_limit+1):
#             kmn = np.sqrt(k**2 - (m*np.pi/lx)**2 - (n*np.pi/ly)**2)
#             delta_m0 = 1 if m == 0 else 0
#             delta_n0 = 1 if n == 0 else 0
#             term1 = ((kmn*Zs)/(k*r_0*c) + 1j * np.tan(kmn*lz)) / (1 + 1j * (( kmn*Zs)/(k*r_0*c)) * np.tan(kmn*lz))
#             term2 = (2 - delta_m0) * (2 - delta_n0) / (kmn * (n**2 * lx**2 + m**2 * ly**2) + delta_m0 * delta_n0)
#             term3 = np.cos((m*np.pi*x1)/lx) * np.cos((n*np.pi*y1)/ly) * j1((np.pi * a * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
#             term4 = np.cos((m*np.pi*x2)/lx) * np.cos((n*np.pi*y2)/ly) * j1((np.pi * a_2 * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
#             sum_mn += term1 * term2 * term3 * term4 

#     Z12 = (r_0 * c * ( (Sd*Sp)/(lx*ly) * (((Zs/(r_0*c)) + 1j * np.tan(k*lz)) / (1 + 1j * ((Zs/(r_0*c)) * np.tan(k*lz)))) + 4 * k*a*a_2*lx*ly* sum_mn)) / (Sd*Sp)   
    
#     return  Z12


# calculate the rectangular port impedance based on equation 13.336, 13.327
def calculate_Za2 (f, r_d,  lx, ly, r_0, c, N):
    k = (2 * np.pi * f / c) * ((1 + a3 *(R_f/f)**b3) -1j * a4 * (R_f/f)**b4)

    Rs_a2 = (r_0*c) / (np.sqrt(np.pi))

    sum_Rs = 0
    sum_Xs = 0

    for m in range(N+1):
        for n in range(N+1):
            term1 = (-1) ** (m + n)
            term2 = (2*m + 1) * (2*n + 1) * np.math.factorial(m+1) * np.math.factorial(n+1) * gamma(m + n + 3/2)
            term3 = (k*lx/2) **(2*m +1)
            term4 = (k*ly/2) **(2*n +1)
            sum_Rs += (term1 / term2) * term3 * term4

    Rs_a2 *= sum_Rs

    for m in range(N+1):
        term1 = (-1) ** m * fm(q, m, n)
        term2 = (2*m + 1) * np.math.factorial(m) * np.math.factorial(m+1) 
        term3 = (k*lx/2) ** (2*m +1)
        sum_Xs += (term1 / term2) * term3

    Xs_a2 = ((2* r_d *c) / (np.sqrt(np.pi))) * ((1- np.sinc(k*lx))/(q*k*lx) + (1- np.sinc(q* k*lx))/(k*lx) + sum_Xs )



    Z_a2 = (Rs_a2 + 1j * Xs_a2 ) / (a_2 * b_2)
    
    return Z_a2


def fm(q,m,n):

    result1 = scipy.special.hyp2f1(1, m + 0.5, m + 1.5, 1 / (1 + q**2))
    result2 = scipy.special.hyp2f1(1, m + 0.5, m + 1.5, 1 / (1 + q**(-2)))

    sum_fm = 0                                             
    for n in range(m+1):
      sum_fm = gmn(m, n, q)

    return (result1 + result2) / ((2*m+1)*(1+q**(-2))**(m+0.5)) + (1/(2*m + 3)) * sum_fm

def gmn(m, n, q):
    first_sum = 0

    for p in range(n, m + 1):
      first_sum += (scipy.special.binom((m - n), p - n) * (-1) ** (p - n) * (q) ** (2 * n - 1)) / ((2 * p - 1) * (1 + q**2)**((p - 1/2)))

    first_term = scipy.special.binom((2 * m + 3), (2 * n)) * first_sum

    second_sum = 0

    for p in range(m - n, m + 1):
      second_sum += (scipy.special.binom((p - m + n), n) * (-1) ** (p - m + n) * (q) ** (2 * n + 2)) / ((2 * p - 1) * (1 + q**(-2))**((p - 1/2)))

    second_term = scipy.special.binom((2 * m + 3), (2 * n + 3)) * second_sum

    return first_term + second_term




# Define a range of frequencies
octave_steps = 24
frequencies_per_octave = octave_steps * np.log(2)
min_frequency = 10
max_frequency = 1000
num_points = int(octave_steps * np.log2(max_frequency / min_frequency)) + 1
frequencies = np.logspace(np.log2(min_frequency), np.log2(max_frequency), num=num_points, base=2)

# Preallocate memory 
response = np.zeros_like(frequencies)
Ze = np.zeros_like(frequencies)

for i in range(len(frequencies)):
   
    Z_e = Re + 1j * 2 * np.pi * frequencies[i] * Le
    Z_md = 1j *  2 * np.pi * frequencies[i]  * Mmd + Rms + 1 / (1j * 2 * np.pi * frequencies[i] * Cms)

    k = (2 * np.pi * frequencies[i] / c) * ((1 + a3 *(R_f/frequencies[i])**b3) - 1j * a4 * (R_f/frequencies[i])**b4)
    
    R_s = r_0 * c * k**2 * a**2
    X_s = (r_0 * c * 8 * k * a)/(3 * np.pi)
    Z_a1 =  (R_s + 1j * X_s)

    Z_ab =  calculate_ZAB(frequencies[i], lz, a,  r_0, c, lx, ly, truncation_limit)


    # Z11 = calculate_Z11(frequencies[i], lz, a, a, r_0, c, lx, ly, truncation_limit)
    # Z22 = calculate_Z22(frequencies[i], lz, a_2, a_2, r_0, c, lx, ly, truncation_limit)
    # Z12 = calculate_Z12(frequencies[i], lz, a, a_2, r_0, c, lx, ly, truncation_limit)
    # Z21 = Z12

    # b11 = (Z11 / Z21)
    # b12 = ((Z11 * Z22 - Z12 * Z21) / Z21)
    # b21 = (1 / Z21)
    # b22 = (Z22 / Z21)

    Ral = 168200
    

    kv = np.sqrt((-1j * np.pi *2 * frequencies[i] * r_0) / m)
    ap = np.sqrt((Sp) / np.pi)
    ξ = np.sqrt(1 - (2 * jv(1, kv * ap)) / (kv * ap * jv(0, kv * ap)))
    kp = (2 * np.pi * frequencies[i]  * ξ) / c
    Zp = (r_0 * c * ξ) / (Sp)

    Kn = lm / a_2

    Bu = (2 * 0.9**(-1) - 1 ) * Kn

    # dynamic density based on equation 4.233
    #r_d = r_0 * (1 - (Q( kv * ap )) / (1 - 0.5 * Bu *kv**2 * ap * Q(kv * ap) )) ** (-1)
  
    r_d = (8*m)/(1j * 2 * np.pi * frequencies[i] * (1 + 4 * Bu) * ap**2)


    # Calculate the impedance of the port 
    Za2 =  calculate_Za2(frequencies[i], r_d,  lx, ly, r_0, c, truncation_limit)

    C = np.array([[1,  2 /2 * Z_e], [0, 1]])
    E = np.array([[0, 1*Bl], [1 /  Bl, 0]])
    D = np.array([[1,  1 * Z_md], [0, 1]])
    M = np.array([[1 * Sd, 0], [0,  1 /  Sd]])
    F = np.array([[1, Z_a1], [0, 1]])
    L = np.array([[1, 0], [1/ Ral, 1]])
    B = np.array([[1, 0], [1 / Z_ab, 1]])
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

    U_ref = ( 1 * e_g * Bl * Sd) / ( 2 * np.pi * frequencies[i] * Mms * Re)

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



