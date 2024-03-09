# Description: This script calculates the impedance and response of 2 rectangular loudspeakers with a rectangular bass reflex port


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
Sp = 0.0051
Rms = 1 / Qms * np.sqrt(Mms / Cms)

c = 343
porosity = 0.99 
P_0 = 10 ** 5 # atmospheric pressure in N/m^2
u = 0.03  # flow velocity in the material in m/s
m = 1.86 * 10**(-5)  # viscosity coefficient in N.s/m^2
r = 60 * 10 ** (-6) # fiber diameter 
truncation_limit = 10 # Truncation limit for the double summation
d1 = 0.15


y1 = 0.23
y2 = 0.52
lz = 0.192
lx = 0.15  
ly = 0.5  
q = ly / lx
a1 = np.sqrt(Sd) # radius of the diaphragm1
a2 = np.sqrt(Sd) # radius of the diaphragm2
b1 = np.sqrt(Sd) # radius of the diaphragm1
b2 = np.sqrt(Sd) # radius of the diaphragm2
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

# flow resistance of material equation 7.8
R_f = ((4*m*(1- porosity))/(porosity * r**2)) * ((1 - 4/np.pi * (1 - porosity))/(2 + np.log((m*porosity)/(2*r*r_0*u))) + (6/np.pi) * (1 - porosity))
print(R_f)

# 2 diaphragms radiation impedance based on equation 13.339 and 13.345
def calculate_Za1(f, a1, r_0, c, d1, truncation_limit):
    k = (2 * np.pi * f / c) * ((1 + a3 *(R_f/f)**b3) -1j * a4 * (R_f/f)**b4)

    # Calculate the Bessel and Struve functions
    J1 = jv(1, k * a1) 
    H1 = mpmath.struveh(1, k * a1) 

    z11 = z22 = r_0 * c * ((1 - (J1**2/ (k * a1))) + 1j* (H1/ (k * a1))) 

    # Calculate Z12
    z12 = (2 * r_0 * c) / np.sqrt(np.pi)
    sum_mn = 0
    for m in range(truncation_limit+1):
        for n in range(truncation_limit+1):
            term1 = ((k * a1) / (k * d1)) ** m
            term2 = ((k * a1) / (k * d1)) ** n
            term3 = gamma(m + n + 0.5)
            term4 = jv(m + 1, k * a1) * jv(n + 1, k * a1)
            term5 = 1 / (np.math.factorial(m) * np.math.factorial(n))
            term6 = spherical_jn(m + n, k * d1) + 1j * spherical_yn(m + n, k * d1)
            sum_mn += term1 * term2 * term3 * term4 * term5 * term6
    z12 *= sum_mn

    Za1 =  (a1**2 * z11 + 2* a1 * a2 * z12 ) / (a1**2 + a2**2)

    return Za1


# box impedance based on equation 7.131 for rectangular loudspeaker
def calculate_Z11(f, a1, a2, r_0, c, truncation_limit,  y1, y2, lx, ly, lz, b, b1, b2, R_f, P_0, a3, a4, b3, b4):
    
  k = (2 * np.pi * f / c) * ((1 + a3 *(R_f/f)**b3) -1j * a4 * (R_f/f)**b4)
  Zs = r_0 *c + P_0 /( 1j * 2 * np.pi * f * b)

  sum1 = 0
  sum2 = 0
  sum3 = 0

  for n in range(1, truncation_limit + 1):
    k0n = np.sqrt(k**2 - (0*np.pi/lx)**2 - (n*np.pi/ly)**2)
    term1 = (k)/(k0n*n**2) * np.cos(n*np.pi*y1/ly) * np.cos(n*np.pi*y1/ly) * np.sin((n*np.pi*b1)/ (2*ly)) * np.sin((n*np.pi*b1)/(2*ly))
    term2 = ((k0n*Zs)/(k*r_0*c) + 1j * np.tan(k0n*lz)) / (1 + 1j * (( k0n*Zs)/(k*r_0*c)) * np.tan(k0n*lz))
    sum1 += term1 * term2

  for m in range(1, truncation_limit + 1):
    km0 = np.sqrt(k**2 - (2* m*np.pi/lx)**2 - (0*np.pi/ly)**2)
    term1 = (k)/(km0*m**2) * np.sin(m*np.pi*a1/lx) * np.sin(m*np.pi*a1/lx) 
    term2 = ((km0*Zs)/(k*r_0*c) + 1j * np.tan(km0*lz)) / (1 + 1j * (( km0*Zs)/(k*r_0*c)) * np.tan(km0*lz))
    sum2 += term1 * term2  

  for m in range(1, truncation_limit + 1):
    for n in range(1, truncation_limit + 1):
      kmn = np.sqrt(k**2 - (2* m*np.pi/lx)**2 - (n*np.pi/ly)**2)
      term1 = (k)/(kmn*m**2*n**2) * np.sin(m*np.pi*a1/lx) * np.sin(n*np.pi*a1/lx) * np.cos((m*np.pi*y1)/ (ly)) 
      term2 = np.cos(n*np.pi*y1/ly) * np.sin((n*np.pi*b1)/ (2*ly)) * np.sin((n*np.pi*b1)/(2*ly))
      term3 = ((kmn*Zs)/(k*r_0*c) + 1j * np.tan(kmn*lz)) / (1 + 1j * (( kmn*Zs)/(k*r_0*c)) * np.tan(kmn*lz))
  
      sum3 += term1 * term2 * term3

  Z11 = r_0*c *( 1 /(lx*ly) * (((Zs/(r_0*c)) + 1j * np.tan(k*lz)) / (1 + 1j * ((Zs/(r_0*c)) * np.tan(k*lz)))) +  (8*ly)/(np.pi**2 * b1 * b1* lx) * sum1  + 
                               (2*lx)/(np.pi**2 * a1 *a1* ly) *sum2 + (16*lx*ly)/(np.pi**4 * a1 * a1 * b1 * b1) * sum3 ) 
  return  Z11



def calculate_Z12(f, a1, a2, r_0, c, truncation_limit, y1, y2, lx, ly, lz, b, b1, b2, R_f, P_0, a3, a4, b3, b4):
    
  k = (2 * np.pi * f / c) * ((1 + a3 *(R_f/f)**b3) -1j * a4 * (R_f/f)**b4)
  Zs = r_0 *c + P_0 /( 1j * 2 * np.pi * f * b)

  sum1 = 0
  sum2 = 0
  sum3 = 0

  for n in range(1, truncation_limit + 1):
    k0n = np.sqrt(k**2 - (0*np.pi/lx)**2 - (n*np.pi/ly)**2)
    term1 = (k)/(k0n*n**2) * np.cos(n*np.pi*y1/ly) * np.cos(n*np.pi*y2/ly) * np.sin((n*np.pi*b1)/ (2*ly)) * np.sin((n*np.pi*b2)/(2*ly))
    term2 = ((k0n*Zs)/(k*r_0*c) + 1j * np.tan(k0n*lz)) / (1 + 1j * (( k0n*Zs)/(k*r_0*c)) * np.tan(k0n*lz))
    sum1 += term1 * term2

  for m in range(1, truncation_limit + 1):
    km0 = np.sqrt(k**2 - (2*m*np.pi/lx)**2 - (0*np.pi/ly)**2)
    term1 = (k)/(km0*m**2) * np.sin(m*np.pi*a1/lx) * np.sin(m*np.pi*a2/lx) 
    term2 = ((km0*Zs)/(k*r_0*c) + 1j * np.tan(km0*lz)) / (1 + 1j * (( km0*Zs)/(k*r_0*c)) * np.tan(km0*lz))
    sum2 += term1 * term2  

  for m in range(1, truncation_limit + 1):
    for n in range(1, truncation_limit + 1):
      kmn = np.sqrt(k**2 - (2*m*np.pi/lx)**2 - (n*np.pi/ly)**2)
      term1 = (k)/(kmn*m**2*n**2) * np.sin(m*np.pi*a1/lx) * np.sin(n*np.pi*a2/lx) * np.cos((m*np.pi*y1)/ (ly)) 
      term2 = np.cos(n*np.pi*y2/ly) * np.sin((n*np.pi*b1)/ (2*ly)) * np.sin((n*np.pi*b2)/(2*ly))
      term3 = ((kmn*Zs)/(k*r_0*c) + 1j * np.tan(kmn*lz)) / (1 + 1j * (( kmn*Zs)/(k*r_0*c)) * np.tan(kmn*lz))
  
      sum3 += term1 * term2 * term3

  Z12 = r_0*c *( 1 /(lx*ly) * (((Zs/(r_0*c)) + 1j * np.tan(k*lz)) / (1 + 1j * ((Zs/(r_0*c)) * np.tan(k*lz)))) +  (8*ly)/(np.pi**2 * b1 * b2* lx) * sum1  + 
                               (2*lx)/(np.pi**2 * a1 *a1*ly) *sum2 + (16*lx*ly)/(np.pi**4 * a1 * a1 * b1 * b2) * sum3 ) 
  return  Z12


def calculate_Z22(f, a1, a2, r_0, c, truncation_limit, y1,  y2, lx, ly, lz, b, b1, b2,  R_f, P_0, a3, a4, b3, b4):
    
  k = (2 * np.pi * f / c) * ((1 + a3 *(R_f/f)**b3) -1j * a4 * (R_f/f)**b4)
  Zs = r_0 *c + P_0 /( 1j * 2 * np.pi * f * b)

  sum1 = 0
  sum2 = 0
  sum3 = 0

  for n in range(1, truncation_limit + 1):
    k0n = np.sqrt(k**2 - (0*np.pi/lx)**2 - (n*np.pi/ly)**2)
    term1 = (k)/(k0n*n**2) * np.cos(n*np.pi*y2/ly) * np.cos(n*np.pi*y2/ly) * np.sin((n*np.pi*b2)/ (2*ly)) * np.sin((n*np.pi*b2)/(2*ly))
    term2 = ((k0n*Zs)/(k*r_0*c) + 1j * np.tan(k0n*lz)) / (1 + 1j * (( k0n*Zs)/(k*r_0*c)) * np.tan(k0n*lz))
    sum1 += term1 * term2

  for m in range(1, truncation_limit + 1):
    km0 = np.sqrt(k**2 - (2*m*np.pi/lx)**2 - (0*np.pi/ly)**2)
    term1 = (k)/(km0*m**2) * np.sin(m*np.pi*a2/lx) * np.sin(m*np.pi*a2/lx) 
    term2 = ((km0*Zs)/(k*r_0*c) + 1j * np.tan(km0*lz)) / (1 + 1j * (( km0*Zs)/(k*r_0*c)) * np.tan(km0*lz))
    sum2 += term1 * term2  

  for m in range(1, truncation_limit  + 1):
    for n in range(1, truncation_limit + 1):
      kmn = np.sqrt(k**2 - (2*m*np.pi/lx)**2 - (n*np.pi/ly)**2)
      term1 = (k)/(kmn*m**2*n**2) * np.sin(m*np.pi*a2/lx) * np.sin(n*np.pi*a2/lx) * np.cos((m*np.pi*y2)/ (ly)) 
      term2 = np.cos(n*np.pi*y2/ly) * np.sin((n*np.pi*b2)/ (2*ly)) * np.sin((n*np.pi*b2)/(2*ly))
      term3 = ((kmn*Zs)/(k*r_0*c) + 1j * np.tan(kmn*lz)) / (1 + 1j * (( kmn*Zs)/(k*r_0*c)) * np.tan(kmn*lz))
  
      sum3 += term1 * term2 * term3

  Z22 = r_0*c *( 1 /(lx*ly) * (((Zs/(r_0*c)) + 1j * np.tan(k*lz)) / (1 + 1j * ((Zs/(r_0*c)) * np.tan(k*lz)))) +  (8*ly)/(np.pi**2 * b2 * b2* lx) * sum1  + 
                               (2*lx)/(np.pi**2 * a1 *a1*ly) *sum2 + (16*lx*ly)/(np.pi**4 * a1 * a1 * b2 * b2) * sum3 ) 
  return  Z22



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
    
    Z_a1 =  calculate_Za1(frequencies[i], a , r_0, c, d1, truncation_limit)

    Z11 = calculate_Z11(frequencies[i], a1, a1, r_0, c, truncation_limit,  y1, y2, lx, ly, lz, b, b1, b2, R_f, P_0, a3, a4, b3, b4)
    Z12 = Z21 = calculate_Z12(frequencies[i], a1, a_2, r_0, c, truncation_limit,  y1, y2, lx, ly, lz, b, b1, b2, R_f, P_0, a3, a4, b3, b4)
    Z22 = calculate_Z22(frequencies[i], a_2, a_2, r_0, c, truncation_limit, y1, y2, lx, ly, lz, b, b2, b2, R_f, P_0, a3, a4, b3, b4)

    b11 = (Z11 / Z21)
    b12 = ((Z11 * Z22 - Z12 * Z21) / Z21)
    b21 = (1 / Z21)
    b22 = (Z22 / Z21)

    Ral = 168200
    

    kv = np.sqrt((-1j * np.pi *2 * frequencies[i] * r_0) / m)
    ap = np.sqrt((a_2 * b_2) / np.pi)
    kvap = kv * ap
    
    #ξ = np.sqrt(1 - 1j * (2 * jv(1, kvap)) / (kvap * jv(0, kvap)))
    ξ =1
    #print(ξ,  kvap,  jv(1,  kvap), jv(0, kvap))
    kp = (2 * np.pi * frequencies[i]  * ξ) / c
    Zp = (r_0 * c * ξ) / (a_2*b_2)
    
    Kn = lm / a_2

    Bu = (2 * 0.9**(-1) - 1 ) * Kn

    # dynamic density based on equation 4.233
    #r_d = r_0 * (1 - (Q( kv * ap )) / (1 - 0.5 * Bu *kv**2 * ap * Q(kv * ap) )) ** (-1)
  
    r_d = (8*m)/(1j * 2 * np.pi * frequencies[i] * (1 + 4 * Bu) * ap**2)
    

    Za2 =  calculate_Za2(frequencies[i], r_d , lx, ly, r_0, c, truncation_limit)

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

    U_ref = (1/2 * e_g * Bl * Sd) / ( 2 * np.pi * frequencies[i] * Mms * Re)

    #response[i] = 20 * np.log10(float(abs(U6)) / float(abs(U_ref)))

    response[i] = 20 * np.log10(float(abs(Up)) / float(abs(U_ref)))
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


