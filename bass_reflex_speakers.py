import numpy as np
import matplotlib.pyplot as plt
import mpmath
import scipy.special
from scipy.special import  j1, jv, gamma, spherical_jn, spherical_yn



class BassReflexEnclosure:
    def __init__(self, speaker_type, speaker_number, port_shape ) :
        
        self.speaker_type = speaker_type
        self.speaker_number = speaker_number
        self.port_shape = port_shape
        
        
        self.Re = 6.27
        self.Le = 0.0006
        self.e_g = 2.83
        self.Qes = 0.522
        self.Qms = 1.9
        self.fs = 37
        self.Sd = 0.012
        self.Vas = 24
        self.Qts = 0.41
        self.Vab = 25.2
        self.Cms = 0.00119
        self.Mms = 0.0156
        self.Bl = 6.59
        self.t = 0.411
        self.Va = 17.2
        self.Ram = 4480
        self.Rab = 492
        self.Qa = 49
        self.Rap = 238
        self.Qp = 101
        self.r_0 = 1.18
        self.Sp = 0.0051
        self.Rms = 1 / self.Qms * np.sqrt(self.Mms / self.Cms)

        self.c = 343
        self.porosity = 0.99 
        self.P_0 = 10 ** 5 # atmospheric pressure in N/m^2
        self.u = 0.03  # flow velocity in the material in m/s
        self.m = 1.86 * 10**(-5)  # viscosity coefficient in N.s/m^2
        self.r = 60 * 10 ** (-6) # fiber diameter 
        self.truncation_limit = 10 # Truncation limit for the double summation
        self.d1 = 0.15

        self.x1 = 0.075
        self.x2 = 0.075
        self.y1 = 0.23
        self.y2 = 0.501
        self.lz = 0.192
        self.lx = 0.15  
        self.ly = 0.5  
        self.q = self.ly / self.lx
        self.a1 = np.sqrt(self.Sd) # radius of the diaphragm1
        self.a2 = np.sqrt(self.Sd) # radius of the diaphragm2
        self.b1 = np.sqrt(self.Sd) # radius of the diaphragm1
        self.b2 = np.sqrt(self.Sd) # radius of the diaphragm2
        self.a = np.sqrt(self.Sd / np.pi)
        self.Mmd = self.Mms - 16 * self.r_0 * self.a**3 / 3
        self.b = 0.064 # Acoustical lining thick
        self.lm = 6 * 10 ** (-8)

        # Values of coefficients (Table 7.1)
        self.a3 = 0.0858
        self.a4 = 0.175
        self.b3 = 0.7
        self.b4 = 0.59
        self.a_2 = 0.15
        self.b_2 = 0.034
        self.a_p = np.sqrt(self.Sp / np.pi)
        self.S1 = np.pi * self.a**2
        self.S2 = np.pi * self.a**2


    def calculate_impedance_response(self, frequencies):
        # Perform calculations based on speaker type and port shape
        if self.speaker_type == 'rectangular' and self.speaker_number == '2'  and self.port_shape == 'rectangular':
            # flow resistance of material equation 7.8
            R_f = ((4*self.m*(1- self.porosity))/(self.porosity * self.r**2)) * ((1 - 4/np.pi * (1 - self.porosity))/(2 + np.log((self.m*self.porosity)/(2*self.r*self.r_0*self.u))) + (6/np.pi) * (1 - self.porosity))
            #print(R_f)

            # 2 diaphragms radiation impedance based on equation 13.339 and 13.345
            def calculate_Za1(f, a1, r_0, c, d1, truncation_limit):
                k = (2 * np.pi * f / c) * ((1 + self.a3 *(R_f/f)**self.b3) -1j * self.a4 * (R_f/f)**self.b4)

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

                Za1 =  (a1**2 * z11 + 2* a1 * self.a2 * z12 ) / (a1**2 + self.a2**2)

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
                k = (2 * np.pi * f / c) * ((1 + self.a3 *(R_f/f)**self.b3) -1j * self.a4 * (R_f/f)**self.b4)

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
                    term1 = (-1) ** m * fm(self.q, m, n)
                    term2 = (2*m + 1) * np.math.factorial(m) * np.math.factorial(m+1) 
                    term3 = (k*lx/2) ** (2*m +1)
                    sum_Xs += (term1 / term2) * term3

                Xs_a2 = ((2* r_d *c) / (np.sqrt(np.pi))) * ((1- np.sinc(k*lx))/(self.q*k*lx) + (1- np.sinc(self.q* k*lx))/(k*lx) + sum_Xs )



                Z_a2 = (Rs_a2 + 1j * Xs_a2 ) / (self.a_2 * self.b_2)
                
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


            response_1 = np.zeros_like(frequencies)
            Ze = np.zeros_like(frequencies)

            for i in range(len(frequencies)):
            
                Z_e = self.Re + 1j * 2 * np.pi * frequencies[i] * self.Le
                Z_md = 1j *  2 * np.pi * frequencies[i]  * self.Mmd + self.Rms + 1 / (1j * 2 * np.pi * frequencies[i] * self.Cms)

                k = (2 * np.pi * frequencies[i] / self.c) * ((1 + self.a3 *(R_f/frequencies[i])**self.b3) - 1j * self.a4 * (R_f/frequencies[i])**self.b4)
                
                Z_a1 =  calculate_Za1(frequencies[i], self.a , self.r_0, self.c, self.d1, self.truncation_limit)

                Z11 = calculate_Z11(frequencies[i], self.a1, self.a1, self.r_0, self.c, self.truncation_limit,  self.y1, self.y2, self.lx, self.ly, self.lz, self.b, self.b1, self.b2, R_f, self.P_0, self.a3, self.a4, self.b3, self.b4)
                Z12 = Z21 = calculate_Z12(frequencies[i], self.a1, self.a1, self.r_0, self.c, self.truncation_limit,  self.y1, self.y2, self.lx, self.ly, self.lz, self.b, self.b1, self.b2, R_f, self.P_0, self.a3, self.a4, self.b3, self.b4)
                Z22 = calculate_Z22(frequencies[i], self.a1, self.a1, self.r_0, self.c, self.truncation_limit,  self.y1, self.y2, self.lx, self.ly, self.lz, self.b, self.b1, self.b2, R_f, self.P_0, self.a3, self.a4, self.b3, self.b4)

                b11 = (Z11 / Z21)
                b12 = ((Z11 * Z22 - Z12 * Z21) / Z21)
                b21 = (1 / Z21)
                b22 = (Z22 / Z21)

                Ral = 168200
                

                kv = np.sqrt((-1j * np.pi *2 * frequencies[i] * self.r_0) / self.m)
                ap = np.sqrt((self.a_2 * self.b_2) / np.pi)
                ξ = np.sqrt(1 - (2 * jv(1, kv * ap)) / (kv * ap * jv(0, kv * ap)))
                kp = (2 * np.pi * frequencies[i]  * ξ) / self.c
                Zp = (self.r_0 * self.c * ξ) / (self.a_2*self.b_2)
                
                Kn = self.lm / self.a_2

                Bu = (2 * 0.9**(-1) - 1 ) * Kn

                # dynamic density based on equation 4.233
                #r_d = r_0 * (1 - (Q( kv * ap )) / (1 - 0.5 * Bu *kv**2 * ap * Q(kv * ap) )) ** (-1)
            
                r_d = (8*self.m)/(1j * 2 * np.pi * frequencies[i] * (1 + 4 * Bu) * ap**2)
                

                Za2 =  calculate_Za2(frequencies[i], r_d , self.lx, self.ly, self.r_0, self.c, self.truncation_limit)

                C = np.array([[1,  1/2 * Z_e], [0, 1]])
                E = np.array([[0, 1*self.Bl], [1 /  self.Bl, 0]])
                D = np.array([[1, 2 * Z_md], [0, 1]])
                M = np.array([[2 * self.Sd, 0], [0,  2 /  self.Sd]])
                F = np.array([[1, Z_a1], [0, 1]])
                L = np.array([[1, 0], [1/ Ral, 1]])
                B = np.array([[b11, b12], [b21, b22]])
                P = np.array([[np.cos(kp*self.t), 1j*Zp*np.sin(kp*self.t)], [1j*(1/Zp)*np.sin(kp*self.t), np.cos(kp*self.t)]])
                R = np.array([[1, 0], [1/Za2, 1]])

                A = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(C, E), D), M), F), L), B), P) , R)

                a11 = A[0, 0]
                a12 = A[0, 1]
                a21 = A[1, 0]
                a22 = A[1, 1]

                p9 = self.e_g / a11
                Up = p9 / Za2
                
                N = np.dot(np.dot(B, P), R)

                n11 = N[0, 0]
                n12 = N[0, 1]
                n21 = N[1, 0]
                n22 = N[1, 1]

                U6 = n21 * p9
                UB = (n21 - 1/Za2) * p9
                Up = self.e_g / (a11 * Za2)

                ig = a21 * p9

                U_ref = (1/2 * self.e_g * self.Bl * self.Sd) / ( 2 * np.pi * frequencies[i] * self.Mms * self.Re)

                #response[i] = 20 * np.log10(float(abs(U6)) / float(abs(U_ref)))

                response_1[i] = 20 * np.log10(float(abs(UB)) / float(abs(U_ref)))
                Ze[i] = abs(((a11) / (a21)))

            return response_1, Ze
        
            # # Plotting the response
            # plt.figure()
            # plt.semilogx( frequencies, response)
            # plt.xlabel('Frequency (Hz)')
            # plt.ylabel('Response (dB)')
            # plt.title('System Response')
            # plt.grid(which='both')
            # plt.show()


            # # Plotting the impedance
            # plt.figure()
            # plt.semilogx( frequencies, Ze)
            # plt.xlabel('Frequency (Hz)')
            # plt.ylabel('Response (dB)')
            # plt.title('System Impedance')
            # plt.grid(which='both')
            # plt.show()
            

            pass
        
        
        elif self.speaker_type == 'circular' and self.speaker_number == '2' and self.port_shape == 'rectangular':
            R_f = ((4*self.m*(1- self.porosity))/(self.porosity * self.r**2)) * ((1 - 4/np.pi * (1 - self.porosity))/(2 + np.log((self.m*self.porosity)/(2*self.r*self.r_0*self.u))) + (6/np.pi) * (1 - self.porosity))
            
            # 2 diaphragms radiation impedance based on equation 13.339 and 13.345
            def calculate_Za1(f, a1, r_0, c, d1, truncation_limit):
                k = (2 * np.pi * f / c) * ((1 + self.a3 *(R_f/f)**self.b3) -1j * self.a4 * (R_f/f)**self.b4)

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

                Za1 =  (self.a**2 * z11 + 2* self.a * self.a * z12 ) / (self.a**2 + self.a**2)

                return Za1


            # box impedance based on equation 7.131 
            def calculate_Z11(f, lz, a, r_0, c, lx, ly, truncation_limit):

                # wave number k equation 7.11
                k = (2 * np.pi * f / c) * ((1 + self.a3 *(R_f/f)**self.b3) -1j * self.a4 * (R_f/f)**self.b4)

                Zs = r_0 * c + self.P_0/( 1j * 2 * np.pi * f * self.b)
                
                sum_mn = 0
                for m in range(truncation_limit+1):
                    for n in range(truncation_limit+1):
                        kmn = np.sqrt(k**2 - (m*np.pi/lx)**2 - (n*np.pi/ly)**2)
                        delta_m0 = 1 if m == 0 else 0
                        delta_n0 = 1 if n == 0 else 0
                        term1 = ((kmn*Zs)/(k*r_0*c) + 1j * np.tan(kmn*lz)) / (1 + 1j * (( kmn*Zs)/(k*r_0*c)) * np.tan(kmn*lz))
                        term2 = (2 - delta_m0) * (2 - delta_n0) / (kmn * (n**2 * lx**2 + m**2 * ly**2) + delta_m0 * delta_n0)
                        term3 = np.cos((m*np.pi*self.x1)/lx) * np.cos((n*np.pi*self.y1)/ly) * j1((np.pi * a * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
                        term4 = np.cos((m*np.pi*self.x1)/lx) * np.cos((n*np.pi*self.y1)/ly) * j1((np.pi * a * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
                        sum_mn += term1 * term2 * term3 * term4 

                Z11 = (r_0 * c * ( (self.S1*self.S1)/(lx*ly) * (((Zs/(r_0*c)) + 1j * np.tan(k*lz)) / (1 + 1j * ((Zs/(r_0*c)) * np.tan(k*lz)))) + 4 * k*a*a*lx*ly* sum_mn)) / self.Sd**2   
                
                return  Z11



            # box impedance based on equation 7.131 
            def calculate_Z22(f, lz, a, r_0, c, lx, ly, truncation_limit):

                # wave number k equation 7.11
                k = (2 * np.pi * f / c) * ((1 + self.a3 *(R_f/f)**self.b3) -1j * self.a4 * (R_f/f)**self.b4)

                Zs = r_0 * c + self.P_0/( 1j * 2 * np.pi * f * self.b)
                
                sum_mn = 0
                for m in range(truncation_limit+1):
                    for n in range(truncation_limit+1):
                        kmn = np.sqrt(k**2 - (m*np.pi/lx)**2 - (n*np.pi/ly)**2)
                        delta_m0 = 1 if m == 0 else 0
                        delta_n0 = 1 if n == 0 else 0
                        term1 = ((kmn*Zs)/(k*r_0*c) + 1j * np.tan(kmn*lz)) / (1 + 1j * (( kmn*Zs)/(k*r_0*c)) * np.tan(kmn*lz))
                        term2 = (2 - delta_m0) * (2 - delta_n0) / (kmn * (n**2 * lx**2 + m**2 * ly**2) + delta_m0 * delta_n0)
                        term3 = np.cos((m*np.pi*self.x2)/lx) * np.cos((n*np.pi*self.y2)/ly) * j1((np.pi * a * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
                        term4 = np.cos((m*np.pi*self.x2)/lx) * np.cos((n*np.pi*self.y2)/ly) * j1((np.pi * a * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
                        sum_mn += term1 * term2 * term3 * term4 

                Z22 = (r_0 * c * ( (self.S2*self.S2)/(lx*ly) * (((Zs/(r_0*c)) + 1j * np.tan(k*lz)) / (1 + 1j * ((Zs/(r_0*c)) * np.tan(k*lz)))) + 4 * k*a*a*lx*ly* sum_mn)) / self.Sd**2   
                
                return  Z22


            # box impedance based on equation 7.131 
            def calculate_Z12(f, lz, a, r_0, c, lx, ly, truncation_limit):

                # wave number k equation 7.11
                k = (2 * np.pi * f / c) * ((1 + self.a3 *(R_f/f)**self.b3) -1j * self.a4 * (R_f/f)**self.b4)

                Zs = r_0 * c + self.P_0/( 1j * 2 * np.pi * f * self.b)
                
                sum_mn = 0
                for m in range(truncation_limit+1):
                    for n in range(truncation_limit+1):
                        kmn = np.sqrt(k**2 - (m*np.pi/lx)**2 - (n*np.pi/ly)**2)
                        delta_m0 = 1 if m == 0 else 0
                        delta_n0 = 1 if n == 0 else 0
                        term1 = ((kmn*Zs)/(k*r_0*c) + 1j * np.tan(kmn*lz)) / (1 + 1j * (( kmn*Zs)/(k*r_0*c)) * np.tan(kmn*lz))
                        term2 = (2 - delta_m0) * (2 - delta_n0) / (kmn * (n**2 * lx**2 + m**2 * ly**2) + delta_m0 * delta_n0)
                        term3 = np.cos((m*np.pi*self.x1)/lx) * np.cos((n*np.pi*self.y1)/ly) * j1((np.pi * a * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
                        term4 = np.cos((m*np.pi*self.x2)/lx) * np.cos((n*np.pi*self.y2)/ly) * j1((np.pi * a * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
                        sum_mn += term1 * term2 * term3 * term4 

                Z12 = (r_0 * c * ( (self.S1*self.S2)/(lx*ly) * (((Zs/(r_0*c)) + 1j * np.tan(k*lz)) / (1 + 1j * ((Zs/(r_0*c)) * np.tan(k*lz)))) + 4 * k*a*a*lx*ly* sum_mn)) / self.Sd**2   
                
                return  Z12



            # calculate the rectangular box impedance based on equation 13.336, 13.327
            def calculate_Za2 (f, r_d,  lx, ly, r_0, c, N):
                k = (2 * np.pi * f / c) * ((1 + self.a3 *(R_f/f)**self.b3) -1j * self.a4 * (R_f/f)**self.b4)

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
                    term1 = (-1) ** m * fm(self.q, m, n)
                    term2 = (2*m + 1) * np.math.factorial(m) * np.math.factorial(m+1) 
                    term3 = (k*lx/2) ** (2*m +1)
                    sum_Xs += (term1 / term2) * term3

                Xs_a2 = ((2* r_d *c) / (np.sqrt(np.pi))) * ((1- np.sinc(k*lx))/(self.q*k*lx) + (1- np.sinc(self.q* k*lx))/(k*lx) + sum_Xs )



                Z_a2 = (Rs_a2 + 1j * Xs_a2 ) / (self.a_2 * self.b_2)
                
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



            # Preallocate memory 
            response = np.zeros_like(frequencies)
            Ze = np.zeros_like(frequencies)

            for i in range(len(frequencies)):
            
                Z_e = self.Re + 1j * 2 * np.pi * frequencies[i] * self.Le
                Z_md = 1j *  2 * np.pi * frequencies[i]  * self.Mmd + self.Rms + 1 / (1j * 2 * np.pi * frequencies[i] * self.Cms)

                k = (2 * np.pi * frequencies[i] / self.c) * ((1 + self.a3 *(R_f/frequencies[i])**self.b3) - 1j * self.a4 * (R_f/frequencies[i])**self.b4)
                
                Z_a1 =  calculate_Za1(frequencies[i], self.a , self.r_0, self.c, self.d1, self.truncation_limit)

                Z11 = calculate_Z11(frequencies[i], self.lz, self.a, self.r_0, self.c, self.lx, self.ly, self.truncation_limit)
                Z12 = Z21 = calculate_Z12(frequencies[i], self.lz, self.a, self.r_0, self.c, self.lx, self.ly, self.truncation_limit)
                Z22 = calculate_Z22(frequencies[i], self.lz, self.a, self.r_0, self.c, self.lx, self.ly, self.truncation_limit)

                b11 = (Z11 / Z21)
                b12 = ((Z11 * Z22 - Z12 * Z21) / Z21)
                b21 = (1 / Z21)
                b22 = (Z22 / Z21)

                Ral = 168200
                

                kv = np.sqrt((-1j * np.pi *2 * frequencies[i] * self.r_0) / self.m)
                ap = np.sqrt((self.a_2 * self.b_2) / np.pi)
                ξ = np.sqrt(1 - (2 * jv(1, kv * ap)) / (kv * ap * jv(0, kv * ap)))
                kp = (2 * np.pi * frequencies[i]  * ξ) / self.c
                Zp = (self.r_0 * self.c * ξ) / (self.a_2*self.b_2)
                
                Kn = self.lm / self.a_2

                Bu = (2 * 0.9**(-1) - 1 ) * Kn

                # dynamic density based on equation 4.233
                #r_d = r_0 * (1 - (Q( kv * ap )) / (1 - 0.5 * Bu *kv**2 * ap * Q(kv * ap) )) ** (-1)
            
                r_d = (8*self.m)/(1j * 2 * np.pi * frequencies[i] * (1 + 4 * Bu) * self.a_2**2)
                

                Za2 =  calculate_Za2(frequencies[i], r_d , self.lx, self.ly, self.r_0, self.c, self.truncation_limit)

                C = np.array([[1,  1/2 * Z_e], [0, 1]])
                E = np.array([[0, 1*self.Bl], [1 /  self.Bl, 0]])
                D = np.array([[1, 2 * Z_md], [0, 1]])
                M = np.array([[2 * self.Sd, 0], [0,  2 /  self.Sd]])
                F = np.array([[1, Z_a1], [0, 1]])
                L = np.array([[1, 0], [1/ Ral, 1]])
                B = np.array([[b11, b12], [b21, b22]])
                P = np.array([[np.cos(kp*self.t), 1j*Zp*np.sin(kp*self.t)], [1j*(1/Zp)*np.sin(kp*self.t), np.cos(kp*self.t)]])
                R = np.array([[1, 0], [1/Za2, 1]])

                A = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(C, E), D), M), F), L), B), P) , R)

                a11 = A[0, 0]
                a12 = A[0, 1]
                a21 = A[1, 0]
                a22 = A[1, 1]

                p9 = self.e_g / a11
                Up = p9 / Za2
                
                N = np.dot(np.dot(B, P), R)

                n11 = N[0, 0]
                n12 = N[0, 1]
                n21 = N[1, 0]
                n22 = N[1, 1]

                U6 = n21 * p9
                UB = (n21 - 1/Za2) * p9
                Up = self.e_g / (a11 * Za2)

                ig = a21 * p9

                U_ref = (2 * self.e_g * self.Bl * self.Sd) / ( 2 * np.pi * frequencies[i] * self.Mms * self.Re)

                #response[i] = 20 * np.log10(float(abs(U6)) / float(abs(U_ref)))

                response[i] = 20 * np.log10(float(abs(UB)) / float(abs(U_ref)))
                Ze[i] = abs(((a11) / (a21)))

            return response, Ze
            # # Plotting the response
            # plt.figure()
            # plt.semilogx( frequencies, response)
            # plt.xlabel('Frequency (Hz)')
            # plt.ylabel('Response (dB)')
            # plt.title('System Response')
            # plt.grid(which='both')
            # plt.show()


            # # Plotting the impedance
            # plt.figure()
            # plt.semilogx( frequencies, Ze)
            # plt.xlabel('Frequency (Hz)')
            # plt.ylabel('Response (dB)')
            # plt.title('System Impedance')
            # plt.grid(which='both')
            # plt.show()
            pass
        
        
        elif self.speaker_type == 'circular' and self.speaker_number == '2' and self.port_shape == 'circular':
 
            # flow resistance of material equation 7.8
            R_f = ((4*self.m*(1- self.porosity))/(self.porosity * self.r**2)) * ((1 - 4/np.pi * (1 - self.porosity))/(2 + np.log((self.m*self.porosity)/(2*self.r*self.r_0*self.u))) + (6/np.pi) * (1 - self.porosity))
            #print(R_f)

            # 2 diaphragms radiation impedance based on equation 13.339 and 13.345
            def calculate_Za1(f, a, r_0, c, d1, truncation_limit):
                k = (2 * np.pi * f / c) * ((1 + self.a3 *(R_f/f)**self.b3) -1j * self.a4 * (R_f/f)**self.b4)

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
                k = (2 * np.pi * f / c) * ((1 + self.a3 *(R_f/f)**self.b3) -1j * self.a4 * (R_f/f)**self.b4)

                Zs = r_0 * c + self.P_0/( 1j * 2 * np.pi * f * self.b)
                
                sum_mn = 0
                for m in range(truncation_limit+1):
                    for n in range(truncation_limit+1):
                        kmn = np.sqrt(k**2 - (m*np.pi/lx)**2 - (n*np.pi/ly)**2)
                        delta_m0 = 1 if m == 0 else 0
                        delta_n0 = 1 if n == 0 else 0
                        term1 = ((kmn*Zs)/(k*r_0*c) + 1j * np.tan(kmn*lz)) / (1 + 1j * (( kmn*Zs)/(k*r_0*c)) * np.tan(kmn*lz))
                        term2 = (2 - delta_m0) * (2 - delta_n0) / (kmn * (n**2 * lx**2 + m**2 * ly**2) + delta_m0 * delta_n0)
                        term3 = np.cos((m*np.pi*self.x1)/lx) * np.cos((n*np.pi*self.y1)/ly) * j1((np.pi * a * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
                        term4 = np.cos((m*np.pi*self.x1)/lx) * np.cos((n*np.pi*self.y1)/ly) * j1((np.pi * a * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
                        sum_mn += term1 * term2 * term3 * term4 

                Z11 = (r_0 * c * ( (self.S1*self.S1)/(lx*ly) * (((Zs/(r_0*c)) + 1j * np.tan(k*lz)) / (1 + 1j * ((Zs/(r_0*c)) * np.tan(k*lz)))) + 4 * k*a*a*lx*ly* sum_mn)) / self.Sd**2   
                
                return  Z11



            # box impedance based on equation 7.131 
            def calculate_Z22(f, lz, a, r_0, c, lx, ly, truncation_limit):

                # wave number k equation 7.11
                k = (2 * np.pi * f / c) * ((1 + self.a3 *(R_f/f)**self.b3) -1j * self.a4 * (R_f/f)**self.b4)

                Zs = r_0 * c + self.P_0/( 1j * 2 * np.pi * f * self.b)
                
                sum_mn = 0
                for m in range(truncation_limit+1):
                    for n in range(truncation_limit+1):
                        kmn = np.sqrt(k**2 - (m*np.pi/lx)**2 - (n*np.pi/ly)**2)
                        delta_m0 = 1 if m == 0 else 0
                        delta_n0 = 1 if n == 0 else 0
                        term1 = ((kmn*Zs)/(k*r_0*c) + 1j * np.tan(kmn*lz)) / (1 + 1j * (( kmn*Zs)/(k*r_0*c)) * np.tan(kmn*lz))
                        term2 = (2 - delta_m0) * (2 - delta_n0) / (kmn * (n**2 * lx**2 + m**2 * ly**2) + delta_m0 * delta_n0)
                        term3 = np.cos((m*np.pi*self.x2)/lx) * np.cos((n*np.pi*self.y2)/ly) * j1((np.pi * a * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
                        term4 = np.cos((m*np.pi*self.x2)/lx) * np.cos((n*np.pi*self.y2)/ly) * j1((np.pi * a * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
                        sum_mn += term1 * term2 * term3 * term4 

                Z22 = (r_0 * c * ( (self.S2*self.S2)/(lx*ly) * (((Zs/(r_0*c)) + 1j * np.tan(k*lz)) / (1 + 1j * ((Zs/(r_0*c)) * np.tan(k*lz)))) + 4 * k*a*a*lx*ly* sum_mn)) / self.Sd**2   
                
                return  Z22


            # box impedance based on equation 7.131 
            def calculate_Z12(f, lz, a, r_0, c, lx, ly, truncation_limit):

                # wave number k equation 7.11
                k = (2 * np.pi * f / c) * ((1 + self.a3 *(R_f/f)**self.b3) -1j * self.a4 * (R_f/f)**self.b4)

                Zs = r_0 * c + self.P_0/( 1j * 2 * np.pi * f * self.b)
                
                sum_mn = 0
                for m in range(truncation_limit+1):
                    for n in range(truncation_limit+1):
                        kmn = np.sqrt(k**2 - (m*np.pi/lx)**2 - (n*np.pi/ly)**2)
                        delta_m0 = 1 if m == 0 else 0
                        delta_n0 = 1 if n == 0 else 0
                        term1 = ((kmn*Zs)/(k*r_0*c) + 1j * np.tan(kmn*lz)) / (1 + 1j * (( kmn*Zs)/(k*r_0*c)) * np.tan(kmn*lz))
                        term2 = (2 - delta_m0) * (2 - delta_n0) / (kmn * (n**2 * lx**2 + m**2 * ly**2) + delta_m0 * delta_n0)
                        term3 = np.cos((m*np.pi*self.x1)/lx) * np.cos((n*np.pi*self.y1)/ly) * j1((np.pi * a * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
                        term4 = np.cos((m*np.pi*self.x2)/lx) * np.cos((n*np.pi*self.y2)/ly) * j1((np.pi * a * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
                        sum_mn += term1 * term2 * term3 * term4 

                Z12 = (r_0 * c * ( (self.S1*self.S2)/(lx*ly) * (((Zs/(r_0*c)) + 1j * np.tan(k*lz)) / (1 + 1j * ((Zs/(r_0*c)) * np.tan(k*lz)))) + 4 * k*a*a*lx*ly* sum_mn)) / self.Sd**2   
                
                return  Z12




            # Preallocate memory 
            response = np.zeros_like(frequencies)
            Ze = np.zeros_like(frequencies)

            for i in range(len(frequencies)):
            
                Z_e = self.Re + 1j * 2 * np.pi * frequencies[i] * self.Le
                Z_md = 1j *  2 * np.pi * frequencies[i]  *self. Mmd + self.Rms + 1 / (1j * 2 * np.pi * frequencies[i] * self.Cms)

                k = (2 * np.pi * frequencies[i] / self.c) * ((1 + self.a3 *(R_f/frequencies[i])**self.b3) - 1j * self.a4 * (R_f/frequencies[i])**self.b4)
                
                Z_a1 =  calculate_Za1(frequencies[i], self.a , self.r_0, self.c, self.d1, self.truncation_limit)

                Z11 = calculate_Z11(frequencies[i], self.lz, self.a, self.r_0, self.c, self.lx, self.ly, self.truncation_limit)
                Z12 = Z21 = calculate_Z12(frequencies[i], self.lz, self.a, self.r_0, self.c, self.lx, self.ly, self.truncation_limit)
                Z22 = calculate_Z22(frequencies[i],self.lz, self.a, self.r_0, self.c,  self.lx, self.ly, self.truncation_limit)

                b11 = (Z11 / Z21)
                b12 = ((Z11 * Z22 - Z12 * Z21) / Z21)
                b21 = (1 / Z21)
                b22 = (Z22 / Z21)

                Ral = 168200
                

                kv = np.sqrt((-1j * np.pi *2 * frequencies[i] * self.r_0) / self.m)
                ap = np.sqrt((self.Sp) / np.pi)
                ξ = np.sqrt(1 - (2 * jv(1, kv * ap)) / (kv * ap * jv(0, kv * ap)))
                kp = (2 * np.pi * frequencies[i]  * ξ) / self.c
                Zp = (self.r_0 * self.c * ξ) / (self.Sp)
                
            
                
                # Calculate the impedance of the port
                R_s = self.r_0 * self.c * (1- jv(1, k*self.a_p)/ (k*self.a_p)) 
                X_s = self.r_0 * self.c * (1 +  mpmath.struveh(1, k * self.a_p) / (k*self.a_p))
                Za2 =  (R_s + 1j * X_s)

                C = np.array([[1,  1/2 * Z_e], [0, 1]])
                E = np.array([[0, 1*self.Bl], [1 /  self.Bl, 0]])
                D = np.array([[1, 2 * Z_md], [0, 1]])
                M = np.array([[2 * self.Sd, 0], [0,  2 /  self.Sd]])
                F = np.array([[1, Z_a1], [0, 1]])
                L = np.array([[1, 0], [1/ Ral, 1]])
                B = np.array([[b11, b12], [b21, b22]])
                P = np.array([[np.cos(kp*self.t), 1j*Zp*np.sin(kp*self.t)], [1j*(1/Zp)*np.sin(kp*self.t), np.cos(kp*self.t)]])
                R = np.array([[1, 0], [1/Za2, 1]])

                A = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(C, E), D), M), F), L), B), P) , R)

                a11 = A[0, 0]
                a12 = A[0, 1]
                a21 = A[1, 0]
                a22 = A[1, 1]

                p9 = self.e_g / a11
                Up = p9 / Za2
                
                N = np.dot(np.dot(B, P), R)

                n11 = N[0, 0]
                n12 = N[0, 1]
                n21 = N[1, 0]
                n22 = N[1, 1]

                U6 = n21 * p9
                UB = (n21 - 1/Za2) * p9
                Up = self.e_g / (a11 * Za2)

                ig = a21 * p9

                U_ref = (2 * self.e_g * self.Bl * self.Sd) / ( 2 * np.pi * frequencies[i] * self.Mms * self.Re)

                #response[i] = 20 * np.log10(float(abs(U6)) / float(abs(U_ref)))

                response[i] = 20 * np.log10(float(abs(UB)) / float(abs(U_ref)))
                Ze[i] = abs(((a11) / (a21)))

            return response, Ze
        
            pass
        
        
        elif self.speaker_type == 'circular' and  self.speaker_number == '1' and self.port_shape == 'rectangular':

            R_f = ((4*self.m*(1- self.porosity))/(self.porosity * self.r**2)) * ((1 - 4/np.pi * (1 - self.porosity))/(2 + np.log((self.m*self.porosity)/(2*self.r*self.r_0*self.u))) + (6/np.pi) * (1 - self.porosity))
            #print(R_f)

            # # box impedance based on equation 7.131
            # def calculate_ZAB(f, lz, a, r_0, c, lx, ly, truncation_limit):

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

            #     ZAB = (r_0 * c * ( (Sd*Sd)/(lx*ly) * (((Zs/(r_0*c)) + 1j * np.tan(k*lz)) / (1 + 1j * ((Zs/(r_0*c)) * np.tan(k*lz)))) + 4 * k*a*a*lx*ly* sum_mn)) / (Sd*Sd)   
                
            #     return  ZAB





            # # box impedance based on equation 7.131 
            def calculate_Z11(f, lz, a, a_2, r_0, c, lx, ly, truncation_limit):

                # wave number k equation 7.11
                k = (2 * np.pi * f / c) * ((1 + self.a3 *(R_f/f)**self.b3) -1j * self.a4 * (R_f/f)**self.b4)

                Zs = r_0 * c + self.P_0/( 1j * 2 * np.pi * f * self.b)
                
                sum_mn = 0
                for m in range(truncation_limit+1):
                    for n in range(truncation_limit+1):
                        kmn = np.sqrt(k**2 - (m*np.pi/lx)**2 - (n*np.pi/ly)**2)
                        delta_m0 = 1 if m == 0 else 0
                        delta_n0 = 1 if n == 0 else 0
                        term1 = ((kmn*Zs)/(k*r_0*c) + 1j * np.tan(kmn*lz)) / (1 + 1j * (( kmn*Zs)/(k*r_0*c)) * np.tan(kmn*lz))
                        term2 = (2 - delta_m0) * (2 - delta_n0) / (kmn * (n**2 * lx**2 + m**2 * ly**2) + delta_m0 * delta_n0)
                        term3 = np.cos((m*np.pi*self.x1)/lx) * np.cos((n*np.pi*self.y1)/ly) * j1((np.pi * a * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
                        term4 = np.cos((m*np.pi*self.x1)/lx) * np.cos((n*np.pi*self.y1)/ly) * j1((np.pi * a * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
                        sum_mn += term1 * term2 * term3 * term4 

                Z11 = (r_0 * c * ( (self.Sd*self.Sd)/(lx*ly) * (((Zs/(r_0*c)) + 1j * np.tan(k*lz)) / (1 + 1j * ((Zs/(r_0*c)) * np.tan(k*lz)))) + 4 * k*a*a*lx*ly* sum_mn)) / self.Sd**2   
                
                return  Z11



            # box impedance based on equation 7.131 
            def calculate_Z22(f, lz, a, a_2, r_0, c, lx, ly, truncation_limit):

                # wave number k equation 7.11
                k = (2 * np.pi * f / c) * ((1 + self.a3 *(R_f/f)**self.b3) -1j * self.a4 * (R_f/f)**self.b4)

                Zs = r_0 * c + self.P_0/( 1j * 2 * np.pi * f * self.b)
                
                sum_mn = 0
                for m in range(truncation_limit+1):
                    for n in range(truncation_limit+1):
                        kmn = np.sqrt(k**2 - (m*np.pi/lx)**2 - (n*np.pi/ly)**2)
                        delta_m0 = 1 if m == 0 else 0
                        delta_n0 = 1 if n == 0 else 0
                        term1 = ((kmn*Zs)/(k*r_0*c) + 1j * np.tan(kmn*lz)) / (1 + 1j * (( kmn*Zs)/(k*r_0*c)) * np.tan(kmn*lz))
                        term2 = (2 - delta_m0) * (2 - delta_n0) / (kmn * (n**2 * lx**2 + m**2 * ly**2) + delta_m0 * delta_n0)
                        term3 = np.cos((m*np.pi*self.x2)/lx) * np.cos((n*np.pi*self.y2)/ly) * j1((np.pi * a_2 * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
                        term4 = np.cos((m*np.pi*self.x2)/lx) * np.cos((n*np.pi*self.y2)/ly) * j1((np.pi * a_2 * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
                        sum_mn += term1 * term2 * term3 * term4 

                Z22 = (r_0 * c * ( (self.Sp*self.Sp)/(lx*ly) * (((Zs/(r_0*c)) + 1j * np.tan(k*lz)) / (1 + 1j * ((Zs/(r_0*c)) * np.tan(k*lz)))) + 4 * k*a_2*a_2*lx*ly* sum_mn)) / self.Sp**2   
                
                return  Z22


            # box impedance based on equation 7.131 
            def calculate_Z12(f, lz, a,a_2, r_0, c, lx, ly, truncation_limit):

                # wave number k equation 7.11
                k = (2 * np.pi * f / c) * ((1 + self.a3 *(R_f/f)**self.b3) -1j * self.a4 * (R_f/f)**self.b4)

                Zs = r_0 * c + self.P_0/( 1j * 2 * np.pi * f * self.b)
                
                sum_mn = 0
                for m in range(truncation_limit+1):
                    for n in range(truncation_limit+1):
                        kmn = np.sqrt(k**2 - (m*np.pi/lx)**2 - (n*np.pi/ly)**2)
                        delta_m0 = 1 if m == 0 else 0
                        delta_n0 = 1 if n == 0 else 0
                        term1 = ((kmn*Zs)/(k*r_0*c) + 1j * np.tan(kmn*lz)) / (1 + 1j * (( kmn*Zs)/(k*r_0*c)) * np.tan(kmn*lz))
                        term2 = (2 - delta_m0) * (2 - delta_n0) / (kmn * (n**2 * lx**2 + m**2 * ly**2) + delta_m0 * delta_n0)
                        term3 = np.cos((m*np.pi*self.x1)/lx) * np.cos((n*np.pi*self.y1)/ly) * j1((np.pi * a * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
                        term4 = np.cos((m*np.pi*self.x2)/lx) * np.cos((n*np.pi*self.y2)/ly) * j1((np.pi * a_2 * np.sqrt(n**2 * lx**2 + m**2 * ly**2))/(lx*ly))
                        sum_mn += term1 * term2 * term3 * term4 

                Z12 = (r_0 * c * ( (self.Sd*self.Sp)/(lx*ly) * (((Zs/(r_0*c)) + 1j * np.tan(k*lz)) / (1 + 1j * ((Zs/(r_0*c)) * np.tan(k*lz)))) + 4 * k*a*a_2*lx*ly* sum_mn)) / (self.Sd*self.Sp)   
                
                return  Z12





            # Preallocate memory 
            response = np.zeros_like(frequencies)
            Ze = np.zeros_like(frequencies)

            for i in range(len(frequencies)):
            
                Z_e = self.Re + 1j * 2 * np.pi * frequencies[i] * self.Le
                Z_md = 1j *  2 * np.pi * frequencies[i]  * self.Mmd + self.Rms + 1 / (1j * 2 * np.pi * frequencies[i] * self.Cms)

                k = (2 * np.pi * frequencies[i] / self.c) * ((1 + self.a3 *(R_f/frequencies[i])**self.b3) - 1j * self.a4 * (R_f/frequencies[i])**self.b4)
                
                R_s = self.r_0 * self.c * k**2 * self.a**2
                X_s = (self.r_0 * self.c * 8 * k * self.a)/(3 * np.pi)
                Z_a1 =  (R_s + 1j * X_s)

                #Z_ab =  calculate_ZAB(frequencies[i], lz, a,  r_0, c, lx, ly, truncation_limit)


                Z11 = calculate_Z11(frequencies[i], self.lz, self.a, self.a, self.r_0, self.c, self.lx, self.ly, self.truncation_limit)
                Z22 = calculate_Z22(frequencies[i], self.lz, self.a_p, self.a_p, self.r_0, self.c, self.lx, self.ly, self.truncation_limit)
                Z12 = calculate_Z12(frequencies[i], self.lz, self.a, self.a_p, self.r_0, self.c, self.lx, self.ly, self.truncation_limit)
                Z21 = Z12

                b11 = (Z11 / Z21)
                b12 = ((Z11 * Z22 - Z12 * Z21) / Z21)
                b21 = (1 / Z21)
                b22 = (Z22 / Z21)

                Ral = 168200
                

                kv = np.sqrt((-1j * np.pi *2 * frequencies[i] * self.r_0) / self.m)
                ap = np.sqrt((self.Sp) / np.pi)
                ξ = np.sqrt(1 - (2 * jv(1, kv * ap)) / (kv * ap * jv(0, kv * ap)))
                kp = (2 * np.pi * frequencies[i]  * ξ) / self.c
                Zp = (self.r_0 * self.c * ξ) / (self.Sp)

                # Calculate the impedance of the port 
                R_s = self.r_0 * self.c * k**2 * self.a_p**2 /2
                X_s = (self.r_0 * self.c * 8 * k * self.a_p)/(3 * np.pi)
                Za2 =  (R_s + 1j * X_s) 

                C = np.array([[1,  2 /2 * Z_e], [0, 1]])
                E = np.array([[0, 1*self.Bl], [1 /  self.Bl, 0]])
                D = np.array([[1,  1 * Z_md], [0, 1]])
                M = np.array([[1 * self.Sd, 0], [0,  1 /  self.Sd]])
                F = np.array([[1, Z_a1], [0, 1]])
                L = np.array([[1, 0], [1/ Ral, 1]])
                B = np.array([[b11, b12], [b21, b22]])
                P = np.array([[np.cos(kp*self.t), 1j*Zp*np.sin(kp*self.t)], [1j*(1/Zp)*np.sin(kp*self.t), np.cos(kp*self.t)]])
                R = np.array([[1, 0], [1/Za2, 1]])

                A = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(C, E), D), M), F), L), B), P) , R)

                a11 = A[0, 0]
                a12 = A[0, 1]
                a21 = A[1, 0]
                a22 = A[1, 1]

                p9 = self.e_g / a11
                Up = p9 / Za2
                
                N = np.dot(np.dot(B, P), R)

                n11 = N[0, 0]
                n12 = N[0, 1]
                n21 = N[1, 0]
                n22 = N[1, 1]

                U6 = n21 * p9
                UB = (n21 - 1/Za2) * p9
                Up = self.e_g / (a11 * Za2)

                ig = a21 * p9

                U_ref = ( 1 * self.e_g * self.Bl * self.Sd) / ( 2 * np.pi * frequencies[i] * self.Mms * self.Re)

                #response[i] = 20 * np.log10(float(abs(U6)) / float(abs(U_ref)))

                response[i] = 20 * np.log10(float(abs(UB)) / float(abs(U_ref)))
                Ze[i] = abs(((a11) / (a21)))

            return response, Ze
            pass
        
        
        elif self.speaker_type == 'circular' and self.speaker_number == '1' and self.port_shape == 'circular':

            pass
        else:
            print("Invalid speaker type or port shape")






# Define a range of frequencies
octave_steps = 24
frequencies_per_octave = octave_steps * np.log(2)
min_frequency = 10
max_frequency = 1000
num_points = int(octave_steps * np.log2(max_frequency / min_frequency)) + 1
frequencies = np.logspace(np.log2(min_frequency), np.log2(max_frequency), num=num_points, base=2)

# Create an instance of the BassReflexEnclosure class
bass_reflex_enclosure_1 = BassReflexEnclosure('rectangular', '2', 'rectangular')
bass_reflex_enclosure_2 = BassReflexEnclosure('circular', '2', 'rectangular')
bass_reflex_enclosure_3 = BassReflexEnclosure('circular', '2', 'circular')
bass_reflex_enclosure_4 = BassReflexEnclosure('circular', '1', 'rectangular')
bass_reflex_enclosure_5 = BassReflexEnclosure('circular', '1', 'circular')

# Calculate the impedance response
response_1, Ze_1 = bass_reflex_enclosure_1.calculate_impedance_response(frequencies)
response_2 , Ze_2 = bass_reflex_enclosure_2.calculate_impedance_response(frequencies)
response_3 , Ze_3 = bass_reflex_enclosure_3.calculate_impedance_response(frequencies)
response_4 , Ze_4 = bass_reflex_enclosure_4.calculate_impedance_response(frequencies)

# plot on the same graph the response of the two cases
plt.figure()
plt.semilogx(frequencies, response_1 , label='2 Rect. Loudspeakers with Rect. port')
plt.semilogx(frequencies, response_2 , label='2 Circ. Loudspeakers with Rect. port') 
plt.semilogx(frequencies, response_3 , label='2 Circ. Loudspeakers with Circ. port') 
plt.semilogx(frequencies, response_4 , label='1 Circ. Loudspeakers with Rect. port')    
plt.xlabel('Frequency (Hz)')
plt.ylabel('Response (dB)')
plt.title('System Response')
plt.grid(which='both')
plt.legend()
plt.show()


# plot on the same graph the impedance of the two cases
plt.figure()
plt.semilogx(frequencies, Ze_1 , label='2 Rect. Loudspeakers with Rect. port')
plt.semilogx(frequencies, Ze_2 , label='2 Circ. Loudspeakers with Rect. port')
plt.semilogx(frequencies, Ze_3 , label='2 Circ. Loudspeakers with Circ. port')   
plt.semilogx(frequencies, Ze_4 , label='1 Circ. Loudspeakers with Rect. port')   
plt.xlabel('Frequency (Hz)')
plt.ylabel('Response (dB)')
plt.title('System Response')
plt.grid(which='both')
plt.legend()
plt.show()


