import numpy as np
import matplotlib.pyplot as plt
import mpmath
import scipy.special
from scipy.special import  j1, jv, gamma, spherical_jn, spherical_yn



class BassReflexEnclosure:
    def __init__(self, speaker_type, port_shape ) :
        
        self.speaker_type = speaker_type
        self.port_shape = port_shape
        
        
        # Parameters
        self.Re = 6.72 # voice coil resistance
        self.Le = 0.00067 # voice coil inductance
        self.e_g = 2.83  # voice coil height
        self.Qes = 0.522 # electrical Q factor
        self.Qms = 1.9  # mechanical Q factor 
        self.fs = 37  # resonant frequency in Hz
        self.Sd = 0.012  # diaphragm area in m^2
        self.Vas = 24  # equivalent volume of compliance in m^3
        self.Qts = 0.41 # total Q factor
        self.Vab = 17.6  # volume of the enclosure in m^3
        self.Cms = 0.00119 # mechanical compliance
        self.Mms = 0.0156 # mechanical mass
        self.Bl = 6.59    # force factor
        self.pmax = 0.28  # maximum linear excursion
        self.Vmax = 0.03  # maximum volume displacement
        self.Vp = 0.3    # volume of the port
        self.fb = 36  # tuning frequency
        self.t = 0.2    # length of the port
        self.Sp = 0.0015  # area of the port
        self.Va = 12  # volume of the enclosure without the volume of lining material in m^3
        self.Vm = 4   # volume of the lining material in m^3
        self.Vb = 16  # total volume of the enclosure in m^3
        self.Rms = 1 / self.Qms * np.sqrt(self.Mms / self.Cms)  # mechanical resistance
        self.a = np.sqrt(self.Sd / np.pi) # radius of the diaphragm
        self.a_p = np.sqrt(self.Sp / np.pi) # radius of the port

        self.c = 343 # speed of sound in air
        self.porosity = 0.99  # porosity
        self.P_0 = 10 ** 5  # atmospheric pressure
        self.u = 0.03  # flow velocity in the material in m/s
        self.m = 1.86 * 10**(-5)  # viscosity coefficient in N.s/m^2
        self.r = 50 * 10 ** (-6) # fiber diameter
        self.truncation_limit = 10 # Truncation limit for the double summation
        self.d = 0.064  # the thickness of the lining material
        self.r_0 = 1.18  # air density in kg/m^3
        self.lm = 6 * 10**(-8) # molecular mean free path length between collisions


        self.lx = 0.15  # width of the enclosure
        self.ly = 0.401  # height of the enclosure
        self.lz = 0.228  # depth of the enclosure
        self.x1 = 0.075  # distance from the center of the diaphragm to the box wall
        self.y1 = 0.25  # distance from the center of the diaphragm to the box wall
        self.q = self.ly / self.lx  # aspect ratio of the enclosure
        self.a2 = 0.1  # width of the port
        self.b2 = 0.015  # height of the port


        # Values of coefficients (Table 7.1)
        self.a3 = 0.0858
        self.a4 = 0.175
        self.b3 = 0.7
        self.b4 = 0.59





    def calculate_impedance_response(self, frequencies):
    # Perform calculations based on speaker type and port shape
      if self.speaker_type == 'circular'  and self.port_shape == 'rectangular':
         
        # flow resistance of material equation 7.8
        R_f = ((4*self.m*(1- self.porosity))/(self.porosity * self.r**2)) * ((1 - 4/np.pi * (1 - self.porosity))/(2 + np.log((self.m*self.porosity)/(2*self.r*self.r_0*self.u))) + (6/np.pi) * (1 - self.porosity))

        # box impedance based on equation 7.131 
        def calculate_ZAB(f, lz, a, r_0, c, lx, ly, truncation_limit):

            # wave number k equation 7.11
            k = (2 * np.pi * f / c) * ((1 + self.a3 *(R_f/f)**self.b3) -1j * self.a4 * (R_f/f)**self.b4)

            Zs = r_0 * c + self.P_0/( 1j * 2 * np.pi * f * self.d)
            
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

            ZAB = (r_0 * c * ( (self.Sd*self.Sd)/(lx*ly) * (((Zs/(r_0*c)) + 1j * np.tan(k*lz)) / (1 + 1j * ((Zs/(r_0*c)) * np.tan(k*lz)))) + 4 * k*a*a*lx*ly* sum_mn)) / self.Sd**2   
            
            return  ZAB




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



            Z_a2 = (Rs_a2 + 1j * Xs_a2 ) / self.Sp
            
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
              first_sum += ( (-1) ** (p - n) * (q) ** (2 * n - 1)) / ((2 * p - 1) * (1 + q**2)**((p - 1/2)) ) * scipy.special.binom((m - n), p - n) 

            first_term = scipy.special.binom((2 * m + 3), (2 * n)) * first_sum

            second_sum = 0

            for p in range(m - n, m + 1):
              second_sum += (scipy.special.binom((p - m + n), n) * (-1) ** (p - m + n) * (q) ** (2 * n + 2)) / ((2 * p - 1) * (1 + q**(-2))**((p - 1/2)))

            second_term = scipy.special.binom((2 * m + 3), (2 * n + 3)) * second_sum

            return first_term + second_term




        # Preallocate memory 
        response_2 = np.zeros_like(frequencies)
        Ze = np.zeros_like(frequencies)


        for i in range(len(frequencies)):
            
            # Calculate the electrical impedance 
            Z_e = self.Re + 1j * 2 * np.pi * frequencies[i] * self.Le
          
            # Calculate the mechanical impedance
            Mmd = self.Mms - 16 * self.r_0 * self.a**3 / 3
            Z_md = 1j *  2 * np.pi * frequencies[i] * Mmd + self.Rms + 1 / (1j * 2 * np.pi * frequencies[i] * self.Cms)
            
            # wave number k equation 7.11
            k = (2 * np.pi * frequencies[i] / self.c) * ((1 + self.a3 *(R_f/frequencies[i])**self.b3) - 1j * self.a4 * (R_f/frequencies[i])**self.b4)
            
            # diaphragm radiation impedance based on equations 13.116 - 13.118
            R_s = self.r_0 * self.c * k**2 * self.a**2
            X_s = (self.r_0 * self.c * 8 * k * self.a)/(3 * np.pi)
            Z_a1 =  (R_s + 1j * X_s)
            
            # calculate box impedance
            Z_ab = calculate_ZAB(frequencies[i], self.lz, self.a, self.r_0, self.c, self.lx, self.ly, self.truncation_limit)

            # calculate the leakage resistance
            CAA = (self.Va*10**(-3))/(1.4*self.P_0) # compliance of the air in the box
            CAM = (self.Vm*10**(-3))/self.P_0  # compliance of the air in the lining material
            Cab = CAA + 1.4 * CAM  # apparent compliance of the air in the box
            Ral = 7/ (2 * np.pi * self.fb * Cab)
            #print(Ral)


            kv = np.sqrt((-1j * np.pi *2 * frequencies[i] * self.r_0) / self.m)
            ap = np.sqrt((self.Sp) / np.pi)
            kvap = kv * ap
            
            #ξ = np.sqrt(1 - 1j * (2 * jv(1, kvap)) / (kvap * jv(0, kvap)))
            ξ = 0.998 + 0.001j
            kp = (2 * np.pi * frequencies[i]  * ξ) / self.c
            Zp = (self.r_0 * self.c * ξ) / (self.Sp)
            Kn = self.lm / self.a2
            Bu = (2 * 0.9**(-1) - 1 ) * Kn

            # dynamic density based on equation 4.233
            r_d = (-8*self.r_0)/ ((1 + 4*Bu)* kv**2 * self.a2**2)

            # calculate the port impedance 
            Z_a2 =  calculate_Za2(frequencies[i], r_d,  self.lx, self.ly, self.r_0, self.c, self.truncation_limit)


            C = np.array([[1, 1* Z_e], [0, 1]])
            E = np.array([[0, 1* self.Bl], [1 / self.Bl, 0]])
            D = np.array([[1, 1*Z_md], [0, 1]])
            M = np.array([[1* self.Sd, 0], [0, 1 / self.Sd]])
            F = np.array([[1, 1*Z_a1], [0, 1]])
            L = np.array([[1, 0], [1/ Ral, 1]])
            B = np.array([[1, 0], [1 / Z_ab, 1]])
            P = np.array([[np.cos(kp*self.t), 1j*Zp*np.sin(kp*self.t)], [1j*(1/Zp)*np.sin(kp*self.t), np.cos(kp*self.t)]])
            R = np.array([[1, 0], [1/Z_a2, 1]])

            A = C @ E @ D @ M @ F @ L @ B @ P @ R


            a11 = A[0, 0]
            a12 = A[0, 1]
            a21 = A[1, 0]
            a22 = A[1, 1]

            p9 = self.e_g / a11
            Up = p9 / Z_a2
            
            N = np.dot(np.dot(B, P), R)

            n11 = N[0, 0]
            n12 = N[0, 1]
            n21 = N[1, 0]
            n22 = N[1, 1]

            U6 = n21 * p9
            UB = (n21 - 1/Z_a2) * p9
            Up = self.e_g / (a11 * Z_a2)

            ig = a21 * p9

            U_ref = (1 * self.e_g * self.Bl * self.Sd) / ( 2 * np.pi * frequencies[i] * self.Mms * self.Re)

            #response[i] = 20 * np.log10(float(abs(U6)) / float(abs(U_ref)))

            response_2[i] = 20 * np.log10(float(abs(UB)) / float(abs(U_ref)))
            Ze[i] = abs(((a11) / (a21)))

        return response_2, Ze
            
      
      elif self.speaker_type == 'circular'  and self.port_shape == 'circular':
          
         # flow resistance of material equation 7.8 
          R_f = ((4*self.m*(1- self.porosity))/(self.porosity * self.r**2)) * ((1 - 4/np.pi * (1 - self.porosity))/(2 + np.log((self.m*self.porosity)/(2*self.r*self.r_0*self.u))) + (6/np.pi) * (1 - self.porosity))

          # box impedance based on equation 7.131
          def calculate_ZAB(f, lz, a, r_0, c, lx, ly, truncation_limit):

              # wave number k equation 7.11
              k = (2 * np.pi * f / c) * ((1 + self.a3 *(R_f/f)**self.b3) -1j *self.a4 * (R_f/f)**self.b4)

              Zs = r_0 * c + self.P_0/( 1j * 2 * np.pi * f * self.d)
              
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

              ZAB = (r_0 * c * ( (self.Sd*self.Sd)/(lx*ly) * (((Zs/(r_0*c)) + 1j * np.tan(k*lz)) / (1 + 1j * ((Zs/(r_0*c)) * np.tan(k*lz)))) + 4 * k*a*a*lx*ly* sum_mn)) / self.Sd**2   
              
              return  ZAB



          # Preallocate memory 
          response_1 = np.zeros_like(frequencies)
          Ze = np.zeros_like(frequencies)



          for i in range(len(frequencies)):
              
              # calculate the electrical impedance
              Z_e = self.Re + 1j * 2 * np.pi * frequencies[i] * self.Le
            
              # calculate the mechanical impedance
              Mmd = self.Mms - 16 * self.r_0 * self.a**3 / 3
              Z_md = 1j *  2 * np.pi * frequencies[i] * Mmd + self.Rms + 1 / (1j * 2 * np.pi * frequencies[i] * self.Cms)
              
              # wave number k equation 7.11
              k = (2 * np.pi * frequencies[i] / self.c) * ((1 + self.a3 *(R_f/frequencies[i])**self.b3) - 1j * self.a4 * (R_f/frequencies[i])**self.b4)
              
              # port radiation impedance based on equations 13.116 - 13.118
              R_s = self.r_0 * self.c * k**2 * self.a**2
              X_s = (self.r_0 * self.c * 8 * k * self.a)/(3 * np.pi)
              Z_a1 =  (R_s + 1j * X_s) 
              
              # calculate box impedance
              Z_ab = calculate_ZAB(frequencies[i], self.lz, self.a,self.r_0, self.c, self.lx, self.ly, self.truncation_limit)

              # calculate the leakage resistance
              CAA = (self.Va*10**(-3))/(1.4*self.P_0) # compliance of the air in the box
              CAM = (self.Vm*10**(-3))/self.P_0 # compliance of the air in the lining material
              Cab = CAA + 1.4 * CAM  # apparent compliance of the air in the box
              Ral = 7/ (2 * np.pi * self.fb * Cab)
              #print(Ral)


              kv = np.sqrt((-1j * np.pi *2 * frequencies[i] * self.r_0) / self.m)
              ap = np.sqrt((self.Sp) / np.pi)
              kvap = kv * ap
              
              #ξ = np.sqrt(1 - 1j * (2 * jv(1, kvap)) / (kvap * jv(0, kvap)))
              ξ = 0.998 + 0.001j
              kp = (2 * np.pi * frequencies[i]  * ξ) / self.c
              Zp = (self.r_0 * self.c * ξ) / (self.Sp)

              # calculate the port impedance
              R_s = self.r_0 * self.c * k**2 * self.a_p**2
              X_s = (self.r_0 * self.c * 8 * k * self.a_p)/(3 * np.pi)
              Z_a2 =  (R_s + 1j * X_s) 


              C = np.array([[1, 1* Z_e], [0, 1]])
              E = np.array([[0, 1* self.Bl], [1 / self.Bl, 0]])
              D = np.array([[1, 1*Z_md], [0, 1]])
              M = np.array([[1* self.Sd, 0], [0, 1 / self.Sd]])
              F = np.array([[1, 1*Z_a1], [0, 1]])
              L = np.array([[1, 0], [1/ Ral, 1]])
              B = np.array([[1, 0], [1 / Z_ab, 1]])
              P = np.array([[np.cos(kp*self.t), 1j*Zp*np.sin(kp*self.t)], [1j*(1/Zp)*np.sin(kp*self.t), np.cos(kp*self.t)]])
              R = np.array([[1, 0], [1/Z_a2, 1]])

              A = C @ E @ D @ M @ F @ L @ B @ P @ R


              a11 = A[0, 0]
              a12 = A[0, 1]
              a21 = A[1, 0]
              a22 = A[1, 1]

              p9 = self.e_g / a11
              Up = p9 / Z_a2
              
              N = np.dot(np.dot(B, P), R)

              n11 = N[0, 0]
              n12 = N[0, 1]
              n21 = N[1, 0]
              n22 = N[1, 1]

              U6 = n21 * p9
              UB = (n21 - 1/Z_a2) * p9
              Up = self.e_g / (a11 * Z_a2)

              ig = a21 * p9

              U_ref = (1 * self.e_g * self.Bl * self.Sd) / ( 2 * np.pi * frequencies[i] * self.Mms * self.Re)

              #response[i] = 20 * np.log10(float(abs(U6)) / float(abs(U_ref)))

              response_1[i] = 20 * np.log10(float(abs(UB)) / float(abs(U_ref)))
              Ze[i] = abs(((a11) / (a21)))


          return response_1, Ze
      




# Define a range of frequencies
octave_steps = 24
frequencies_per_octave = octave_steps * np.log(2)
min_frequency = 10
max_frequency = 10000
num_points = int(octave_steps * np.log2(max_frequency / min_frequency)) + 1
frequencies = np.logspace(np.log2(min_frequency), np.log2(max_frequency), num=num_points, base=2)

# Create an instance of the BassReflexEnclosure class
bass_reflex_enclosure_1 = BassReflexEnclosure('circular',  'rectangular')
bass_reflex_enclosure_2 = BassReflexEnclosure('circular', 'circular')


# Calculate the impedance response
response_1, Ze_1 = bass_reflex_enclosure_1.calculate_impedance_response(frequencies)
response_2 , Ze_2 = bass_reflex_enclosure_2.calculate_impedance_response(frequencies)




# plot on the same graph the response of the two cases
plt.figure()
plt.semilogx(frequencies, response_1 , label='1 Circ. Loudspeaker with Rect. port')
plt.semilogx(frequencies, response_2 , label='1 Circ. Loudspeaker with Circ. port')    
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
plt.xlabel('Frequency (Hz)')
plt.ylabel('Response (dB)')
plt.title('System Response')
plt.grid(which='both')
plt.legend()
plt.show()