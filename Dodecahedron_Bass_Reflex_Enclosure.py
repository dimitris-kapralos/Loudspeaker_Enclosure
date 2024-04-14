import numpy as np
from Loudspeakers_Enclosures import DodecahedronBassReflexEnclosure
import matplotlib.pyplot as plt


lsp_parameters = {
    "Re": 6.72,
    "Le": 0.00067,
    "e_g": 2.83,
    "Qes": 0.522,
    "Qms": 2.2,
    "fs": 39,
    "Sd": 0.012,
    "Vas": 21.6,
    "Qts": 0.44,
    "Cms": 0.00107,
    "Mms": 0.0156,
    "Bl": 6.59,
    "radius": 0.45,
    "height": 0.7,
}

clb_parameters1 = {
    "number_of_speakers": 1,
    "lx": 0.25,
    "ly": 0.65,
    "lz": 0.19,
    "r" : 0.15,
    "d" : 0.064,
    "Wmax": 100,
    "truncation_limit" : 10  
}

dod_parameters = {  
    "Vab": 11.4,           
                  
}

dodbr_parameters = {
    "t": 0.04,
    "radius": 0.01,
}

octave_steps = 24
frequencies_per_octave = octave_steps * np.log(2)
min_frequency = 10
max_frequency = 10000
num_points = int(octave_steps * np.log2(max_frequency / min_frequency)) + 1
frequencies = np.logspace(
    np.log2(min_frequency), np.log2(max_frequency), num=num_points, base=2
)

# Create instance of Loudspeaker and ClosedBoxEnclosure
enclosure1 = DodecahedronBassReflexEnclosure(clb_parameters1, dod_parameters, dodbr_parameters, lsp_parameters)

# Calculate the system response
response, impedance, power, spl = enclosure1.calculate_dodecahedron_bass_reflex_response(frequencies)


# Plot the system response
plt.figure()
plt.semilogx(frequencies, response)
plt.xlabel('Frequency (Hz)')
plt.ylabel('dB rel. Uref')
plt.title('System Response')
plt.grid(which='both')
plt.show()

# Plot the impedance
plt.figure()
plt.semilogx(frequencies, impedance)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Impedance (Ohms)')
plt.title('System Impedance')
plt.grid(which='both')
plt.show()

# Plot the power
plt.figure()
plt.semilogx(frequencies, power, label='Docecahedron')

plt.xlabel('Frequency (Hz)')
plt.ylabel('dB rel. 1pW')
plt.title("Sound Power Lw")
plt.grid(which='both')
plt.show()


# plot the sound power level
plt.figure()
plt.semilogx(frequencies, spl, label='Dodecahedron')
plt.xlabel('Frequency (Hz)')
plt.ylabel('dB rel. 20uPa')     
plt.title('Sound Presure Level (SPL)')
plt.grid(which='both')
plt.show()


