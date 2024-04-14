""" Description: This script calculates the system response, impedance and power of a bass reflex enclosure with a single loudspeaker. The loudspeaker and enclosure parameters are defined in the script. The system response, impedance and power are plotted in separate figures. """

import numpy as np
from Loudspeakers_Enclosures import Loudspeaker,  BassReflexEnclosure
import matplotlib.pyplot as plt


lsp_parameters = {
    "Re": 6.27,
    "Le": 0.00067,
    "e_g": 2.83,
    "Qes": 0.522,
    "Qms": 1.9,
    "fs": 37,
    "Sd": 0.012,
    "Vas": 24,
    "Qts": 0.41,
    "Cms": 0.00119,
    "Mms": 0.0156,
    "Bl": 6.59,
    "radius": 0.15,
    "height": 0.064,

}

clb_parameters1 = {
    "number_of_speakers": 1,
    "lx": 0.15,
    "ly": 0.35,
    "lz": 0.19,
    "r" : 0.15,
    "d" : 0.064,
    "Wmax": 100,
    "truncation_limit" : 10  
}

clb_parameters2 = {
    "number_of_speakers": 2,
    "lx": 0.15,
    "ly": 0.35,
    "lz": 0.19,
    "r" : 0.15,
    "d" : 0.064,
    "Wmax": 100,
    "truncation_limit" : 10  
}
clb_parameters3 = {
    "number_of_speakers": 3,
    "lx": 0.15,
    "ly": 0.35,
    "lz": 0.19,
    "r" : 0.15,
    "d" : 0.064,
    "Wmax": 100,
    "truncation_limit" : 10  
}

clb_parameters4 = {
    "number_of_speakers": 4,
    "lx": 0.15,
    "ly": 0.35,
    "lz": 0.19,
    "r" : 0.15,
    "d" : 0.064,
    "Wmax": 100,
    "truncation_limit" : 10  
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
loudspeaker = Loudspeaker(lsp_parameters)
enclosure1 = BassReflexEnclosure(clb_parameters1, lsp_parameters)
enclosure2 = BassReflexEnclosure(clb_parameters2, lsp_parameters)
enclosure3 = BassReflexEnclosure(clb_parameters3, lsp_parameters)
enclosure4 = BassReflexEnclosure(clb_parameters4, lsp_parameters)

# Calculate the system response
response1, impedance1, power1 = enclosure1.calculate_bass_reflex_response(frequencies , number_of_speakers=1)
response2, impedance2, power2 = enclosure2.calculate_bass_reflex_response(frequencies , number_of_speakers=2)
response3, impedance3, power3 = enclosure3.calculate_bass_reflex_response(frequencies , number_of_speakers=3)
response4, impedance4, power4 = enclosure4.calculate_bass_reflex_response(frequencies , number_of_speakers=4)

# Plot the system response
plt.figure()
plt.semilogx(frequencies, response1, label='1 loudspeaker with port')
plt.semilogx(frequencies, response2, label='2 loudspeakers with port')
plt.semilogx(frequencies, response3, label='3 loudspeakers with port')
plt.semilogx(frequencies, response4, label='4 loudspeakers with port')
plt.xlabel('Frequency (Hz)')
plt.ylabel('dB rel. Uref')
plt.title('System Response')
plt.grid(which='both')
plt.show()

# Plot the impedance
plt.figure()
plt.semilogx(frequencies, impedance1, label='1 loudspeaker with port')
plt.semilogx(frequencies, impedance2, label='2 loudspeakers with port')
plt.semilogx(frequencies, impedance3, label='3 loudspeakers with port')
plt.semilogx(frequencies, impedance4, label='4 loudspeakers with port')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Impedance (Ohms)')
plt.title('System Impedance')
plt.grid(which='both')
plt.show()

# Plot the power
plt.figure()
plt.semilogx(frequencies, power1, label='1 loudspeaker with port')
plt.semilogx(frequencies, power2, label='2 loudspeakers with port')
plt.semilogx(frequencies, power3, label='3 loudspeakers with port')
plt.semilogx(frequencies, power4, label='4 loudspeakers with port')
plt.xlabel('Frequency (Hz)')
plt.ylabel('dB rel. 1pW')
plt.title("Sound Power Lw")
plt.grid(which='both')
plt.show()