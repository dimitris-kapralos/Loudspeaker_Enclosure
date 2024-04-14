import numpy as np
from Loudspeakers_Enclosures import Loudspeaker, ClosedBoxEnclosure
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
    "radius": 0.15,
    "height": 0.064,
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
clb_parameters2 = {
    "number_of_speakers": 2,
    "lx": 0.25,
    "ly": 0.65,
    "lz": 0.19,
    "r" : 0.15,
    "d" : 0.064,
    "Wmax": 100,
    "truncation_limit" : 10  
}
clb_parameters3 = {
    "number_of_speakers": 3,
    "lx": 0.25,
    "ly": 0.65,
    "lz": 0.19,
    "r" : 0.15,
    "d" : 0.064,
    "Wmax": 100,
    "truncation_limit" : 10  
}
clb_parameters4 = {
    "number_of_speakers": 4,
    "lx": 0.25,
    "ly": 0.65,
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
power, spl = loudspeaker.calculate_loudspeaker_response(frequencies)

enclosure1 = ClosedBoxEnclosure(clb_parameters1, lsp_parameters)
enclosure2 = ClosedBoxEnclosure(clb_parameters2, lsp_parameters)
enclosure3 = ClosedBoxEnclosure(clb_parameters3, lsp_parameters)
enclosure4 = ClosedBoxEnclosure(clb_parameters4, lsp_parameters)

# Calculate the system response
response1, impedance1, power1, spl1 = enclosure1.calculate_closed_box_response(frequencies , number_of_speakers=1)
response2, impedance2, power2, spl2 = enclosure2.calculate_closed_box_response(frequencies , number_of_speakers=2)
response3, impedance3, power3, spl3 = enclosure3.calculate_closed_box_response(frequencies , number_of_speakers=3)
response4, impedance4, power4, spl4 = enclosure4.calculate_closed_box_response(frequencies , number_of_speakers=4)

# Plot the system response
plt.figure()
plt.semilogx(frequencies, response1)
plt.semilogx(frequencies, response2)
plt.semilogx(frequencies, response3)
plt.semilogx(frequencies, response4)
plt.xlabel('Frequency (Hz)')
plt.ylabel('dB rel. Uref')
plt.title('System Response')
plt.grid(which='both')
plt.show()

# Plot the impedance
plt.figure()
plt.semilogx(frequencies, impedance1)
plt.semilogx(frequencies, impedance2)
plt.semilogx(frequencies, impedance3)
plt.semilogx(frequencies, impedance4)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Impedance (Ohms)')
plt.title('System Impedance')
plt.grid(which='both')
plt.show()

# Plot the power
plt.figure()
plt.semilogx(frequencies, power, label='1 loudspeaker in infinite baffle')
# plt.semilogx(frequencies, power2)
# plt.semilogx(frequencies, power3)
# plt.semilogx(frequencies, power4)
plt.xlabel('Frequency (Hz)')
plt.ylabel('dB rel. 1pW')
plt.title("Sound Power Lw")
plt.grid(which='both')
plt.show()


# plot the sound power level
plt.figure()
plt.semilogx(frequencies, spl, label='1 loudspeaker in infinite baffle')
# plt.semilogx(frequencies, spl2)
# plt.semilogx(frequencies, spl3)
# plt.semilogx(frequencies, spl4)
plt.xlabel('Frequency (Hz)')
plt.ylabel('dB rel. 20uPa')     
plt.title('Sound Presure Level (SPL)')
plt.grid(which='both')
plt.show()


