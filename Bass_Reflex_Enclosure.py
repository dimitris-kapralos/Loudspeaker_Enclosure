""" Description: This script calculates the system response, impedance and power of a bass reflex enclosure with multiple loudspeakers. The loudspeaker and enclosure parameters are defined in the script. The system response, impedance and power are plotted in separate figures. """

import numpy as np
from Loudspeakers_Enclosures import Loudspeaker, BassReflexEnclosure
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
    "r": 0.045,
}

clb_parameters1 = {
    "number_of_speakers": 1,
    "lx": 0.15,
    "ly": 0.35,
    "lz": 0.19,
    "x": 0.075,
    "y1": 0.15,
    "y2": 0.3,
    "r": 0.15,
    "d": 0.064,
    "Wmax": 100,
    "truncation_limit": 10,
}
clb_parameters2 = {
    "number_of_speakers": 2,
    "lx": 0.15,
    "ly": 0.63,
    "lz": 0.19,
    "x": 0.1,
    "y1": 0.095,
    "y2": 0.52,
    "r": 0.15,
    "d": 0.064,
    "Wmax": 100,
    "truncation_limit": 10,
}
clb_parameters3 = {
    "number_of_speakers": 3,
    "lx": 0.2,
    "ly": 0.671,
    "lz": 0.19,
    "x": 0.1,
    "y1": 0.33,
    "y2": 0.6,
    "r": 0.15,
    "d": 0.064,
    "Wmax": 100,
    "truncation_limit": 10,
}
clb_parameters4 = {
    "number_of_speakers": 4,
    "lx": 0.2,
    "ly": 0.894,
    "lz": 0.19,
    "x": 0.1,
    "y1": 0.4,
    "y2": 0.8,
    "r": 0.15,
    "d": 0.064,
    "Wmax": 100,
    "truncation_limit": 10,
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
response1, impedance1, power1, spl1 = enclosure1.calculate_bass_reflex_response(
    frequencies, number_of_speakers=1
)
response2, impedance2, power2, spl2 = enclosure2.calculate_bass_reflex_response(
    frequencies, number_of_speakers=2
)
response3, impedance3, power3, spl3 = enclosure3.calculate_bass_reflex_response(
    frequencies, number_of_speakers=3
)
response4, impedance4, power4, spl4 = enclosure4.calculate_bass_reflex_response(
    frequencies, number_of_speakers=4
)

# Plot the system response
fig1, ax1 = plt.subplots()
ax1.semilogx(frequencies, response1, label="1 loudspeaker")
ax1.semilogx(frequencies, response2, label="2 loudspeakers")
ax1.semilogx(frequencies, response3, label="3 loudspeakers")
ax1.semilogx(frequencies, response4, label="4 loudspeakers")
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("dB rel. Uref")
ax1.set_title("System Response")
ax1.grid(which="both")
ax1.legend()
plt.show()

# Plot the impedance
fig2, ax2 = plt.subplots()
ax2.semilogx(frequencies, impedance1, label="1 loudspeaker")
ax2.semilogx(frequencies, impedance2, label="2 loudspeakers")
ax2.semilogx(frequencies, impedance3, label="3 loudspeakers")
ax2.semilogx(frequencies, impedance4, label="4 loudspeakers")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Impedance (Ohms)")
ax2.set_title("System Impedance")
ax2.grid(which="both")
ax2.legend()
plt.show()

# Plot the power
fig3, ax3 = plt.subplots()
ax3.semilogx(frequencies, power1, label="1 loudspeaker")
ax3.semilogx(frequencies, power2, label="2 loudspeakers")
ax3.semilogx(frequencies, power3, label="3 loudspeakers")
ax3.semilogx(frequencies, power4, label="4 loudspeakers")
ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("dB rel. 1pW")
ax3.set_title("Sound Power Lw")
ax3.grid(which="both")
ax3.legend()
plt.show()


# plot the sound power level
fig4, ax4 = plt.subplots()
ax4.semilogx(frequencies, spl1, label="1 loudspeaker")
ax4.semilogx(frequencies, spl2, label="2 loudspeakers")
ax4.semilogx(frequencies, spl3, label="3 loudspeakers")
ax4.semilogx(frequencies, spl4, label="4 loudspeakers")
ax4.set_xlabel("Frequency (Hz)")
ax4.set_ylabel("dB rel. 20uPa")
ax4.set_title("Sound Pressure Lw")
ax4.grid(which="both")
ax4.legend()
