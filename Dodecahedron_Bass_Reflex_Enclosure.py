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
    "radius": 0.045,
    "height": 0.09,
}

clb_parameters0 = {
    "number_of_speakers": 1,
    "lx": 0.17,
    "ly": 0.1,
    "lz": 0.19,
    "x": 0.075,
    "y1": 0.15,
    "y2": 0.15,
    "r" : 0.14,
    "d" : 0.064,
    "Wmax": 100,
    "truncation_limit" : 10  
}

clb_parameters1 = {
    "number_of_speakers": 1,
    "lx": 0.11,
    "ly": 0.2,
    "lz": 0.19,
    "x": 0.075,
    "y1": 0.15,
    "y2": 0.15,
    "r" : 0.16,
    "d" : 0.064,
    "Wmax": 100,
    "truncation_limit" : 10  
}

clb_parameters2 = {
    "number_of_speakers": 1,
    "lx": 0.175,
    "ly": 0.2,
    "lz": 0.19,
    "x": 0.075,
    "y1": 0.15,
    "y2": 0.15,
    "r" : 0.21,
    "d" : 0.064,
    "Wmax": 100,
    "truncation_limit" : 10  
}

clb_parameters3 = {
    "number_of_speakers": 1,
    "lx": 0.2,
    "ly": 0.22,
    "lz": 0.19,
    "x": 0.075,
    "y1": 0.15,
    "y2": 0.15,
    "r" : 0.227,
    "d" : 0.064,
    "Wmax": 100,
    "truncation_limit" : 10  
}

clb_parameters4 = {
    "number_of_speakers": 1,
    "lx": 0.25,
    "ly": 0.22,
    "lz": 0.19,
    "x": 0.075,
    "y1": 0.15,
    "y2": 0.15,
    "r" : 0.25,
    "d" : 0.064,
    "Wmax": 100,
    "truncation_limit" : 10  
}

dod_parameters0 = {
    "Vab": 7.64,
}

dod_parameters1 = {  
    "Vab": 11.4,           
                  
}

dod_parameters2 = {
    "Vab": 22.28,
}

dod_parameters3 = {
    "Vab": 31.72,
}

dod_parameters4 = { 
    "Vab": 43.52,
}

dodbr_parameters1 = {
    "t": 0.04,
    "radius": 0.01,
}

dodbr_parameters2 = {
    "t": 0.04,
    "radius": 0.01,
}

dodbr_parameters3 = {
    "t": 0.04,
    "radius": 0.01
}

dodbr_parameters4 = {
    "t": 0.04,
    "radius": 0.01
}

dodbr_parameters5 = {
    "t": 0.04,
    "radius": 0.01,
}

octave_steps = 24
frequencies_per_octave = octave_steps * np.log(2)
min_frequency = 1
max_frequency = 10000
num_points = int(octave_steps * np.log2(max_frequency / min_frequency)) + 1
frequencies = np.logspace(
    np.log2(min_frequency), np.log2(max_frequency), num=num_points, base=2
)

# Create instance of Loudspeaker and ClosedBoxEnclosure
enclosure0 = DodecahedronBassReflexEnclosure(clb_parameters0, dod_parameters0, dodbr_parameters1, lsp_parameters)
enclosure1 = DodecahedronBassReflexEnclosure(clb_parameters1, dod_parameters1, dodbr_parameters2, lsp_parameters)
enclosure2 = DodecahedronBassReflexEnclosure(clb_parameters2, dod_parameters2, dodbr_parameters3, lsp_parameters)
enclosure3 = DodecahedronBassReflexEnclosure(clb_parameters3, dod_parameters3, dodbr_parameters4, lsp_parameters)
enclosure4 = DodecahedronBassReflexEnclosure(clb_parameters4, dod_parameters4, dodbr_parameters5, lsp_parameters)

# Calculate the system response
response0, impedance0, power0, spl0 = enclosure0.calculate_dodecahedron_bass_reflex_response(frequencies, 20, 1, 0.3)
response1, impedance1, power1, spl1 = enclosure1.calculate_dodecahedron_bass_reflex_response(frequencies, 20, 1, 0.33)
response2, impedance2, power2, spl2 = enclosure2.calculate_dodecahedron_bass_reflex_response(frequencies, 20, 1, 0.42)
response3, impedance3, power3, spl3 = enclosure3.calculate_dodecahedron_bass_reflex_response(frequencies, 20, 1, 0.45)
response4, impedance4, power4, spl4 = enclosure4.calculate_dodecahedron_bass_reflex_response(frequencies, 20, 1, 0.5)


# Plot the system response
fig1, ax1 = plt.subplots()
ax1.semilogx(frequencies, response0, label='Diameter = 1 cm')
ax1.semilogx(frequencies, response1, label='Diameter = 1.5 cm')
ax1.semilogx(frequencies, response2, label='Diameter = 2 cm')
ax1.semilogx(frequencies, response3, label='Diameter = 4 cm')
ax1.semilogx(frequencies, response4, label='Diameter = 5 cm')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('dB rel. Uref')
ax1.set_title('System Response')
ax1.grid(which='both')
ax1.legend()
plt.show()

# Plot the impedance
fig2, ax2 = plt.subplots()
ax2.semilogx(frequencies, impedance0, label='Diameter = 1 cm')
ax2.semilogx(frequencies, impedance1, label='Diameter = 1.5 cm')
ax2.semilogx(frequencies, impedance2, label='Diameter = 2 cm')
ax2.semilogx(frequencies, impedance3, label='Diameter = 4 cm')
ax2.semilogx(frequencies, impedance4, label='Diameter = 5 cm')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Impedance (Ohms)')
ax2.set_title('System Impedance')
ax2.grid(which='both')
ax2.set_ylim(0, 80)
ax2.legend()
plt.show()

# Plot the power
fig3, ax3 = plt.subplots()
ax3.semilogx(frequencies, power0, label='V = 7.64 L')
ax3.semilogx(frequencies, power1, label='V = 11.4 L')
ax3.semilogx(frequencies, power2, label='V = 22.28 L')
ax3.semilogx(frequencies, power3, label='V = 31.72 L')
ax3.semilogx(frequencies, power4, label='V = 43.52 L')
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('dB rel. 1pW')
ax3.set_title("Sound Power Lw")
ax3.grid(which='both')
ax3.legend()
plt.show()


# plot the sound power level
# fig4, ax4 = plt.subplots()
# ax4.semilogx(frequencies, spl, label='12 loudspeakers')
# ax4.set_xlabel('Frequency (Hz)')
# ax4.set_ylabel('dB rel. 20uPa')
# ax4.set_title("Sound Pressure Level (SPL)")
# ax4.grid(which='both')
# ax4.legend()
# plt.show()


