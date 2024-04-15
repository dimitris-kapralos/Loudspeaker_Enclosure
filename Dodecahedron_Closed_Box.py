import numpy as np
from Loudspeakers_Enclosures import DodecahedronEnclosure
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

octave_steps = 24
frequencies_per_octave = octave_steps * np.log(2)
min_frequency = 10
max_frequency = 10000
num_points = int(octave_steps * np.log2(max_frequency / min_frequency)) + 1
frequencies = np.logspace(
    np.log2(min_frequency), np.log2(max_frequency), num=num_points, base=2
)

# Create instance of Loudspeaker and ClosedBoxEnclosure
enclosure1 = DodecahedronEnclosure(clb_parameters1, dod_parameters ,lsp_parameters)

# Calculate the system response
response, impedance, power, spl = enclosure1.calculate_dodecahedron_response(frequencies)


# Plot the system response
fig1, ax1 = plt.subplots()
ax1.semilogx(frequencies, response, label='1 loudspeaker')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('dB rel. Uref')
ax1.set_title('System Response')
ax1.grid(which='both')
ax1.legend()
plt.show()

# Plot the impedance
fig2, ax2 = plt.subplots()
ax2.semilogx(frequencies, impedance, label='1 loudspeaker')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Impedance (Ohms)')
ax2.set_title('System Impedance')
ax2.grid(which='both')
ax2.legend()
plt.show()

# Plot the power
fig3, ax3 = plt.subplots()
ax3.semilogx(frequencies, power, label='1 loudspeaker')
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('dB rel. 1pW')
ax3.set_title("Sound Power Lw")
ax3.grid(which='both')
ax3.legend()
plt.show()


# plot the sound power level
fig4, ax4 = plt.subplots()
ax4.semilogx(frequencies, spl, label='1 loudspeaker')
ax4.set_xlabel('Frequency (Hz)')
ax4.set_ylabel('dB rel. 20uPa')
ax4.set_title("Sound Pressure Lw")
ax4.grid(which='both')
ax4.legend()


