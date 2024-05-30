import numpy as np
import matplotlib.pyplot as plt
from Loudspeakers_Enclosures import DodecahedronBassReflexEnclosure
from scipy.optimize import differential_evolution
from time import time


start_time = time()

# Define loudspeakers with their characteristics
loudspeakers = [
    {
        "name": " Dayton Audio PC105 -4 Ohm",
        "Re": 3.3,
        "Le": 0.00032,
        "e_g": 1.82,
        "Qes": 0.67,
        "Qms": 2.17,
        "fs": 81.4,
        "Sd": 0.0053,
        "Vas": 3.68,
        "Qts": 0.67,
        "Cms": 0.00092,
        "Mms": 0.0042,
        "Bl": 3.3,
        "radius": 0.035,
        "height": 0.044,
        "r": 0.065,
    },    
    {
        "name": "Visaton W100S-4 Ohm",
        "Re": 3.4,
        "Le": 0.0005,
        "e_g": 1.83,
        "Qes": 0.54,
        "Qms": 3.04,
        "fs": 37,
        "Sd": 0.0129,
        "Vas": 4.49,
        "Qts": 0.46,
        "Cms": 0.001127,
        "Mms": 0.053,
        "Bl": 3.7,
        "radius": 0.035,
        "height": 0.048,
        "r": 0.05,
    },
    {
        "name": "Dayton Audio DSA135-8",
        "Re": 5.9,
        "Le": 0.00078,
        "e_g": 2.43,
        "Qes": 0.47,
        "Qms": 2.04,
        "fs": 51.8,
        "Sd": 0.0075,
        "Vas": 7.93,
        "Qts": 0.38,
        "Cms": 0.00099,
        "Mms": 0.0095,
        "Bl": 6.22,
        "radius": 0.05,
        "height": 0.045,
        "r": 0.05,
    },
    {
        "name": "Visaton FR 10 - 8 Ohm",
        "Re": 7.2,
        "Le": 0.0002,
        "e_g": 2.68,
        "Qes": 0.95,
        "Qms": 2.29,
        "fs": 90,
        "Sd": 0.005,
        "Vas": 2.29,
        "Qts": 0.67,
        "Cms": 0.000648,
        "Mms": 0.0057,
        "Bl": 4.6,
        "radius": 0.035,
        "height": 0.044,
        "r": 0.065,
    },
    {
        "name": "Peerless by Tymphany 830656 5-1/4- 8",
        "Re": 6.1,
        "Le": 0.00037,
        "e_g": 2.47,
        "Qes": 0.7,
        "Qms": 3.6,
        "fs": 57,
        "Sd": 0.0087,
        "Vas": 8.78,
        "Qts": 0.59,
        "Cms": 0.00083,
        "Mms": 0.0093,
        "Bl": 5.36,
        "radius": 0.035,
        "height": 0.058,
        "r": 0.066,
    },
    {
        "name": "Epique E150HE-44 5-1/2 - 8",
        "Re": 6.6,
        "Le": 0.0016,
        "e_g": 2.6,
        "Qes": 0.45,
        "Qms": 2.94,
        "fs": 40,
        "Sd": 0.0095,
        "Vas": 7.65,
        "Qts": 0.39,
        "Cms": 0.0006,
        "Mms": 0.026,
        "Bl": 4.6,
        "radius": 0.06,
        "height": 0.055,
        "r": 0.069,
    },
    {
        "name": "Tang Band W5-1138SMF - 4",
        "Re": 3.4,
        "Le": 0.00034,
        "e_g": 1.84,
        "Qes": 0.57,
        "Qms": 3.56,
        "fs": 45,
        "Sd": 0.0094,
        "Vas": 4.8,
        "Qts": 0.49,
        "Cms": 0.00037,
        "Mms": 0.025,
        "Bl": 7.17,
        "radius": 0.05,
        "height": 0.081,
        "r": 0.065,
    },
    {
        "name": "Visaton SC 13 - 8 Ohm",
        "Re": 7.2,
        "Le": 0.0007,
        "e_g": 2.68,
        "Qes": 0.94,
        "Qms": 2.95,
        "fs": 78,
        "Sd": 0.0085,
        "Vas": 7.7,
        "Qts": 0.71,
        "Cms": 0.00163,
        "Mms": 0.0079,
        "Bl": 4.2,
        "radius": 0.033,
        "height": 0.051,
        "r": 0.065,
    },
    {
        "name": "Dayton Audio TCP115-4 - 4",
        "Re": 3.2,
        "Le": 0.00097,
        "e_g": 1.8,
        "Qes": 0.4,
        "Qms": 3.14,
        "fs": 53.8,
        "Sd": 0.005,
        "Vas": 3.11,
        "Qts": 0.35,
        "Cms": 0.00082,
        "Mms": 0.01,
        "Bl": 5.2,
        "radius": 0.04,
        "height": 0.06,
        "r": 0.05,
    },  
    {
        "name": "VISATON WF 130 ND - 8 Ohm  ",
        "Re": 6.7,
        "Le": 0.00076,
        "e_g": 2.58,
        "Qes": 0.89,
        "Qms": 2.97,
        "fs": 46,
        "Sd": 0.0094,
        "Vas": 10.6,
        "Qts": 0.69,
        "Cms": 0.000846,
        "Mms": 0.012,
        "Bl": 5.39,
        "radius": 0.044,
        "height": 0.019,
        "r": 0.065,
    },                   
    {
        "name": "Visaton W130s - 4 Ohm",
        "Re": 3.5,
        "Le": 0.00061,
        "e_g": 1.87,
        "Qes": 0.55,
        "Qms": 2.54,
        "fs": 52,
        "Sd": 0.0074,
        "Vas": 14,
        "Qts": 0.45,    
        "Cms": 0.0014,
        "Mms": 0.0054,
        "Bl": 3.6,
        "radius": 0.035,
        "height": 0.053,
        "r" : 0.065
    },
    {
        "name": "Dayton Audio CF120-4 Ohm  ",
        "Re": 3.4,
        "Le": 0.00045,
        "e_g": 1.84,
        "Qes": 0.32,
        "Qms": 1.89,
        "fs": 53,
        "Sd": 0.0051,
        "Vas": 4.8,
        "Qts": 0.28,
        "Cms": 0.001306,
        "Mms": 0.0068,
        "Bl": 4.94,
        "radius": 0.045,
        "height": 0.055,
        "r": 0.057,
    }, 
    {
        "name": "Dayton Audio ND140-8 Ohm  ",
        "Re": 7.8,
        "Le": 0.00144,
        "e_g": 2.8,
        "Qes": 0.79,
        "Qms": 5.01,
        "fs": 56,
        "Sd": 0.0087,
        "Vas": 7.74,
        "Qts": 0.68,
        "Cms": 0.00076,
        "Mms": 0.01,
        "Bl": 6.05,
        "radius": 0.045,
        "height": 0.055,
        "r": 0.057,
    },   
]

clb_par = {
    "number_of_speakers": 1,
    "lx": 0.25,
    "ly": 0.22,
    "lz": 0.19,
    "x": 0.075,
    "y1": 0.15,
    "y2": 0.15,
    "r": 0.18,
    "d": 0.064,
    "Wmax": 100,
    "truncation_limit": 10,
}

# Define volumes of icosidodecahedron enclosures
pentagon_edges = [
    {"edge": 0.1, "type":"icosidodecahedron", "rd": 0.137},
    {"edge": 0.11, "type":"icosidodecahedron", "rd": 0.151},  
    {"edge": 0.12, "type":"icosidodecahedron", "rd": 0.165},
    {"edge": 0.13, "type":"icosidodecahedron", "rd": 0.178},
    {"edge": 0.14, "type":"icosidodecahedron", "rd": 0.192},
    {"edge": 0.15, "type":"icosidodecahedron", "rd": 0.206},
    {"edge": 0.16, "type":"icosidodecahedron", "rd": 0.22},
    {"edge": 0.17, "type":"icosidodecahedron", "rd": 0.234},
    {"edge": 0.18, "type":"icosidodecahedron", "rd": 0.247},
]

# Define volumes of dodecahedron enclosures
# pentagon_edges  = [
#     {"edge": 0.1, "type":"dodecahedron", "rd": 0.111},
#     {"edge": 0.11, "type":"dodecahedron", "rd": 0.122},  
#     {"edge": 0.12, "type":"dodecahedron", "rd": 0.13},
#     {"edge": 0.13, "type":"dodecahedron", "rd": 0.134},
#     {"edge": 0.14, "type":"dodecahedron", "rd": 0.156},
#     {"edge": 0.15, "type":"dodecahedron", "rd": 0.167},
#     {"edge": 0.16, "type":"dodecahedron", "rd": 0.178},
#     {"edge": 0.17, "type":"dodecahedron", "rd": 0.19},
#     {"edge": 0.18, "type":"dodecahedron", "rd": 0.20},
# ]


# Define diode parameters
diode_parameters = [
    {"t": 0.023, "radius": 0.0079},
    {"t": 0.0247, "radius": 0.0105},
    {"t": 0.0265, "radius": 0.0133},
    {"t": 0.0292, "radius": 0.0175},
    {"t": 0.0311, "radius": 0.0205},
    {"t": 0.03575, "radius": 0.0079},
    {"t": 0.0480, "radius": 0.0105},
    {"t": 0.0456, "radius": 0.0133},
    {"t": 0.0525, "radius": 0.0175},
    {"t": 0.0578, "radius": 0.0205},
    {"t": 0.04075, "radius": 0.0079},
    {"t": 0.0458, "radius": 0.0105},
    {"t": 0.0506, "radius": 0.0133},
    {"t": 0.0577, "radius": 0.0175},
    {"t": 0.0628, "radius": 0.0205},
    {"t": 0.0464, "radius": 0.0079},
    {"t": 0.0508, "radius": 0.0105},
    {"t": 0.0556, "radius": 0.0133},
    {"t": 0.0627, "radius": 0.0175},
    {"t": 0.0678, "radius": 0.0205},
    {"t": 0.0514, "radius": 0.0079},
    {"t": 0.0558, "radius": 0.0105},
    {"t": 0.0606, "radius": 0.0133},
    {"t": 0.0677, "radius": 0.0175},
    {"t": 0.0728, "radius": 0.0205},
    {"t": 0.00000000000001, "radius": 0.00000000000001},
]


# Define the number of ports
num_ports = [ 4, 8, 12, 16, 20]



# Function to calculate 1/3 octave bands
def third_octave_bands(start_freq, stop_freq, num_bands_per_octave):
    bands = []
    f1 = start_freq
    while f1 < stop_freq:
        bands.append(f1)
        for i in range(1, num_bands_per_octave):
            bands.append(f1 * (2 ** (i / (3 * num_bands_per_octave))))
        f1 *= 2 ** (1 / 3)
    bands.append(stop_freq)
    return bands


# Define the central frequencies based on provided values
central_frequencies = [
    50, 63, 80, 100, 125, 160, 200, 250, 315, 400,
    500, 630, 800, 1000, 1250, 1600, 2000, 2500, 
    3150, 4000, 5000, 6300, 8000
]

# Calculate third octave bands using the provided central frequencies
frequencies = third_octave_bands(50, 500, 3)

# Define the range of frequencies for calculation
frequencies = np.array(frequencies)

# Filter out frequencies outside the desired range
frequencies = frequencies[(frequencies >= 50) & (frequencies <= 500)]


def calculate_sound_power(
    loudspeaker, clb_parameters, pentagon_edges, diode_parameters, num_ports
):

    enclosure = DodecahedronBassReflexEnclosure(
        clb_parameters, pentagon_edges, diode_parameters, loudspeaker
    )
    _, _, power, _ = enclosure.calculate_dodecahedron_bass_reflex_response(
        frequencies, num_ports, 3, 0.3
    )

    return power


# Define the fitness function for the genetic algorithm
def objective_function(solution):
    # Convert float indices to integers
    loudspeaker_idx = int(solution[0])
    edge_idx = int(solution[1])
    diode_param_idx = int(solution[2])
    ports_idx = int(solution[3])

    loudspeaker = loudspeakers[loudspeaker_idx]
    edge = pentagon_edges[edge_idx]
    diode_param = diode_parameters[diode_param_idx]  # Corrected this line
    ports = num_ports[ports_idx]

    sound_power = calculate_sound_power(
        loudspeaker, clb_par, edge, diode_param, ports
    )
    
    for i in range(len(edge)):
        rd = edge['rd']  # Accessing the 'rd' key directly
        r = loudspeaker["r"]
        rho = np.sqrt(rd**2 + r**2)
        open_angle =  np.arccos( rd / rho ) * 180 / np.pi
        if open_angle < 22 or open_angle > 26.6:
            sound_power[i] = 0    

    # Calculate the difference from 120 dB at all frequencies
    diff_from_120dB = np.abs(sound_power - 120)
    fitness = np.mean(diff_from_120dB)  # Use mean difference as fitness

    return fitness


# Define the bounds for the search space
bounds = [
    (0, len(loudspeakers) - 1),  # Loudspeaker index
    (0, len(pentagon_edges) - 1),  # Volume index
    (0, len(diode_parameters) - 1),  # Diode index
    (0, len(num_ports) - 1),
]  # Ports index


fitness_evolution = []


def callback(xk, convergence):
    fitness_evolution.append(objective_function(xk))


# Use differential evolution for optimization
result = differential_evolution(
    objective_function,
    bounds,
    strategy="randtobest1bin",
    maxiter=31,  # Increase the number of iterations
    popsize=20,
    mutation=(0.5, 1),
    polish=True,
    tol=1e-5,
    callback=callback,
    disp=True,
)


# Extract the optimized solution
optimal_solution = result.x.astype(int)
best_fitness = result.fun

best_solution_indices = optimal_solution
best_speaker_idx, best_edge_idx, best_diode_idx, best_ports_idx = (
    best_solution_indices.astype(int)
)

best_speaker = loudspeakers[best_speaker_idx]["name"]
best_edge = pentagon_edges[best_edge_idx]
best_diode = diode_parameters[best_diode_idx]
best_ports = num_ports[best_ports_idx]


print("Best solution:")
print("Loudspeaker:", best_speaker)
print("Volume:", best_edge)
print("Diode:", best_diode)
print("Ports:", best_ports)

# Calculate and print the open angle
rd = best_edge['rd']
r = loudspeakers[best_speaker_idx]["r"]
rho = np.sqrt(rd**2 + r**2)
open_angle = np.arccos(rd / rho) * 180 / np.pi
print("Open Angle:", open_angle)



power = calculate_sound_power(
    loudspeakers[best_speaker_idx], clb_par, best_edge, best_diode, best_ports
)

# Calculate central frequencies of 1/3 octave bands
power_1_3_octave = np.zeros(len(central_frequencies) - 1)


for j in range(len(central_frequencies) - 1):
    start_idx = np.argmax(frequencies >= central_frequencies[j])
    stop_idx = np.argmax(frequencies >= central_frequencies[j + 1])

    power_1_3_octave[j] = np.mean(power[start_idx:stop_idx])


# Define the number of bars
num_bars = len(central_frequencies) - 1

# Create evenly spaced x-coordinates for the bars
x_values = np.linspace(0, num_bars, num_bars)

# Calculate the bar width based on the number of bars
bar_width = 0.8

# Define the specific x-axis values to display
x_ticks = [50, 100, 200, 400, 800, 1600, 3150]

# Create an array to store the indices of central frequencies that correspond to the specified x_ticks
tick_indices = [central_frequencies.index(x) for x in x_ticks]

# Calculate the bar width based on the number of bars
bar_width = 0.8

# Plot the bars with evenly spaced x-coordinates
plt.bar(np.arange(len(power_1_3_octave)), power_1_3_octave, width=bar_width, align='center')

# Set the x-tick locations and labels to the specified x_ticks
plt.xticks(tick_indices, x_ticks)

plt.title("Best Sound Power")
plt.xlabel("Frequency (Hz)")
plt.ylabel('dB rel. 1pW')
plt.grid(which='both', axis='y')

plt.show()


# Print the optimized solution and fitness
print("Optimized solution:", optimal_solution)
print("Best fitness:", best_fitness)

# Extract the optimized parameters
optimal_loudspeaker = loudspeakers[optimal_solution[0]]
optimal_edge = pentagon_edges[optimal_solution[1]]
optimal_diode_param = diode_parameters[optimal_solution[2]]
optimal_ports = num_ports[optimal_solution[3]]

# Calculate sound power with the optimized parameters
optimal_sound_power = calculate_sound_power(
    optimal_loudspeaker, clb_par, optimal_edge, optimal_diode_param, optimal_ports
)

# Print the optimized sound power
print(f"Took: {time() - start_time} seconds")


# Plot the fitness through evolution
plt.plot(np.arange(len(fitness_evolution)), fitness_evolution)
plt.xlabel("Iteration")
plt.ylabel("Best Fitness Value")
plt.title("Fitness Through Evolution")
plt.grid(True)
plt.show()
