import numpy as np
import matplotlib.pyplot as plt
from Loudspeakers_Enclosures import DodecahedronBassReflexEnclosure
import pyswarms as ps
import itertools

from time import time

start_time = time()

# Define loudspeakers with their characteristics
loudspeakers = [
    {
        "name": "Speaker 1",
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
        "r" : 0.065,
    },
    {
        "name": "Speaker 2",
        "Re": 6.3,
        "Le": 0.000025,
        "e_g": 2.83,
        "Qes": 0.51,
        "Qms": 2.68,
        "fs": 55,
        "Sd": 0.0057,
        "Vas": 9.061,
        "Qts": 0.43,
        "Cms": 0.00107,
        "Mms": 0.004,
        "Bl": 4.19,
        "radius": 0.047,
        "height": 0.076,
        "r" : 0.065,
    },
    {
        "name": "Speaker 3",
        "Re": 6.5,
        "Le": 0.0004,
        "e_g": 2.83,
        "Qes": 0.5,
        "Qms": 8,
        "fs": 42,
        "Sd": 0.0093,
        "Vas": 15,
        "Qts": 0.47,
        "Cms": 0.00122,
        "Mms": 0.012,
        "Bl": 6.4,
        "radius": 0.045,
        "height": 0.06,
        "r": 0.05,
    },
    {
        "name": "Speaker 4",
        "Re": 5.9,
        "Le": 0.00078,
        "e_g": 2.83,
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
        "height": 0.068,
        "r": 0.05,
    },
    {
        "name": "Speaker 5",
        "Re": 3.7,
        "Le": 0.00076,
        "e_g": 2.83,
        "Qes": 0.66,
        "Qms": 7.42,
        "fs": 78.7,
        "Sd": 0.00515,
        "Vas": 1.78,
        "Qts": 0.61,
        "Cms": 0.00055,
        "Mms": 0.0074,
        "Bl": 4.52,
        "radius": 0.03,
        "height": 0.057,
        "r": 0.065,
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
    "r": 0.233,
    "d": 0.064,
    "Wmax": 100,
    "truncation_limit": 10,
}

# Define volumes of dodecahedron enclosures
dodecahedron_volumes = [
    {"Vab": 10, "rd": 0.13},
    {"Vab": 15 , "rd": 0.14},
    {"Vab": 20 , "rd": 0.15},
    {"Vab": 25 , "rd": 0.16},
    {"Vab": 30 , "rd": 0.18},
    {"Vab": 40 , "rd": 0.20},
    {"Vab": 50 , "rd": 0.23},
]

# Create an empty list to store all combinations of diode parameters
diode_parameters = []

t_range = [ 0.038, 0.028, 0.018, 0.023, 0.033]
radius_range = [0.005, 0.02, 0.0075, 0.01]
# Create all combinations of diode parameters
diode_parameters = [
    {"t": t, "radius": radius} for t, radius in itertools.product(t_range, radius_range)
]

# print("diode parameters"  ,diode_parameters)


# Define the number of ports
num_ports = [16, 8, 12, 20, 4]




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
frequencies = third_octave_bands(50, 8000, 3)

# Define the range of frequencies for calculation
frequencies = np.array(frequencies)

# Filter out frequencies outside the desired range
frequencies = frequencies[(frequencies >= 50) & (frequencies <= 7000)]


def calculate_sound_power(
    loudspeaker, clb_parameters, dodecahedron_volumes, diode_parameters, num_ports
):

    enclosure = DodecahedronBassReflexEnclosure(
        clb_parameters, dodecahedron_volumes, diode_parameters, loudspeaker
    )
    _, _, power, _ = enclosure.calculate_dodecahedron_bass_reflex_response(
        frequencies, num_ports, 3, 0.3
    )

    return power


lower_bound = np.array([0, 0, 0, 0])
upper_bound = np.array(
    [
        len(loudspeakers) - 1,
        len(dodecahedron_volumes) - 1,
        len(diode_parameters) - 1,
        len(num_ports) - 1,
    ]
)
bounds = (lower_bound, upper_bound)


def objective_function_pso(solution):
    # Iterate over each row of the solution array
    fitness_values = []
    for sol in solution:
        loudspeaker_idx = int(sol[0])
        volume_idx = int(sol[1])
        diode_param_idx = int(sol[2])
        ports_idx = int(sol[3])

        loudspeaker = loudspeakers[loudspeaker_idx]
        volume = dodecahedron_volumes[volume_idx]
        diode_param = diode_parameters[diode_param_idx]
        ports = num_ports[ports_idx]

        sound_power = calculate_sound_power(
            loudspeaker, clb_par, volume, diode_param, ports
        )
        
        for i in range(len(volume)):
            rd = volume['rd']  # Accessing the 'rd' key directly
            r = loudspeaker["r"]
            rho = np.sqrt(rd**2 + r**2)
            open_angle =  np.arccos( rd / rho ) * 180 / np.pi
            if open_angle < 23 or open_angle > 31.7:
                sound_power[i] = 0

        diff_from_120dB = np.abs(sound_power - 120)
        fitness = np.mean(diff_from_120dB)

        fitness_values.append(fitness)

    return np.array(fitness_values)


# Initialize PSO optimizer
options = {"c1": 2, "c2": 2, "w": 0.7}
optimizer = ps.single.GlobalBestPSO(
    n_particles=40, dimensions=4, options=options, bounds=bounds
)

# Perform optimization
start_time = time()
best_cost, best_solution = optimizer.optimize(objective_function_pso, iters=20)
end_time = time()


# Extract best solution indices
best_solution_indices = best_solution.astype(int)
best_speaker_idx, best_volume_idx, best_diode_idx, best_ports_idx = (
    best_solution_indices
)

best_speaker = loudspeakers[best_speaker_idx]["name"]
best_volume = dodecahedron_volumes[best_volume_idx]
best_diode = diode_parameters[best_diode_idx]
best_ports = num_ports[best_ports_idx]

print("Best solution:")
print("Loudspeaker:", best_speaker)
print("Volume:", best_volume)
print("Diode:", best_diode)
print("Ports:", best_ports)

# Calculate and print the open angle
rd = best_volume['rd']
r = loudspeakers[best_speaker_idx]["r"]
rho = np.sqrt(rd**2 + r**2)
open_angle = np.arccos(rd / rho) * 180 / np.pi
print("Open Angle:", open_angle)




power = calculate_sound_power(
    loudspeakers[best_speaker_idx], clb_par, best_volume, best_diode, best_ports
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


# Plot the bars with evenly spaced x-coordinates
plt.bar(np.arange(len(power_1_3_octave)), power_1_3_octave, width=bar_width, align='center')
plt.xticks(tick_indices, x_ticks)
plt.title("Best Sound Power")
plt.xlabel("Frequency (Hz)")
plt.ylabel('dB rel. 1pW')
plt.grid(which='both', axis='y')
plt.show()


# Print the optimized solution and fitness
print("Optimized solution:", best_solution_indices)
print("Best fitness:", best_cost)
print(f"Took: {end_time - start_time} seconds")


# Plot Fitness vs. Iterations
plt.plot(optimizer.cost_history)
plt.title("Best Fitness Value")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness Value")
plt.grid(True)
plt.show()
