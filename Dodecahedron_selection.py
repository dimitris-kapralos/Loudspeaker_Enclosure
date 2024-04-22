import numpy as np
import pygad
import matplotlib.pyplot as plt
from Loudspeakers_Enclosures import DodecahedronBassReflexEnclosure

# Define loudspeakers with their characteristics
loudspeakers = [
    {
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
    },
    {
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
    },
    {
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
    },
    {
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
    },
    {
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
    {"Vab": 10},
    {"Vab": 15},
    {"Vab": 20},
    {"Vab": 25},
    {"Vab": 30},
    {"Vab": 40},
    {"Vab": 50},
    {"Vab": 60},
]

# Define parameters for varying diode lengths and diameters
diode_parameters = [
    {"t": 0.01, "radius": 0.005},
    {"t": 0.01, "radius": 0.0075},
    {"t": 0.01, "radius": 0.01},
    {"t": 0.01, "radius": 0.02},
    {"t": 0.01, "radius": 0.025},
    {"t": 0.02, "radius": 0.005},
    {"t": 0.02, "radius": 0.0075},
    {"t": 0.02, "radius": 0.01},
    {"t": 0.02, "radius": 0.02},
    {"t": 0.02, "radius": 0.025},
    {"t": 0.04, "radius": 0.005},
    {"t": 0.04, "radius": 0.0075},
    {"t": 0.04, "radius": 0.01},
    {"t": 0.04, "radius": 0.02},
    {"t": 0.04, "radius": 0.025},
    {"t": 0.06, "radius": 0.005},
    {"t": 0.06, "radius": 0.0075},
    {"t": 0.06, "radius": 0.01},
    {"t": 0.06, "radius": 0.02},
    {"t": 0.06, "radius": 0.025},
    {"t": 0.08, "radius": 0.005},
    {"t": 0.08, "radius": 0.0075},
    {"t": 0.08, "radius": 0.01},
    {"t": 0.08, "radius": 0.02},
    {"t": 0.08, "radius": 0.025},
    # Add more diode parameters as needed
]

# Define the number of ports
num_ports = [5, 10, 15, 20, 25]


# Function to calculate sound power
def calculate_sound_power(
    loudspeaker, clb_parameters, dodecahedron_volumes, diode_parameters, num_ports
):
    octave_steps = 24
    min_frequency = 1
    max_frequency = 10000
    num_points = int(octave_steps * np.log2(max_frequency / min_frequency)) + 1
    frequencies = np.logspace(
        np.log2(min_frequency), np.log2(max_frequency), num=num_points, base=2
    )
    enclosure = DodecahedronBassReflexEnclosure(
        clb_parameters, dodecahedron_volumes, diode_parameters, loudspeaker
    )
    _, _, power, _ = enclosure.calculate_dodecahedron_bass_reflex_response(
        frequencies, num_ports, 1, 0.3
    )

    return power


# Define the fitness function for the genetic algorithm
def fitness_func(ag_instance, solution, solution_idx):
    # Convert float indices to integers
    loudspeaker_idx = int(solution[0])
    volume_idx = int(solution[1])
    diode_idx = int(solution[2])
    ports_idx = int(solution[3])

    loudspeaker = loudspeakers[loudspeaker_idx]

    volume = dodecahedron_volumes[volume_idx]
    diode_param = diode_parameters[diode_idx]
    ports = num_ports[ports_idx]

    sound_power = calculate_sound_power(
        loudspeaker, clb_par, volume, diode_param, ports
    )

    # Calculate the difference from 120 dB at all frequencies
    diff_from_120dB = np.abs(sound_power - 120)
    fitness = np.mean(diff_from_120dB)  # Use mean difference as fitness

    return 1 / (fitness + 1e-6)  # Return the inverse fitness (minimize difference)


# Define the bounds for the variables (loudspeaker, volume, diode, ports)
variable_boundaries = [
    (0, len(loudspeakers) - 1),  # Loudspeaker index
    (0, len(dodecahedron_volumes) - 1),  # Volume index
    (0, len(diode_parameters) - 1),  # Diode index
    (0, len(num_ports) - 1),  # Ports index
]

# Create the initial population for the genetic algorithm
initial_population = np.random.randint(0, len(loudspeakers), (20, 4))

# Create the genetic algorithm optimizer
ga_instance = pygad.GA(
    num_generations=20,
    num_parents_mating=10,
    fitness_func=fitness_func,
    sol_per_pop=10,
    num_genes=7,
    gene_space=variable_boundaries,
    initial_population=initial_population,
)

# Run the genetic algorithm optimization
ga_instance.run()

# Get the best solution
best_solution = ga_instance.best_solution()

# Print the best solution
best_solution_indices, best_solution_fitness, generation_number = best_solution
best_speaker_idx, best_volume_idx, best_diode_idx, best_ports_idx = (
    best_solution_indices.astype(int)
)

best_speaker = loudspeakers[best_speaker_idx]
best_volume = dodecahedron_volumes[best_volume_idx]
best_diode = diode_parameters[best_diode_idx]
best_ports = num_ports[best_ports_idx]

print("Best solution:")
print("Loudspeaker:", best_speaker)
print("Volume:", best_volume)
print("Diode:", best_diode)
print("Ports:", best_ports)

# Get the fitness values for each generation
fitness_values = ga_instance.best_solutions_fitness

# Plot the fitness values
plt.plot(fitness_values, marker="o")
plt.title("Fitness Progression")
plt.xlabel("Generation")
plt.ylabel("Fitness Value")
plt.grid(True)
plt.show()
