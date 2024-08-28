import numpy as np
import pygad
import random
import matplotlib.pyplot as plt
from Loudspeakers_Enclosures import (
    DodecahedronBassReflexEnclosure,
    DodecahedronEnclosure,
)
import parameters as par
from time import time


def run_pygad(loudspeakers, pentagon_edges, port_params, num_of_ports, configuration):

    start_time = time()

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
        50,
        63,
        80,
        100,
        125,
        160,
        200,
        250,
        315,
        400,
        500,
        630,
        800,
        1000,
        1250,
        1600,
        2000,
        2500,
        3150,
        4000,
        5000,
        6300,
        8000,
    ]

    # Calculate third octave bands using the provided central frequencies
    frequencies = third_octave_bands(50, 2000, 3)
    frequencies = np.array(frequencies)
    frequencies = frequencies[(frequencies >= 50) & (frequencies <= 2000)]

    def calculate_sound_power(
        loudspeaker, clb_parameters, pentagon_edge, port_params, num_of_ports, configuration
    ):
        if configuration == "bass_reflex":
            enclosure = DodecahedronBassReflexEnclosure(
                clb_parameters, pentagon_edge, port_params, loudspeaker
            )
            _, _, _, _, lw, _, power = (
                enclosure.calculate_dodecahedron_bass_reflex_response(
                    frequencies, num_of_ports, 3, 0.3
                )
            )
        elif configuration == "closed_box":
            enclosure = DodecahedronEnclosure(
                clb_parameters, pentagon_edge, loudspeaker
            )
            _, _, _, lw, power = enclosure.calculate_dodecahedron_response(
                frequencies, 3, 0.3
            )
        return lw, power

    # Define the fitness function for the genetic algorithm
    def fitness_func(ag_instance, solution, solution_idx):
        fitness_values = []
        loudspeaker_idx = int(solution[0])
        edge_idx = int(solution[1])
        if configuration == "bass_reflex":
            port_param_idx = int(solution[2])
            ports_idx = int(solution[3])

            loudspeaker = loudspeakers[loudspeaker_idx]
            edge = pentagon_edges[edge_idx]
            port_param = port_params[port_param_idx]
            ports = num_of_ports[ports_idx]

            lw, power = calculate_sound_power(
                loudspeaker, par.clb_par, edge, port_param, ports, configuration
            )
        elif configuration == "closed_box":
            loudspeaker = loudspeakers[loudspeaker_idx]
            edge = pentagon_edges[edge_idx]

            lw, power = calculate_sound_power(
                loudspeaker, par.clb_par, edge, None, None, configuration
            )

        for i in range(len(frequencies)):
            rd = edge["rd"]
            r = loudspeaker["r"]
            rho = np.sqrt(rd**2 + r**2)
            open_angle = np.arccos(rd / rho) * 180 / np.pi
            if edge["type"] == "dodecahedron" and (open_angle < 21 or open_angle > 28):
                power[i] = -np.inf
                lw[i] = -np.inf
            elif edge["type"] == "icosidodecahedron" and (
                open_angle < 20 or open_angle > 26.6
            ):
                power[i] = -np.inf
                lw[i] = -np.inf

        max_power_index = np.argmax(power)
        max_power_freq = frequencies[max_power_index]

        # Calculate the sum of power up to the max power frequency
        up_to_peak_indices = frequencies <= max_power_freq
        power_up_to_peak = np.sum(power[up_to_peak_indices])

        # Calculate the sum of power in the same number of 1/3 octave bands after the peak
        num_bands_before_peak = np.sum(up_to_peak_indices)
        # include the peak itself
        remaining_power = power[max_power_index:]
        after_peak_power = remaining_power[:num_bands_before_peak]
        power_after_peak = np.sum(after_peak_power) 

        # Calculate the ratio
        power_ratio = power_up_to_peak / power_after_peak 
        log_power_ratio = 10 * np.log10(power_ratio) if power_ratio > 0 else -np.inf

        # total_power = np.sum(lw) 
        avarage_power = np.mean(lw)
        
        total = 0.5 * avarage_power + 0.5 * log_power_ratio
        
        fitness_values.append((total))

        return fitness_values

    if configuration == "bass_reflex":
        variable_boundaries = [
            (0, len(loudspeakers) - 1),  # Loudspeaker index
            (0, len(pentagon_edges) - 1),  # Volume index
            (0, len(port_params) - 1),  # port index
            (0, len(num_of_ports) - 1),  # Ports index
        ]
        num_genes = 4
    elif configuration == "closed_box":
        variable_boundaries = [
            (0, len(loudspeakers) - 1),  # Loudspeaker index
            (0, len(pentagon_edges) - 1),  # Volume index
        ]
        num_genes = 2

    # Define the number of samples you want to generate
    num_samples = 150

    # Generate random samples of combinations
    random_samples = []
    for _ in range(num_samples):
        sample = [random.randint(bound[0], bound[1]) for bound in variable_boundaries]
        random_samples.append(sample)

    # Convert to numpy array for compatibility with PyGAD
    initial_population = np.array(random_samples)

    # Create the genetic algorithm optimizer
    ga_instance = pygad.GA(
        num_generations=30,
        num_parents_mating=20,
        fitness_func=fitness_func,
        num_genes=num_genes,
        gene_space=variable_boundaries,
        initial_population=initial_population,
        crossover_type="two_points",
        mutation_type="random",
        mutation_probability=0.25,
    )

    # Run the genetic algorithm optimization
    ga_instance.run()

    # Get the best solution
    best_solution = ga_instance.best_solution()
    best_port_param = None
    best_port = None

    # Print the best solution

    best_solution_indices, best_solution_fitness, generation_number = best_solution
    if configuration == "bass_reflex":
        best_speaker_idx, best_edge_idx, best_port_param_idx, best_port_idx = (
            best_solution_indices.astype(int)
        )
        best_speaker = loudspeakers[int(best_speaker_idx)][
            "name"
        ]  # Convert to int here
        best_edge = pentagon_edges[int(best_edge_idx)]
        best_port_param = port_params[int(best_port_param_idx)]
        best_port = num_of_ports[int(best_port_idx)]


        power = calculate_sound_power(
            loudspeakers[int(best_speaker_idx)],
            par.clb_par,
            best_edge,
            best_port_param,
            best_port,
            configuration
        )
    else:
        best_speaker_idx, best_edge_idx = best_solution_indices.astype(int)
        best_speaker = loudspeakers[int(best_speaker_idx)][
            "name"
        ]  # Convert to int here
        best_edge = pentagon_edges[int(best_edge_idx)]

        power = calculate_sound_power(
            loudspeakers[int(best_speaker_idx)], par.clb_par, best_edge, None, None, configuration
        )

    # Calculate and print the open angle
    rd = best_edge["rd"]
    r = loudspeakers[best_speaker_idx]["r"]
    rho = np.sqrt(rd**2 + r**2)
    open_angle = np.arccos(rd / rho) * 180 / np.pi
    # print("Open Angle:", open_angle)

    Lw = power[0]
    power_1_3_octave = np.zeros(len(central_frequencies) - 1)

    for j in range(len(central_frequencies) - 1):
        start_idx = np.argmax(frequencies >= central_frequencies[j])
        stop_idx = np.argmax(frequencies >= central_frequencies[j + 1])
        power_1_3_octave[j] = np.mean(Lw[start_idx:stop_idx])

    # Plotting the results
    x_ticks = [50, 100, 200, 400, 800, 1600, 3150]
    tick_indices = [central_frequencies.index(x) for x in x_ticks]

    plt.bar(
        np.arange(len(power_1_3_octave)), power_1_3_octave, width=0.8, align="center"
    )
    plt.xticks(tick_indices, x_ticks)
    plt.title("Best Sound Power")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("dB rel. 1pW")
    plt.grid(which="both", axis="y")
    plt.show()

    # print("Optimized solution:", best_solution_indices)
    # print("Best fitness:", float(best_solution_fitness))
    time_passed = time() - start_time

    plt.plot(ga_instance.best_solutions_fitness)
    plt.title("Best Fitness Value")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness Value")
    plt.grid(True)
    plt.show()

    return (
        float(best_solution_fitness),
        best_speaker,
        best_edge,
        best_port_param,
        best_port,
        configuration,
        open_angle,
        time_passed,
    )


def display_results(
    best_cost,
    best_speaker,
    best_edge,
    best_port_param,
    best_port,
    configuration,
    open_angle,
    time_passed,
):
    print("Best fitness:", best_cost)
    print("Loudspeaker:", best_speaker)
    print("Edge:", best_edge)
    if configuration == "bass_reflex":
        print("port:", best_port_param)
        print("Ports:", best_port)
    print("Open Angle:", open_angle)
    print(f"Took: {time_passed} seconds")
    return None


if __name__ == "__main__":
    # Choose configuration: "bass_reflex" or "closed_box"
    (
        best_cost,
        best_speaker,
        best_edge,
        best_port_param,
        best_port,
        configuration,
        open_angle,
        time_passed,
    ) = run_pygad(
        par.list_of_loudspeakers,
        par.pentagon_edges_icosi,
        port_params=par.port_parameters,
        num_of_ports=par.number_of_ports,
        configuration="bass_reflex",
        # port_params=None,
        # num_of_ports=None,
        # configuration="closed_box",
    )
    display_results(
        best_cost,
        best_speaker,
        best_edge,
        best_port_param,
        best_port,
        configuration,
        open_angle,
        time_passed,
    )
    
