import numpy as np
import matplotlib.pyplot as plt
from Loudspeakers_Enclosures import (
    DodecahedronBassReflexEnclosure,
    DodecahedronEnclosure,
)
from scipy.optimize import differential_evolution
import parameters as par
from time import time


def run_differential_evolution(
    loudspeakers, pentagon_edges, port_params, num_of_ports, configuration
):

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
    frequencies = third_octave_bands(50, 1000, 3)
    frequencies = np.array(frequencies)
    frequencies = frequencies[(frequencies >= 50) & (frequencies <= 1000)]

    def calculate_sound_power(
        loudspeaker, clb_parameters, pentagon_edge, port_params, num_of_ports
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
    def objective_function(solution):
        fitness_values = []

        # Convert float indices to integers
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
                loudspeaker, par.clb_par, edge, port_param, ports
            )
        elif configuration == "closed_box":
            loudspeaker = loudspeakers[loudspeaker_idx]
            edge = pentagon_edges[edge_idx]

            lw, power = calculate_sound_power(
                loudspeaker, par.clb_par, edge, None, None
            )

        for i in range(len(frequencies)):
            rd = edge["rd"]
            r = loudspeaker["r"]
            rho = np.sqrt(rd**2 + r**2)
            open_angle = np.arccos(rd / rho) * 180 / np.pi
            if edge["type"] == "dodecahedron" and (open_angle < 20 or open_angle > 28):
                power[i] = -np.inf
            elif edge["type"] == "icosidodecahedron" and (
                open_angle < 18 or open_angle > 26.6
            ):
                power[i] = -np.inf

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

        fitness_values.append(np.abs(log_power_ratio))

        return fitness_values

    if configuration == "bass_reflex":
        # Define the bounds for the search space
        bounds = [
            (0, len(loudspeakers) - 1),  # Loudspeaker index
            (0, len(pentagon_edges) - 1),  # Volume index
            (0, len(port_params) - 1),  # port index
            (0, len(num_of_ports) - 1),
        ]  # Ports index
    elif configuration == "closed_box":
        bounds = [
            (0, len(loudspeakers) - 1),  # Loudspeaker index
            (0, len(pentagon_edges) - 1),  # Volume index
        ]

    fitness_evolution = []

    def callback(xk, convergence):
        fitness_evolution.append(objective_function(xk))

    # Use differential evolution for optimization
    result = differential_evolution(
        objective_function,
        bounds,
        strategy="randtobest1bin",
        maxiter=30,  # Increase the number of iterations
        popsize=10,
        mutation=(0.5, 1),
        polish=True,
        tol=1e-5,
        callback=callback,
        disp=True,
    )
    best_port_param = None
    best_port = None

    # Extract the optimized solution
    optimal_solution = result.x.astype(int)
    best_fitness = result.fun

    best_solution_indices = optimal_solution
    if configuration == "bass_reflex":
        best_speaker_idx, best_edge_idx, best_port_param_idx, best_port_idx = (
            best_solution_indices
        )
        best_speaker = loudspeakers[best_speaker_idx]["name"]
        best_edge = pentagon_edges[best_edge_idx]
        best_port_param = port_params[best_port_param_idx]
        best_port = num_of_ports[best_port_idx]

        power = calculate_sound_power(
            loudspeakers[best_speaker_idx],
            par.clb_par,
            best_edge,
            best_port_param,
            best_port,
        )
    else:
        best_speaker_idx, best_edge_idx = best_solution_indices
        best_speaker = loudspeakers[best_speaker_idx]["name"]
        best_edge = pentagon_edges[best_edge_idx]

        power = calculate_sound_power(
            loudspeakers[best_speaker_idx], par.clb_par, best_edge, None, None
        )

    # Calculate the open angle
    rd = best_edge["rd"]
    r = loudspeakers[best_speaker_idx]["r"]
    rho = np.sqrt(rd**2 + r**2)
    open_angle = np.arccos(rd / rho) * 180 / np.pi

    # Calculate the power in 1/3 octave bands
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

    time_passed = time() - start_time

    plt.plot(fitness_evolution)
    plt.title("Best Fitness Value")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness Value")
    plt.grid(True)
    plt.show()

    return (
        best_fitness,
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

    (
        best_cost,
        best_speaker,
        best_edge,
        best_port_param,
        best_port,
        configuration,
        open_angle,
        time_passed,
    ) = run_differential_evolution(
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
