import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import platform
import warnings
from Loudspeakers_Enclosures import DodecahedronEnclosure, DodecahedronBassReflexEnclosure

# Function to load parameters from a JSON file
def load_params(file_path):
    try:
        with open(file_path, 'r') as file:
            params = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {file_path}.")
        exit(1)
    return params

# Function to clear the console screen
def clear_screen():
    if platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear')

# Main menu function
def main_menu():
    while True:
        clear_screen()
        print("Main Menu:")
        print("1. Optimization")
        print("2. Individual Analysis")
        print("0. Exit")
        
        choice = input("Enter your choice: ").strip()
        
        if choice == '1':
            optimization_menu()
        elif choice == '2':
            individual_analysis_menu()
        elif choice == '0':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 0.")

# Placeholder function for Optimization menu
def optimization_menu():
    while True:
        clear_screen()
        print("Optimization Menu:")
        print("1. Start Optimization (functionality to be implemented)")
        print("0. Back to Main Menu")
        
        choice = input("Enter your choice: ").strip()
        
        if choice == '1':
            print("Optimization functionality will be implemented later.")
            input("Press Enter to return to the Optimization Menu...")
        elif choice == '0':
            break
        else:
            print("Invalid choice. Please enter 1 or 0.")


# Function for Individual Analysis option
def individual_analysis_menu():
    while True:
        clear_screen()
        print("Individual Analysis Options:")
        print("1. Choose Solid Type")
        print("0. Back to main Menu")

        choice = input("Enter your choice: ").strip()

        if choice == '1':
            solid_type = choose_solid_type()
            if solid_type is None:
                continue  # User chose to go back

            enclosure_type = choose_enclosure_type()
            if enclosure_type is None:
                continue  # User chose to go back
            
            if enclosure_type == 'bass reflex':
                num_ports = choose_ports()
                if num_ports is None:
                    continue  # User chose to go back

                port_length = choose_port_length()
                if port_length is None:
                    continue  # User chose to go back

                port_radius = choose_port_radius()
                if port_radius is None:
                    continue  # User chose to go back

            edge_length = choose_edge_length()
            if edge_length is None:
                continue  # User chose to go back

            loudspeaker = choose_loudspeaker()
            if loudspeaker is None:
                continue  # User chose to go back
            
            plot_style = choose_plot_style()
            if plot_style is None:
                continue

            parser = argparse.ArgumentParser()
            args = parser.parse_args([])

            # Manually set the arguments based on user choices
            args.solid = solid_type
            args.enclosure = enclosure_type
            args.edge = edge_length
            args.loudspeaker = loudspeaker  # Add loudspeaker attribute
            args.params = 'params.json'
            args.plot_style = plot_style

            params = load_params(args.params)

            chosen_loudspeaker_params = params['loudspeakers'][args.loudspeaker]


            if args.solid == 'dodecahedron':
                chosen_dodecahedron = {'edge': args.edge, 'type': 'dodecahedron'}
            else:
                chosen_icosidodecahedron = {'edge': args.edge, 'type': 'icosidodecahedron'}

            frequencies, central_frequencies = setup_frequencies(args.plot_style)
            

            if args.solid == 'dodecahedron' and args.enclosure == 'closed box':
                run_simulation_cb(DodecahedronEnclosure, params['box'], chosen_dodecahedron, chosen_loudspeaker_params, frequencies, central_frequencies, plot_style)
            elif args.solid == 'icosidodecahedron' and args.enclosure == 'closed box':
                run_simulation_cb(DodecahedronEnclosure, params['box'], chosen_icosidodecahedron, chosen_loudspeaker_params, frequencies, central_frequencies, plot_style)
            elif args.solid == 'dodecahedron' and args.enclosure == 'bass reflex':
                run_simulation_br(DodecahedronBassReflexEnclosure, params['box'], chosen_dodecahedron, params['diode_parameters'], chosen_loudspeaker_params, frequencies, central_frequencies, plot_style, num_ports, port_length, port_radius)
            elif args.solid == 'icosidodecahedron' and args.enclosure == 'bass reflex':
                run_simulation_br(DodecahedronBassReflexEnclosure, params['box'], chosen_icosidodecahedron, params['diode_parameters'], chosen_loudspeaker_params, frequencies, central_frequencies, plot_style, num_ports, port_length, port_radius)

            input("Press Enter to return to the Individual Analysis Menu...")
        elif choice == '0':
            break
        else:
            print("Invalid choice. Please enter 1 or 0.")

def choose_solid_type():
    while True:
        clear_screen()
        print("Choose Solid Type:")
        print("1. Dodecahedron")
        print("2. Icosidodecahedron")
        print("0. Back")

        choice = input("Enter your choice: ").strip()

        if choice == '1':
            return 'dodecahedron'
        elif choice == '2':
            return 'icosidodecahedron'
        elif choice == '0':
            return None
        else:
            print("Invalid choice. Please enter 1, 2, or 0.")

def choose_enclosure_type():
    while True:
        clear_screen()
        print("Choose Enclosure Type:")
        print("1. Closed Box")
        print("2. Bass Reflex")
        print("0. Back")

        choice = input("Enter your choice: ").strip()

        if choice == '1':
            return 'closed box'
        elif choice == '2':
            return 'bass reflex'
        elif choice == '0':
            return None
        else:
            print("Invalid choice. Please enter 1, 2, or 0.")

# Function to choose the number of ports
def choose_ports():
    while True:
        clear_screen()
        print("Choose Number of Ports:")
        print("1. 4 Ports")
        print("2. 8 Ports")
        print("3. 12 Ports")
        print("4. 16 Ports")
        print("5. 20 Ports")
        print("0. Back")

        choice = input("Enter your choice: ").strip()

        num_ports = {
            '1': 4,
            '2': 8,
            '3': 12,
            '4': 16,
            '5': 20,
        }
        
        if choice in num_ports:
            return num_ports[choice]
        elif choice == '0':
            return None
        else:
            print("Invalid choice. Please enter a number between 0 and 5.")

# Function to choose the length of the ports
def choose_port_length():
    while True:
        clear_screen()
        print("Choose Port Length:")
        print("1. 0.018 m")
        print("2. 0.023 m")
        print("3. 0.028 m")
        print("4. 0.033 m")
        print("5. 0.038 m")
        print("0. Back")

        choice = input("Enter your choice: ").strip()

        port_lengths = {
            '1': 0.018,
            '2': 0.023,
            '3': 0.028,
            '4': 0.033,
            '5': 0.038,
        }

        if choice in port_lengths:
            return port_lengths[choice]
        elif choice == '0':
            return None
        else:
            print("Invalid choice. Please enter a number between 0 and 5.")

# Function to choose the radius of the ports
def choose_port_radius():
    while True:
        clear_screen()
        print("Choose Port radius:")
        print("1. 0.0079 m")
        print("2. 0.0105 m")
        print("3. 0.0133 m")
        print("4. 0.0175 m")
        print("5. 0.0205 m")
        print("0. Back")

        choice = input("Enter your choice: ").strip()

        port_radius = {
            '1': 0.0079,
            '2': 0.0105,
            '3': 0.0133,
            '4': 0.0175,
            '5': 0.0205,
        }

        if choice in port_radius:
            return port_radius[choice]
        elif choice == '0':
            return None
        else:
            print("Invalid choice. Please enter a number between 0 and 5.")

def choose_edge_length():
    while True:
        clear_screen()
        print("Choose Pentagon Edge Length:")
        print("1. 0.10 m")
        print("2. 0.11 m")
        print("3. 0.12 m")
        print("4. 0.13 m")
        print("5. 0.14 m")
        print("6. 0.15 m")
        print("7. 0.16 m")
        print("8. 0.17 m")
        print("9. 0.18 m")
        print("10. 0.19 m")
        print("0. Back")

        choice = input("Enter your choice: ").strip()

        edge_lengths = {
            '1': 0.10,
            '2': 0.11,
            '3': 0.12,
            '4': 0.13,
            '5': 0.14,
            '6': 0.15,
            '7': 0.16,
            '8': 0.17,
            '9': 0.18,
            '10': 0.19,
        }

        if choice in edge_lengths:
            return edge_lengths[choice]
        elif choice == '0':
            return None
        else:
            print("Invalid choice. Please enter a number between 0 and 9.")


def choose_loudspeaker():
    while True:
        clear_screen()
        print("Choose Loudspeaker:")
        params = load_params('params.json')
        loudspeakers = params['loudspeakers']
        for i, (key, value) in enumerate(loudspeakers.items(), 1):
            print(f"{i}. {key}")

        print("0. Back")

        choice = input("Enter your choice: ").strip()

        if choice == '0':
            return None
        elif choice.isdigit() and 1 <= int(choice) <= len(loudspeakers):
            return list(loudspeakers.keys())[int(choice) - 1]
        else:
            print("Invalid choice. Please enter a number between 0 and", len(loudspeakers))
            
def choose_plot_style():
    while True:
        clear_screen()
        print("Choose Plot Style:")
        print("1. 1/3 Octave Bands")
        print("2. Octave Bands")
        print("3. Linear Scale")
        print("0. Back")

        choice = input("Enter your choice: ").strip()

        plot_style = {
            '1': '1/3 octave',
            '2': 'octave',
            '3': 'linear',
        }

        if choice in plot_style:
            return plot_style[choice]
        elif choice == '0':
            return None
        else:
            print("Invalid choice. Please enter a number between 0 and 3.")            

# Function to create the argument parser
def create_parser():
    parser = argparse.ArgumentParser(description='Loudspeaker and Enclosure Simulation CLI', add_help=False)
    
    parser.add_argument('--solid', choices=['dodecahedron', 'icosidodecahedron'], required=True,
                        help='The type of geometric solid for the enclosure.')
    parser.add_argument('--edge', type=float, default=0.1, help='Edge length of the pentagon in meters.')
    parser.add_argument('--loudspeaker', type=str, required=True, help='Loudspeaker model to use.')
    parser.add_argument('--enclosure', choices=['closed box', 'bass reflex'], required=True, help='Type of enclosure.')
    parser.add_argument('--params', type=str, default='params.json', help='Path to the JSON file with parameters.')
    parser.add_argument('--frequency-type', choices=['1/3 octave', 'octave', 'linear'], default='1/3 octave',
                    help='Type of frequency bands to use in the simulation.')
    
    return parser

# Function to set up frequency bands
def setup_frequencies(plot_style):
    if plot_style == '1/3 octave':
        central_frequencies = [
            50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
            1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000
        ]
        frequencies = third_octave_bands(50, 7000, 3)
        frequencies = np.array(frequencies)
        frequencies = frequencies[(frequencies >= 50) & (frequencies <= 7000)]
    elif plot_style == 'octave':
        central_frequencies = [
            63, 125, 250, 500, 1000, 2000, 4000, 8000
        ]
        frequencies = octave_bands(50, 8000, 1)
        frequencies = np.array(frequencies)
        frequencies = frequencies[(frequencies >= 50) & (frequencies <= 8000)]        
    elif plot_style == 'linear':
        octave_steps = 24 
        min_frequency = 10
        max_frequency = 15000
        num_points = int(octave_steps * np.log2(max_frequency / min_frequency)) + 1
        frequencies = np.logspace(
            np.log2(min_frequency), np.log2(max_frequency), num=num_points, base=2
        )
        central_frequencies = None  # Placeholder for linear frequencies

    return frequencies, central_frequencies

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

# Function to calculate octave bands
def octave_bands(start_freq, stop_freq, num_bands_per_octave):
    bands = []
    f1 = start_freq
    while f1 < stop_freq:
        bands.append(f1)
        for i in range(1, num_bands_per_octave):
            bands.append(f1 * (2 ** (i / (1 * num_bands_per_octave))))
        f1 *= 2 ** (1 / 1)
    bands.append(stop_freq)
    return bands

# Function to run the simulation for Closed Box (CB) enclosures
def run_simulation_cb(enclosure_class, enclosure_params, chosen_solid, loudspeaker_params, frequencies, central_frequencies, plot_style):
    enclosure = enclosure_class(enclosure_params, chosen_solid, loudspeaker_params)
    response, impedance, power, spl, w = enclosure.calculate_dodecahedron_response(frequencies, 3, 0.3)
    plot_power(frequencies, power, central_frequencies, plot_style)

# Function to run the simulation for Bass Reflex (BR) enclosures
def run_simulation_br(enclosure_class, enclosure_params, chosen_solid, diode_params, loudspeaker_params, frequencies, central_frequencies, plot_style, num_ports, port_length, port_radius):
    diode_params['t'] = port_length
    diode_params['radius'] = port_radius
    enclosure = enclosure_class(enclosure_params, chosen_solid, diode_params, loudspeaker_params)
    response, impedance, power, slp, w = enclosure.calculate_dodecahedron_bass_reflex_response(frequencies, num_ports, 3, 0.3)
    plot_power(frequencies, power, central_frequencies, plot_style)


# Unified function to plot power in different scales
def plot_power(frequencies, power, central_freqs, plot_style):
    if plot_style == '1/3 octave':
        plot_power_1_3_octave(frequencies, power, central_freqs)
    elif plot_style == 'octave':
        plot_power_octave(frequencies, power, central_freqs)
    elif plot_style == 'linear':
        plot_power_linear(frequencies, power)
    else:
        print("Invalid frequency type.")


# Function to plot power in 1/3 octave bands
def plot_power_1_3_octave(frequencies, power, central_freqs):
    power_1_3_octave = np.zeros(len(central_freqs) - 1)

    for j in range(len(central_freqs) - 1):
        start_idx = np.argmax(frequencies >= central_freqs[j])
        stop_idx = np.argmax(frequencies >= central_freqs[j + 1])

        power_1_3_octave[j] = np.mean(power[start_idx:stop_idx])

    fig4, ax4 = plt.subplots()
    bar_width = 0.6
    x_ticks = [50, 100, 200, 400, 800, 1600, 3150]
    tick_indices = [central_freqs.index(x) for x in x_ticks]

    ax4.bar(np.arange(len(power_1_3_octave)), power_1_3_octave, width=bar_width, align="center")
    ax4.set_xticks(tick_indices)
    ax4.set_xticklabels(x_ticks)
    ax4.set_title("Sound Power Lw in 1/3 Octave Bands")
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("dB rel. 1pW")
    ax4.grid(which="both", axis="y")
    ax4.set_ylim(60, 120)
    plt.show()
    
def plot_power_octave(frequencies, power, central_freqs):
    power_octave = np.zeros(len(central_freqs) - 1)

    for j in range(len(central_freqs) - 1):
        start_idx = np.argmax(frequencies >= central_freqs[j])
        stop_idx = np.argmax(frequencies >= central_freqs[j + 1])

        power_octave[j] = np.mean(power[start_idx:stop_idx])

    fig, ax = plt.subplots()
    bar_width = 0.8
    x_ticks = [63, 125, 250, 500, 1000, 2000, 4000]
    tick_indices = [central_freqs.index(x) for x in x_ticks]

    ax.bar(np.arange(len(power_octave)), power_octave, width=bar_width, align="center")
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(x_ticks)
    ax.set_title("Sound Power Lw in Octave Bands")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("dB rel. 1pW")
    ax.grid(which="both", axis="y")
    ax.set_ylim(60, 120)
    plt.show()
    
# Function to plot power in linear scale
def plot_power_linear(frequencies, power):
    fig, ax = plt.subplots()
    ax.plot(frequencies, power, 'b')
    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.set_title("Sound Power in Linear Scale")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.grid(True, which="both", linestyle='--')
    plt.show()
    
def suppress_specific_warnings():
    warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part", category=np.ComplexWarning) 
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide", category=RuntimeWarning)


if __name__ == "__main__":
    suppress_specific_warnings()
    main_menu()
