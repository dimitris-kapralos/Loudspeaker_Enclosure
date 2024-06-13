import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import platform
import warnings
from Loudspeakers_Enclosures import DodecahedronEnclosure, DodecahedronBassReflexEnclosure
from PSO import run_pso
from differential_evolution import run_differential_evolution
from PyGAD import run_pygad
import parameters as parameters

# Function to clear the console screen
def clear_screen():
    if platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear')

def main_menu():
    while True:
        clear_screen()
        print("Welcome to Omni-directional Sources Analysis Software!")
        print("What would you like to do?")
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
        clear_screen()  # Clear the console
        print("Select a source:")
        print("1. Dodecahedron Closed Box")
        print("2. Icosidodecahedron Closed Box")
        print("3. Icosidodecahedron Bass Reflex")
        print("4. All cases")
        print("0. Exit")

        source_choice = input("Enter your choice: ")

        if source_choice == "1":
            pentagon_edges = parameters.pentagon_edges_dodeca
            selected_diode_parameters = None
            selected_ports = None
            selected_loudspeakers = select_loudspeakers()
            selected_edges = select_pentagon_edges(pentagon_edges)
            clear_screen()  # Clear the console
            select_algorithm(source_choice, selected_edges, selected_loudspeakers, selected_diode_parameters, selected_ports, 'closed_box')            
        elif source_choice == "2":
            pentagon_edges = parameters.pentagon_edges_icosi
            selected_diode_parameters = None
            selected_ports = None
            selected_loudspeakers = select_loudspeakers()
            selected_edges = select_pentagon_edges(pentagon_edges)   
            clear_screen()  # Clear the console
            select_algorithm(source_choice, selected_edges, selected_loudspeakers, selected_diode_parameters, selected_ports, 'closed_box')         
        elif source_choice == "3":
            pentagon_edges = parameters.pentagon_edges_icosi
            selected_diode_parameters, selected_ports = select_port_parameters() 
            selected_loudspeakers = select_loudspeakers()
            selected_edges = select_pentagon_edges(pentagon_edges)  
            clear_screen()  # Clear the console                     
            select_algorithm(source_choice, selected_edges, selected_loudspeakers, selected_diode_parameters, selected_ports, 'bass_reflex')
       
        elif source_choice == "4": 
            pentagon_edges = parameters.pentagon_edges_dodeca
            pentagon_edges1 = parameters.pentagon_edges_icosi 
            selected_diode_parameters = None
            selected_ports = None
            selected_diode_parameters1, selected_ports1 = select_port_parameters()
            selected_loudspeakers = select_loudspeakers()
            print("Select pentagon edges for Dodecahedron Enclosure")
            selected_edges = select_pentagon_edges(pentagon_edges) 
            print("Select pentagon edges for Icosidodecahedron Enclosure")
            selected_edges1 = select_pentagon_edges(pentagon_edges1) 
            clear_screen()  # Clear the console
            print("For Dodecahedron Closed Box")          

            results = []       
                 
            results.append(select_algorithm(source_choice, selected_edges, selected_loudspeakers, selected_diode_parameters, selected_ports, 'closed_box'))
        
            # clear_screen()  # Clear the console
            print("For Icosidodecahedron Closed Box")
            results.append(select_algorithm(source_choice, selected_edges1, selected_loudspeakers, selected_diode_parameters, selected_ports, 'closed_box'))
            
            # clear_screen()  # Clear the console
            print("For Icosidodecahedron Bass Reflex")
            results.append(select_algorithm(source_choice, selected_edges1, selected_loudspeakers, selected_diode_parameters1, selected_ports1, 'bass_reflex'))
            
            best_result, best_cost = compare_results(results)
            
            print("Best Result:")
            print(f"Cost: {best_cost}")
            print(f"Loudspeaker: {best_result[1]}")
            print(f"Edge: {best_result[2]['edge']}, type: {best_result[2]['type']}")
            if best_result[3] is not None:
                print(f"port length: {best_result[3]['t']}, port radius: {best_result[3]['radius']}")
            if best_result[4] is not None:        
                print(f"Number of Ports: {best_result[4]}")
            print(f"Configuration: {best_result[5]}")
            input("Press any key to return to the main menu...") 
        
        elif source_choice == "0":
            print("Exiting...")
            break

        else:
            print("Invalid choice") 
            continue


# Placeholder function for Individual Analysis menu
def individual_analysis_menu():
    while True:
        plot_style = None  # Initialize plot_style
        clear_screen()
        print("Individual Analysis Options:")
        print("Select Solid Type:")
        print("1. Dodecahedron")
        print("2. Icosidodecahedron")
        print("0. Back")

        choice = input("Enter your choice: ").strip()

        if choice == '1':
            solid_type = 'dodecahedron' 
            
            enclosure_type = select_enclosure_type()
            if enclosure_type is None:
                continue  # User chose to go back
            
            if enclosure_type == 'bass reflex':
                num_ports = select_ports()
                if num_ports is None:
                    continue  # User chose to go back

                port_length = select_port_length()
                if port_length is None:
                    continue  # User chose to go back

                port_radius = select_port_radius()
                if port_radius is None:
                    continue  # User chose to go back

            edge_length = select_edge_length()
            if edge_length is None:
                continue  # User chose to go back

            loudspeaker = select_loudspeaker()
            if loudspeaker is None:
                continue  # User chose to go back

            while True:
                plot_type = select_plot_type(enclosure_type)  # Pass enclosure type here
                if plot_type is None:
                    break  # User chose to go back

                if plot_type == 'power':
                    plot_style = select_plot_style()
                    if plot_style is None:
                        continue

                # Set up the parameters based on user choices
                
                chosen_solid = {'edge': edge_length, 'type': 'dodecahedron'}

                frequencies, central_frequencies = setup_frequencies(plot_style, plot_type)

                chosen_loudspeaker_params = next((speaker for speaker in parameters.list_of_loudspeakers if speaker['name'] == loudspeaker), None)

                if solid_type == 'dodecahedron' and enclosure_type == 'closed box':
                    run_simulation_cb(DodecahedronEnclosure, parameters.clb_par, chosen_solid, chosen_loudspeaker_params, frequencies, central_frequencies, plot_style, plot_type)
                elif solid_type == 'dodecahedron' and enclosure_type == 'bass reflex':
                    run_simulation_br(DodecahedronBassReflexEnclosure, parameters.clb_par, chosen_solid, parameters.empty_list_of_port_parameters,  chosen_loudspeaker_params, frequencies, central_frequencies, plot_style, plot_type, num_ports, port_length, port_radius)

                plot_again = input("Press 1 if you would you like to plot again or press any key: ").strip().lower()
                if plot_again != '1':
                    print("Invalid choice. Please enter '1' to plot again or '0' to go back.")
                    break  # Continue the loop to prompt for input again
                elif plot_again == '1':
                    continue
        elif choice == '2':
            solid_type = 'icosidodecahedron'
            enclosure_type = select_enclosure_type()
            if enclosure_type is None:
                continue  # User chose to go back
            
            if enclosure_type == 'bass reflex':
                num_ports = select_ports()
                if num_ports is None:
                    continue  # User chose to go back

                port_length = select_port_length()
                if port_length is None:
                    continue  # User chose to go back

                port_radius = select_port_radius()
                if port_radius is None:
                    continue  # User chose to go back

            edge_length = select_edge_length()
            if edge_length is None:
                continue  # User chose to go back

            loudspeaker = select_loudspeaker()
            if loudspeaker is None:
                continue  # User chose to go back

            while True:
                plot_type = select_plot_type(enclosure_type)  # Pass enclosure type here
                if plot_type is None:
                    break  # User chose to go back

                if plot_type == 'power':
                    plot_style = select_plot_style()
                    if plot_style is None:
                        continue

                # Set up the parameters based on user choices                
                chosen_solid = {'edge': edge_length, 'type': 'icosidodecahedron'}
    
                frequencies, central_frequencies = setup_frequencies(plot_style, plot_type)

                chosen_loudspeaker_params = next((speaker for speaker in parameters.list_of_loudspeakers if speaker['name'] == loudspeaker), None)

                if solid_type == 'icosidodecahedron' and enclosure_type == 'closed box':
                    run_simulation_cb(DodecahedronEnclosure, parameters.clb_par, chosen_solid, chosen_loudspeaker_params, frequencies, central_frequencies, plot_style, plot_type)
                elif solid_type == 'icosidodecahedron' and enclosure_type == 'bass reflex':
                    run_simulation_br(DodecahedronBassReflexEnclosure, parameters.clb_par, chosen_solid, parameters.empty_list_of_port_parameters,  chosen_loudspeaker_params, frequencies, central_frequencies, plot_style, plot_type, num_ports, port_length, port_radius)

                plot_again = input("Press 1 if you would you like to plot again or press anything to return back: ").strip().lower()
                if plot_again != '1':
                    print("Invalid choice. Please enter '1' to plot again or '0' to go back.")
                    break  # Continue the loop to prompt for input again
                elif plot_again == '1':
                    continue
        elif choice == '0':
            break  # Exit the loop and go back to main menu
        else:
            print("Invalid choice. Please enter 1 or 0.")
# Function to compare the algorithm results and display the best result
def compare_results(results):
    
    
    best_result = None
    best_cost = float('inf')
    for result in results:
        cost, speaker, edge, port_param , port, configuration = result
        if cost < best_cost:
            best_result = result
            best_cost = cost
    clear_screen()  
    return best_result, best_cost            
                

def select_pentagon_edges(pentagon_edges):

    print("Available pentagon edges:")
    for i, edge in enumerate(pentagon_edges, start=1):
        print(f"{i}. Edge: {edge['edge']}")

    while True:
        edge_choice = input("Enter number of pentagon edge choice (comma-separated numbers), or enter 'a' to select them all: ")
        if edge_choice.lower() == "a":
            selected_edges = pentagon_edges
            break
        try:
            edge_choices = [int(x.strip()) for x in edge_choice.split(",")]
            selected_edges = [pentagon_edges[i - 1] for i in edge_choices]
            break  # Break the loop if input is valid
        except (ValueError, IndexError):
            print("Invalid input. Please enter valid pentagon edges choice.")
    clear_screen()
            
    return selected_edges            

def select_loudspeakers():
    clear_screen()  # Clear the console

    print("Available loudspeakers:")
    for i, speaker in enumerate(parameters.list_of_loudspeakers, start=1):
        print(f"{i}. {speaker['name']}")

    while True:
        loudspeaker_choice = input("Enter number of loudspeakers choice (comma-separated numbers), or enter 'a' to select them all: ")
        if loudspeaker_choice.lower() == "a":
            selected_loudspeakers = parameters.list_of_loudspeakers
            break
        try:
            loudspeaker_choices = [int(x.strip()) for x in loudspeaker_choice.split(",")]
            selected_loudspeakers = [parameters.list_of_loudspeakers[i - 1] for i in loudspeaker_choices]
            break  # Break the loop if input is valid
        except (ValueError, IndexError):
            print("Invalid input. Please enter valid loudspeaker choice.")
    clear_screen()  # Clear the console
    return selected_loudspeakers            

def select_port_parameters():  
    
    print("Available port parameters:")
    for i, params in enumerate(parameters.port_parameters, start=1):
        print(f"{i}. t: {params['t']}, radius: {params['radius']}")
    while True:
        port_params_choice = input("Enter number of port parameters choice (comma-separated numbers), or enter 'a' to select all: ")
        if port_params_choice.lower() == "a":
            selected_diode_parameters = parameters.port_parameters
            break
        try:
            port_params_choices = [int(x.strip()) for x in port_params_choice.split(",")]
            selected_diode_parameters = [parameters.port_parameters[i - 1] for i in port_params_choices]
            break  # Break the loop if input is valid
        except (ValueError, IndexError):
            print("Invalid input. Please enter valid port parameters or a.")
    clear_screen()  # Clear the console
    print("Available number of ports:")
    for i, num in enumerate(parameters.number_of_ports, start=1):
        print(f"{i}. {num} ports")
    while True:
        ports_choice = input("Enter number of ports choice (comma-separated numbers), or enter 'a' to select them all: ")
        if ports_choice.lower() == "a":
            selected_ports = parameters.number_of_ports
            break
        try:
            ports_choices = [int(x.strip()) for x in ports_choice.split(",")]
            selected_ports = [parameters.number_of_ports[i - 1] for i in ports_choices]
            break  # Break the loop if input is valid
        except (ValueError, IndexError):
            print("Invalid input. Please enter valid number of ports or a.") 
    clear_screen()  # Clear the console            
    return selected_diode_parameters, selected_ports            
    

def select_algorithm(source, pentagon_edges, loudspeakers, diode_params, num_ports, configuration):
    print("Select Algorithm")
    print("1. Genetic Algorithm")
    print("2. Particle Swarm Optimization")
    print("3. Differential Evolution")
    
    choice = int(input("Enter your choice: "))
    
    if choice == 1:
        if source == "1" or source == "2" :
           cost, speaker, edge, port_param, port, configuration = run_pygad(loudspeakers, pentagon_edges, diode_params, num_ports,  'closed_box')
        elif source == "3":
           cost, speaker, edge, port_param, port, configuration = run_pygad(loudspeakers, pentagon_edges, diode_params, num_ports, 'bass_reflex') 
        elif source == "4":
            if configuration == 'closed_box':
                cost, speaker, edge, port_param, port, configuration = run_pygad(loudspeakers, pentagon_edges, diode_params, num_ports, configuration)
            elif configuration == 'bass_reflex':
                cost, speaker, edge, port_param, port, configuration = run_pygad(loudspeakers, pentagon_edges, diode_params, num_ports, configuration)           
    elif choice == 2:
        if source == "1" or source == "2":
            cost, speaker, edge, port_param, port, configuration = run_pso(loudspeakers, pentagon_edges, diode_params, num_ports,  'closed_box')
        elif source == "3":
            cost, speaker, edge, port_param, port, configuration = run_pso(loudspeakers, pentagon_edges, diode_params, num_ports, 'bass_reflex') 
        elif source == "4":
            if configuration == 'closed_box':
                cost, speaker, edge, port_param, port, configuration = run_pso(loudspeakers, pentagon_edges, diode_params, num_ports, configuration)
            elif configuration == 'bass_reflex':
                cost, speaker, edge, port_param, port, configuration = run_pso(loudspeakers, pentagon_edges, diode_params, num_ports, configuration)           
    elif choice == 3:
        if source == "1" or source == "2":
           cost, speaker, edge, port_param, port, configuration = run_differential_evolution(loudspeakers, pentagon_edges, diode_params, num_ports,  'closed_box')
        elif source == "3":
           cost, speaker, edge, port_param, port, configuration = run_differential_evolution(loudspeakers, pentagon_edges, diode_params, num_ports, 'bass_reflex') 
        elif source == "4":
            if configuration == 'closed_box':
               cost, speaker, edge, port_param, port, configuration = run_differential_evolution(loudspeakers, pentagon_edges, diode_params, num_ports, configuration)
            elif configuration == 'bass_reflex':
               cost, speaker, edge, port_param, port, configuration = run_differential_evolution(loudspeakers, pentagon_edges, diode_params, num_ports, configuration)            
    else:
        print("Invalid choice!")
        return None
    
    return cost, speaker, edge, port_param , port, configuration



def select_enclosure_type():
    while True:
        clear_screen()
        print("Select Enclosure Type:")
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

# Function to select the number of ports
def select_ports():
    while True:
        clear_screen()
        print("Select Number of Ports:")
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

# Function to select the length of the ports
def select_port_length():
    while True:
        clear_screen()
        print("Select Port Length:")
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

# Function to select the radius of the ports
def select_port_radius():
    while True:
        clear_screen()
        print("Select Port radius:")
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

def select_edge_length():
    while True:
        clear_screen()
        print("Select Pentagon Edge Length:")
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


def select_loudspeaker():
    while True:
        clear_screen()
        print("Select Loudspeaker:")
        loudspeakers = parameters.list_of_loudspeakers
        for i, loudspeaker in enumerate(loudspeakers, 1):
            print(f"{i}. {loudspeaker['name']}")

        print("0. Back")

        choice = input("Enter your choice: ").strip()

        if choice == '0':
            return None
        elif choice.isdigit() and 1 <= int(choice) <= len(loudspeakers):
            return loudspeakers[int(choice) - 1]['name']
        else:
            print("Invalid choice. Please enter a number between 0 and", len(loudspeakers))

            
def select_plot_type(enclosure_type):
    while True:
        clear_screen()
        print("Select Plot Type:")
        print("1. Impedance Response")
        print("2. Sound Power Lw")
        if enclosure_type == 'bass reflex':
            print("3. Port and Diaphragm Response")
        print("0. Back")

        choice = input("Enter your choice: ").strip()

        plot_type = {
            '1': 'impedance',
            '2': 'power',
        }

        if choice in plot_type:
            return plot_type[choice]
        elif choice == '3' and enclosure_type == 'bass reflex':
            return 'port and diaphragm'
        elif choice == '0':
            return None
        else:
            print("Invalid choice. Please enter a number between 0 and", '3' if enclosure_type == 'bass reflex' else '2')
          
            
def select_plot_style():
    while True:
        clear_screen()
        print("Select Plot Style:")
        print("1. 1/3 Octave Bands")
        print("2. Octave Bands")
        print("3. Logarithmic Scale")
        print("0. Back")

        choice = input("Enter your choice: ").strip()

        plot_style = {
            '1': '1/3 octave',
            '2': 'octave',
            '3': 'logarithmic',
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
    parser.add_argument('--frequency-type', choices=['1/3 octave', 'octave', 'logarithmic'], default='1/3 octave',
                    help='Type of frequency bands to use in the simulation.')
    
    return parser

# Function to set up frequency bands
def setup_frequencies(plot_style, plot_type):
    if plot_type == 'power':
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
        elif plot_style == 'logarithmic':
            octave_steps = 24 
            min_frequency = 10
            max_frequency = 15000
            num_points = int(octave_steps * np.log2(max_frequency / min_frequency)) + 1
            frequencies = np.logspace(
                np.log2(min_frequency), np.log2(max_frequency), num=num_points, base=2
            )
            central_frequencies = None  # Placeholder for logarithmic frequencies
    elif plot_type == 'impedance':
            octave_steps = 24 
            min_frequency = 10
            max_frequency = 15000
            num_points = int(octave_steps * np.log2(max_frequency / min_frequency)) + 1
            frequencies = np.logspace(
                np.log2(min_frequency), np.log2(max_frequency), num=num_points, base=2
            )
            central_frequencies = None  # Placeholder for logarithmic frequencies 
    elif plot_type == 'port and diaphragm':
            octave_steps = 24 
            min_frequency = 10
            max_frequency = 15000
            num_points = int(octave_steps * np.log2(max_frequency / min_frequency)) + 1
            frequencies = np.logspace(
                np.log2(min_frequency), np.log2(max_frequency), num=num_points, base=2
            )
            central_frequencies = None                     

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
def run_simulation_cb(enclosure_class, enclosure_params, chosen_solid, loudspeaker_params, frequencies, central_frequencies, plot_style, plot_type):
    enclosure = enclosure_class(enclosure_params, chosen_solid, loudspeaker_params)
    response, impedance, power, spl, w = enclosure.calculate_dodecahedron_response(frequencies, 3, 0.3)
    if plot_type == 'impedance':
        plot_impedance(frequencies, impedance)
    elif plot_type == 'power':    
        plot_power(frequencies, power, central_frequencies, plot_style)

# Function to run the simulation for Bass Reflex (BR) enclosures
def run_simulation_br(enclosure_class, enclosure_params, chosen_solid, diode_params, loudspeaker_params, frequencies, central_frequencies, plot_style, plot_type, num_ports, port_length, port_radius):
    diode_params['t'] = port_length
    diode_params['radius'] = port_radius
    enclosure = enclosure_class(enclosure_params, chosen_solid, diode_params, loudspeaker_params)
    response, response_diaphragm, response_port, impedance, power, spl, w = enclosure.calculate_dodecahedron_bass_reflex_response(frequencies, num_ports, 3, 0.3)
    
    if plot_type == 'impedance':
        plot_impedance(frequencies, impedance)
    elif plot_type == 'power':
        plot_power(frequencies, power, central_frequencies, plot_style)
    elif plot_type == 'port and diaphragm':
        plot_port_diaphragm_response(frequencies, response_diaphragm, response_port)



def plot_impedance(frequencies, impedance):
    fig, ax2 = plt.subplots()
    ax2.plot(frequencies, impedance)
    ax2.set_xscale('log')
    ax2.set_yscale('logarithmic')
    ax2.set_title("Impedance Response")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Impedance (Ohms)")
    ax2.grid(True, which="both", linestyle='--')
    plt.show()


# Unified function to plot power in different scales
def plot_power(frequencies, power, central_freqs, plot_style):
    if plot_style == '1/3 octave':
        plot_power_1_3_octave(frequencies, power, central_freqs)
    elif plot_style == 'octave':
        plot_power_octave(frequencies, power, central_freqs)
    elif plot_style == 'logarithmic':
        plot_power_logarithmic(frequencies, power)
    else:
        print("Invalid frequency type.")


# Function to plot power in 1/3 octave bands
def plot_power_1_3_octave(frequencies, power, central_freqs):
    power_1_3_octave = np.zeros(len(central_freqs) - 1)

    for j in range(len(central_freqs) - 1):
        start_idx = np.argmax(frequencies >= central_freqs[j])
        stop_idx = np.argmax(frequencies >= central_freqs[j + 1])

        power_1_3_octave[j] = np.mean(power[start_idx:stop_idx])

    fig, ax = plt.subplots()
    bar_width = 0.6
    x_ticks = [50, 100, 200, 400, 800, 1600, 3150]
    tick_indices = [central_freqs.index(x) for x in x_ticks]

    ax.bar(np.arange(len(power_1_3_octave)), power_1_3_octave, width=bar_width, align="center")
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(x_ticks)
    ax.set_title("Sound Power Lw in 1/3 Octave Bands")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("dB rel. 1pW")
    ax.grid(which="both", axis="y")
    ax.set_ylim(60, 120)
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

    
# Function to plot power in logarithmic scale
def plot_power_logarithmic(frequencies, power):
    fig, ax = plt.subplots()
    ax.plot(frequencies, power, 'b')
    ax.set_xscale('log')
    ax.set_yscale('logarithmic')
    ax.set_title("Sound Power in Logarithmic Scale")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.grid(True, which="both", linestyle='--')
    plt.show()
  

def plot_port_diaphragm_response(frequencies, response_diaphragm, response_port):
    fig, ax = plt.subplots()
    ax.plot(frequencies, response_diaphragm, label="Diaphragm Response", color='b')
    ax.plot(frequencies, response_port, label="Port Response", color='g')
    ax.set_xscale('log')
    ax.set_yscale('logarithmic')
    ax.set_title("Port and Diaphragm Response")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Response (dB)")
    ax.legend()
    ax.grid(True, which="both", linestyle='--')
    plt.show()
    
    
def suppress_specific_warnings():
    warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part", category=np.ComplexWarning) 
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide", category=RuntimeWarning)


if __name__ == "__main__":
    suppress_specific_warnings()
    main_menu()