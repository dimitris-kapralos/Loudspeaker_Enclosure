import curses
import numpy as np
import matplotlib.pyplot as plt
import sys
from Loudspeakers_Enclosures import DodecahedronEnclosure, DodecahedronBassReflexEnclosure
from PSO import run_pso
from differential_evolution import run_differential_evolution
from PyGAD import run_pygad
import parameters as parameters
import warnings

def clear_screen():
    pass  # This function will not be needed with curses

def main_menu(stdscr):
    curses.curs_set(0)  # Hide cursor
    current_row = 0
    main_menu_items = ["Optimization Menu", "Individual Analysis Menu", "Exit"]

    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "Welcome to the Omni-directional Sound Sources Software!")
        stdscr.addstr(1, 0, "Select an option:")

        for idx, item in enumerate(main_menu_items):
            x = 2
            y = 2 + idx
            if idx == current_row:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(y, x, item)
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(y, x, item)

        key = stdscr.getch()

        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(main_menu_items) - 1:
            current_row += 1
        elif key == curses.KEY_ENTER or key in [10, 13]:
            if current_row == 0:
                optimization_menu(stdscr)
            elif current_row == 1:
                individual_analysis_menu(stdscr)
            elif current_row == 2:
                sys.exit()



def optimization_menu(stdscr):
    curses.curs_set(0)  # Hide cursor
    current_row = 0
    opt_menu_items = ["Dodecahedron Closed Box", "Icosidodecahedron Closed Box", "Icosidodecahedron Bass Reflex", "All cases", "Return to Main Menu"]
    
    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "Optimization Menu")
        stdscr.addstr(1, 0, "Select a source:")
        
        for idx, row in enumerate(opt_menu_items):
            x = 2
            y = 2 + idx
            if idx == current_row:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(y, x, row)
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(y, x, row)

        key = stdscr.getch()

        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(opt_menu_items) - 1:
            current_row += 1     
        elif key == curses.KEY_ENTER or key in [10, 13]:
            if current_row == 0:
                # Handle Dodecahedron Closed Box
                selected_edges = select_pentagon_edges(stdscr, parameters.pentagon_edges_dodeca)
                selected_loudspeakers = select_loudspeakers(stdscr)
                selected_diode_parameters = None
                selected_ports = None
                cost, speaker, edge, port_param, port, configuration , open_angle, time_passed = select_algorithm(stdscr, "1", selected_edges, selected_loudspeakers, selected_diode_parameters, selected_ports, 'closed_box')
                display_results(stdscr, cost, speaker, edge, port_param, port, configuration, open_angle, time_passed)
            elif current_row == 1:
                # Handle Icosidodecahedron Closed Box
                selected_edges = select_pentagon_edges(stdscr, parameters.pentagon_edges_icosi)
                selected_loudspeakers = select_loudspeakers(stdscr)
                selected_diode_parameters = None
                selected_ports = None
                cost, speaker, edge, port_param, port, configuration, open_angle, time_passed = select_algorithm(stdscr, "2", selected_edges, selected_loudspeakers, selected_diode_parameters, selected_ports, 'closed_box')
                display_results(stdscr, cost, speaker, edge, port_param, port, configuration, open_angle, time_passed)
            elif current_row == 2:
                # Handle Icosidodecahedron Bass Reflex
                selected_edges = select_pentagon_edges(stdscr, parameters.pentagon_edges_icosi)
                selected_loudspeakers = select_loudspeakers(stdscr)
                selected_diode_parameters, selected_ports = select_port_parameters(stdscr)

                cost, speaker, edge, port_param, port, configuration, open_angle, time_passed = select_algorithm(stdscr, "3", selected_edges, selected_loudspeakers, selected_diode_parameters, selected_ports, 'bass_reflex')
                display_results(stdscr, cost, speaker, edge, port_param, port, configuration, open_angle, time_passed)
            elif current_row == 3:
                # Handle All cases
                selected_edges_dodeca = select_pentagon_edges(stdscr, parameters.pentagon_edges_dodeca)
                selected_edges_icosi = select_pentagon_edges(stdscr, parameters.pentagon_edges_icosi)
                selected_loudspeakers = select_loudspeakers(stdscr)
                selected_diode_parameters_cl = None
                selected_ports_cl = None
                selected_diode_parameters, selected_ports = select_port_parameters(stdscr)
                results = []
                results.append(select_algorithm(stdscr, "1", selected_edges_dodeca, selected_loudspeakers, selected_diode_parameters_cl, selected_ports_cl, 'closed_box'))
                results.append(select_algorithm(stdscr, "2", selected_edges_icosi, selected_loudspeakers, selected_diode_parameters_cl, selected_ports_cl, 'closed_box'))
                results.append(select_algorithm(stdscr, "3", selected_edges_icosi, selected_loudspeakers, selected_diode_parameters, selected_ports, 'bass_reflex'))
                cost, speaker, edge, port_param, port, configuration, open_angle, time_passed = compare_results(results)
                display_results(stdscr, cost, speaker, edge, port_param, port, configuration, open_angle, time_passed)
                
            elif current_row == 4:
                # Return to Main Menu
                main_menu(stdscr)
                break
            
            current_row = 0  # Reset current row for next menu display

    curses.curs_set(1)  # Show cursor after exiting
    curses.endwin()





# Placeholder function for Individual Analysis menu
def individual_analysis_menu(stdscr):
    # Initialize curses
    plot_style = None
    curses.curs_set(0)  # Hide cursor
    stdscr.clear()
    stdscr.refresh()

    # Define menu items
    menu_items = ["Dodecahedron", "Icosidodecahedron", "Return to Main Menu"]
    current_row = 0

    while True:
        stdscr.clear()
        stdscr.addstr("Individual Analysis Options:\n")
        
        # Display menu items
        for idx, item in enumerate(menu_items):
            if idx == current_row:
                stdscr.addstr(f" {item}\n", curses.A_REVERSE)
            else:
                stdscr.addstr(f" {item}\n")

        key = stdscr.getch()

        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(menu_items) - 1:
            current_row += 1
        elif key == curses.KEY_ENTER or key in [10, 13]:
            if current_row == len(menu_items) - 1:
                break  # Exit the loop and go home to main menu
            else:
                stdscr.clear()
                stdscr.refresh()
                solid_type = 'dodecahedron' if current_row == 0 else 'icosidodecahedron'
                
                enclosure_type = select_enclosure_type(stdscr)
                if enclosure_type is None:
                    continue  # User chose to go home

                if enclosure_type == 'bass reflex':
                    num_ports = select_ports(stdscr)
                    if num_ports is None:
                        continue  # User chose to go home

                    port_length = select_port_length(stdscr)
                    if port_length is None:
                        continue  # User chose to go home

                    port_radius = select_port_radius(stdscr)
                    if port_radius is None:
                        continue  # User chose to go home

                edge_length = select_edge_length(stdscr)
                if edge_length is None:
                    continue  # User chose to go home

                loudspeaker = select_loudspeaker(stdscr)
                if loudspeaker is None:
                    continue  # User chose to go home

                while True:
                    plot_type = select_plot_type(enclosure_type, stdscr)  # Pass enclosure type here
                    if plot_type is None:
                        break  # User chose to go home

                    if plot_type == 'power':
                        plot_style = select_plot_style(stdscr)
                        if plot_style is None:
                            continue

                    # Set up the parameters based on user choices
                    chosen_solid = {'edge': edge_length, 'type': solid_type}

                    frequencies, central_frequencies = setup_frequencies(plot_style, plot_type)

                    chosen_loudspeaker_params = next((speaker for speaker in parameters.list_of_loudspeakers if speaker['name'] == loudspeaker), None)

                    if solid_type == 'dodecahedron' and enclosure_type == 'closed box':
                        run_simulation_cb(DodecahedronEnclosure, parameters.clb_par, chosen_solid, chosen_loudspeaker_params, frequencies, central_frequencies, plot_style, plot_type)
                    elif solid_type == 'dodecahedron' and enclosure_type == 'bass reflex':
                        run_simulation_br(DodecahedronBassReflexEnclosure, parameters.clb_par, chosen_solid, parameters.empty_list_of_port_parameters, chosen_loudspeaker_params, frequencies, central_frequencies, plot_style, plot_type, num_ports, port_length, port_radius)
                    elif solid_type == 'icosidodecahedron' and enclosure_type == 'closed box':
                        run_simulation_cb(DodecahedronEnclosure, parameters.clb_par, chosen_solid, chosen_loudspeaker_params, frequencies, central_frequencies, plot_style, plot_type)
                    elif solid_type == 'icosidodecahedron' and enclosure_type == 'bass reflex':
                        run_simulation_br(DodecahedronBassReflexEnclosure, parameters.clb_par, chosen_solid, parameters.empty_list_of_port_parameters, chosen_loudspeaker_params, frequencies, central_frequencies, plot_style, plot_type, num_ports, port_length, port_radius)    
                    
                    while plot_type == 'power':
                        plot_style = select_plot_style(stdscr)
                        if plot_style is None:
                            break

                        # Set up the parameters based on user choices
                        chosen_solid = {'edge': edge_length, 'type': solid_type}

                        frequencies, central_frequencies = setup_frequencies(plot_style, plot_type)

                        chosen_loudspeaker_params = next((speaker for speaker in parameters.list_of_loudspeakers if speaker['name'] == loudspeaker), None)

                        if solid_type == 'dodecahedron' and enclosure_type == 'closed box':
                            run_simulation_cb(DodecahedronEnclosure, parameters.clb_par, chosen_solid, chosen_loudspeaker_params, frequencies, central_frequencies, plot_style, plot_type)
                        elif solid_type == 'dodecahedron' and enclosure_type == 'bass reflex':
                            run_simulation_br(DodecahedronBassReflexEnclosure, parameters.clb_par, chosen_solid, parameters.empty_list_of_port_parameters, chosen_loudspeaker_params, frequencies, central_frequencies, plot_style, plot_type, num_ports, port_length, port_radius)
                        elif solid_type == 'icosidodecahedron' and enclosure_type == 'closed box':
                            run_simulation_cb(DodecahedronEnclosure, parameters.clb_par, chosen_solid, chosen_loudspeaker_params, frequencies, central_frequencies, plot_style, plot_type)
                        elif solid_type == 'icosidodecahedron' and enclosure_type == 'bass reflex':
                            run_simulation_br(DodecahedronBassReflexEnclosure, parameters.clb_par, chosen_solid, parameters.empty_list_of_port_parameters, chosen_loudspeaker_params, frequencies, central_frequencies, plot_style, plot_type, num_ports, port_length, port_radius)    
                        
                        else:
                            break
                    else:
                            stdscr.clear()
                            stdscr.refresh()

def display_main_menu(stdscr):
    stdscr.clear()
    stdscr.addstr(0, 0, "Select a source:")
    stdscr.refresh()

def compare_results(results):
    best_cost = float('inf')
    for result in results:
        cost, speaker, edge, port_param, port, configuration, open_angle, time_passed = result
        if cost < best_cost:
            best_cost = cost
    return best_cost, speaker, edge, port_param, port, configuration, open_angle, time_passed

def display_results(stdscr, cost, speaker, edge, port_param, port, configuration, open_angle, time_passed):
    if cost is not None:
        stdscr.clear()
        stdscr.addstr(0, 0, f"Best configuration found: {configuration}")
        stdscr.addstr(1, 0, f"Cost: {cost}")
        stdscr.addstr(2, 0, f"Loudspeaker: {speaker}")
        stdscr.addstr(3, 0, f"Pentagon Edge: {edge['edge']} cm, Type: {edge['type']}")
        if port_param is not None:
            stdscr.addstr(4, 0, f"Port Parameters: Length: {port_param['length']} cm, Radius: {port_param['radius']} cm")
        else:
            stdscr.addstr(4, 0, "Port Parameters: None")                        
        stdscr.addstr(5, 0, f"Number of Ports: {port}")
        stdscr.addstr(6, 0, f"Open Angle: {open_angle}")
        stdscr.addstr(7, 0, f"Time taken: {time_passed} seconds")
        stdscr.refresh()
        stdscr.getch()

def select_pentagon_edges(stdscr, pentagon_edges):
    
    curses.curs_set(0)  # Hide cursor
    current_row = 0
    selected_edges = []

    while True:
        stdscr.clear()
        # Redraw the header
        stdscr.addstr(0, 0, "Available pentagon edges of" + (" Dodecahedron" if pentagon_edges[1]["type"] == "dodecahedron" else " Icosidodecahedron") + ":")

        # Display the edges with checkboxes and current selection
        for i, edge in enumerate(pentagon_edges, start=1):
            checkbox = "[X]" if edge in selected_edges else "[ ]"
            if i - 1 == current_row:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(i, 0, f"{checkbox} {edge['edge']}")
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(i, 0, f"{checkbox} {edge['edge']}")

        stdscr.addstr(len(pentagon_edges) + 1, 0, "Use arrow keys to navigate, Space to toggle selection, Enter to confirm selection, 'a' to select all, 'q' to deselect all, 'h' to return to Optimization Menu : ")
        stdscr.refresh()

        key = stdscr.getch()

        if key == curses.KEY_DOWN and current_row < len(pentagon_edges) - 1:
            current_row += 1
        elif key == curses.KEY_UP and current_row > 0:
            current_row -= 1         
        elif key == ord(' '):
            if pentagon_edges[current_row] in selected_edges:
                selected_edges.remove(pentagon_edges[current_row])
            else:
                selected_edges.append(pentagon_edges[current_row])
        elif key == ord('a'):
            selected_edges = pentagon_edges[:]
        elif key == ord('q'):
            selected_edges = []
        elif key == curses.KEY_ENTER or key in [10, 13]:
            if not selected_edges:  # No selection made
                stdscr.addstr(len(pentagon_edges) + 3, 0, "You must select at least one option! Press any key to continue...")
                stdscr.refresh()
                stdscr.getch()  # Wait for user to press a key
            else:
                break
        elif key == ord('h') :
            optimization_menu(stdscr)    

    return selected_edges

def select_loudspeakers(stdscr):
    
    curses.curs_set(0)  # Hide cursor
    current_row = 0
    selected_loudspeakers = []

    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "Available loudspeakers:")
        # Display the loudspeakers with checkboxes and current selection
        for i, speaker in enumerate(parameters.list_of_loudspeakers, start=1):
            checkbox = "[X]" if speaker in selected_loudspeakers else "[ ]"
            if i - 1 == current_row:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(i, 0, f"{checkbox} {speaker['name']}")
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(i, 0, f"{checkbox} {speaker['name']}")

        stdscr.addstr(len(parameters.list_of_loudspeakers) + 1, 0, "Use arrow keys to navigate, Space to toggle selection, Enter to confirm selection, 'a' to select all, 'q' to deselect all, 'h' to return to Optimization Menu : ")
        stdscr.refresh()

        key = stdscr.getch()

        if key == curses.KEY_DOWN and current_row < len(parameters.list_of_loudspeakers) - 1:
            current_row += 1
        elif key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == ord(' '):
            if parameters.list_of_loudspeakers[current_row] in selected_loudspeakers:
                selected_loudspeakers.remove(parameters.list_of_loudspeakers[current_row])
            else:
                selected_loudspeakers.append(parameters.list_of_loudspeakers[current_row])
        elif key == ord('a'):
            selected_loudspeakers = parameters.list_of_loudspeakers[:]
        elif key == ord('q'):
            selected_loudspeakers = []
        elif key == curses.KEY_ENTER or key in [10, 13]:
            if current_row == len(parameters.list_of_loudspeakers) - 1:
                return None
            elif not selected_loudspeakers:  # No selection made
                stdscr.addstr(len(parameters.list_of_loudspeakers) + 3, 0, "You must select at least one option! Press any key to continue...")
                stdscr.refresh()
                stdscr.getch()  # Wait for user to press a key
            else:
                break
        elif key == ord('h') :
            optimization_menu(stdscr)    

        stdscr.clear()

    return selected_loudspeakers
        
def select_port_parameters(stdscr):
    
    curses.curs_set(0)  # Hide cursor
    current_row_params = 0
    current_row_ports = 0
    selected_diode_parameters = []
    selected_ports = []

    # Select port parameters
    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "Available port parameters:")
        
        # Display port parameters with checkboxes and current selection
        for i, params in enumerate(parameters.port_parameters, start=1):
            checkbox = "[X]" if params in selected_diode_parameters else "[ ]"
            if i - 1 == current_row_params:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(i, 0, f"{checkbox} {i}. length: {params['length']}, radius: {params['radius']}")
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(i, 0, f"{checkbox} {i}. length: {params['length']}, radius: {params['radius']}")

        stdscr.addstr(len(parameters.port_parameters) + 2, 0, "Use arrow keys to navigate, Space to toggle selection, Enter to confirm selection, 'a' to select all, 'q' to deselect all, 'h' to return to Optimization Menu : ")
        stdscr.refresh()

        key = stdscr.getch()

        if key == curses.KEY_UP and current_row_params > 0:
            current_row_params -= 1
        elif key == curses.KEY_DOWN and current_row_params < len(parameters.port_parameters) - 1:
            current_row_params += 1
        elif key == ord(' '):
            if parameters.port_parameters[current_row_params] in selected_diode_parameters:
                selected_diode_parameters.remove(parameters.port_parameters[current_row_params])
            else:
                selected_diode_parameters.append(parameters.port_parameters[current_row_params])
        elif key == ord('a'):
            selected_diode_parameters = parameters.port_parameters[:]
        elif key == ord('q'):
            selected_diode_parameters = []
        elif key == curses.KEY_ENTER or key in [10, 13]:
            if not selected_diode_parameters:  # No selection made
                stdscr.addstr(len(parameters.port_parameters) + 3, 0, "You must select at least one port parameter! Press any key to continue...")
                stdscr.refresh()
                stdscr.getch()  # Wait for user to press a key
            else:
                break
        elif key == ord('h'):
            optimization_menu(stdscr)
    # Select number of ports
    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "Available number of ports:")
        
        # Display number of ports with checkboxes and current selection
        for i, num in enumerate(parameters.number_of_ports, start=1):
            checkbox = "[X]" if num in selected_ports else "[ ]"
            if i - 1 == current_row_ports:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(i, 0, f"{checkbox} {i}. {num} ports")
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(i, 0, f"{checkbox} {i}. {num} ports")

        stdscr.addstr(len(parameters.number_of_ports) + 2, 0, "Use arrow keys to navigate, Space to toggle selection, Enter to confirm selection, 'a' to select all, 'q' to deselect all, 'h' to return to Optimization Menu : ")
        stdscr.refresh()

        key = stdscr.getch()

        if key == curses.KEY_UP and current_row_ports > 0:
            current_row_ports -= 1
        elif key == curses.KEY_DOWN and current_row_ports < len(parameters.number_of_ports) - 1:
            current_row_ports += 1
        elif key == ord(' '):
            if parameters.number_of_ports[current_row_ports] in selected_ports:
                selected_ports.remove(parameters.number_of_ports[current_row_ports])
            else:
                selected_ports.append(parameters.number_of_ports[current_row_ports])
        elif key == ord('a'):
            selected_ports = parameters.number_of_ports[:]
        elif key == ord('q'):
            selected_ports = []
        elif key == curses.KEY_ENTER or key in [10, 13]:
            if current_row_params == len(parameters.port_parameters) - 1:
                return None, None
            elif not selected_ports:  # No selection made
                stdscr.addstr(len(parameters.number_of_ports) + 3, 0, "You must select at least one port! Press any key to continue...")
                stdscr.refresh()
                stdscr.getch()  # Wait for user to press a key
            else:
                break
        elif key == ord('h'):
            optimization_menu(stdscr)
    
    # selected_diode_parameters = [{"t": parameter["t"], "radius": parameter["radius"]} for parameter in selected_diode_parameters]
    return selected_diode_parameters, selected_ports




def select_algorithm(stdscr, source, pentagon_edges, loudspeakers, diode_params, num_ports, configuration):
    
    stdscr.clear()
    curses.curs_set(0)  # Hide cursor
    current_row = 0

    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "Select Algorithm:")
        stdscr.addstr(1, 0, "1. Genetic Algorithm")
        stdscr.addstr(2, 0, "2. Particle Swarm Optimization")
        stdscr.addstr(3, 0, "3. Differential Evolution")
        stdscr.addstr(4, 0, "Enter your choice: ")
        
        for i in range(1, 4):
            if i - 1 == current_row:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(i, 0, f" {'Genetic Algorithm' if i == 1 else 'Particle Swarm Optimization' if i == 2 else 'Differential Evolution'}")
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(i, 0, f" {'Genetic Algorithm' if i == 1 else 'Particle Swarm Optimization' if i == 2 else 'Differential Evolution'}")

        stdscr.addstr(5, 0, "Use arrow keys to navigate, Enter to confirm selection, 'h' to return to Optimization Menu : ")
        stdscr.refresh()

        key = stdscr.getch()

        if key == curses.KEY_UP:
            current_row = max(0, current_row - 1)
        elif key == curses.KEY_DOWN:
            current_row = min(2, current_row + 1)
        elif key == ord('h') :
            optimization_menu(stdscr)            
        elif key == curses.KEY_ENTER or key in [10, 13]:
            choice = current_row + 1  # Convert 0-based index to 1-based choice
            stdscr.clear()
            stdscr.addstr(0, 0, "Wait while the code is running...")  # Print the waiting message
            stdscr.refresh()
            if choice == 1:
                if source in ["1", "2"]:
                    cost, speaker, edge, port_param, port, configuration, open_angle, time_passed = run_pygad(loudspeakers, pentagon_edges, diode_params, num_ports, 'closed_box')
                elif source == "3":
                    cost, speaker, edge, port_param, port, configuration, open_angle, time_passed = run_pygad(loudspeakers, pentagon_edges, diode_params, num_ports, 'bass_reflex')
                elif source == "4":
                    if configuration == 'closed_box':
                        cost, speaker, edge, port_param, port, configuration, open_angle, time_passed = run_pygad(loudspeakers, pentagon_edges, diode_params, num_ports, configuration)
                    elif configuration == 'bass_reflex':
                        cost, speaker, edge, port_param, port, configuration, open_angle, time_passed = run_pygad(loudspeakers, pentagon_edges, diode_params, num_ports, configuration)
            elif choice == 2:
                if source in ["1", "2"]:
                    cost, speaker, edge, port_param, port, configuration, open_angle, time_passed = run_pso(loudspeakers, pentagon_edges, diode_params, num_ports, 'closed_box')
                elif source == "3":
                    cost, speaker, edge, port_param, port, configuration, open_angle, time_passed = run_pso(loudspeakers, pentagon_edges, diode_params, num_ports, 'bass_reflex')
                elif source == "4":
                    if configuration == 'closed_box':
                        cost, speaker, edge, port_param, port, configuration, open_angle, time_passed = run_pso(loudspeakers, pentagon_edges, diode_params, num_ports, configuration)
                    elif configuration == 'bass_reflex':
                        cost, speaker, edge, port_param, port, configuration, open_angle, time_passed = run_pso(loudspeakers, pentagon_edges, diode_params, num_ports, configuration)
            elif choice == 3:
                if source in ["1", "2"]:
                    cost, speaker, edge, port_param, port, configuration, open_angle, time_passed = run_differential_evolution(loudspeakers, pentagon_edges, diode_params, num_ports, 'closed_box')
                elif source == "3":
                    cost, speaker, edge, port_param, port, configuration, open_angle, time_passed = run_differential_evolution(loudspeakers, pentagon_edges, diode_params, num_ports, 'bass_reflex')
                elif source == "4":
                    if configuration == 'closed_box':
                        cost, speaker, edge, port_param, port, configuration, open_angle, time_passed = run_differential_evolution(loudspeakers, pentagon_edges, diode_params, num_ports, configuration)
                    elif configuration == 'bass_reflex':
                        cost, speaker, edge, port_param, port, configuration, open_angle, time_passed = run_differential_evolution(loudspeakers, pentagon_edges, diode_params, num_ports, configuration)
            break
        else:
            stdscr.addstr(5, 0, "Invalid choice!")
            stdscr.refresh()

    curses.curs_set(1)  # Restore cursor visibility
    stdscr.clear()
    return cost, speaker, edge, port_param, port, configuration, open_angle, time_passed



def select_enclosure_type(stdscr):
    curses.curs_set(0)  # Hide cursor
    stdscr.clear()
    stdscr.refresh()

    menu_items = ["Closed Box", "Bass Reflex", "Return to Select Individual Analysis Menu"]
    current_row = 0

    while True:
        stdscr.clear()
        stdscr.addstr("Select Enclosure Type:\n")
        
        # Display menu items
        for idx, item in enumerate(menu_items):
            if idx == current_row:
                stdscr.addstr(f" {item}\n", curses.A_REVERSE)
            else:
                stdscr.addstr(f" {item}\n")

        key = stdscr.getch()

        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(menu_items) - 1:
            current_row += 1
        elif key == curses.KEY_ENTER or key in [10, 13]:
            if current_row == len(menu_items) - 1:
                return None  # Return None to go back to previous menu
            else:
                stdscr.clear()
                stdscr.refresh()
                if current_row == 0:
                    return 'closed box'
                elif current_row == 1:
                    return 'bass reflex'

        stdscr.refresh()
                

def select_ports(stdscr):
    curses.curs_set(0)  # Hide cursor
    stdscr.clear()
    stdscr.refresh()

    menu_items = ["4 Ports", "8 Ports", "12 Ports", "16 Ports", "20 Ports", "Return to Select Individual Analysis Menu"]
    num_ports = [4, 8, 12, 16, 20, None]  # None corresponds to Return option
    current_row = 0

    while True:
        stdscr.clear()
        stdscr.addstr("Select Number of Ports:\n")
        
        # Display menu items
        for idx, item in enumerate(menu_items):
            if idx == current_row:
                stdscr.addstr(f" {item}\n", curses.A_REVERSE)
            else:
                stdscr.addstr(f" {item}\n")

        key = stdscr.getch()

        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(menu_items) - 1:
            current_row += 1
        elif key == curses.KEY_ENTER or key in [10, 13]:
            if current_row == len(menu_items) - 1:
                return None  # Return None to go back to previous menu
            else:
                return num_ports[current_row]

        stdscr.refresh()

def select_port_length(stdscr):
    curses.curs_set(0)  # Hide cursor
    stdscr.clear()
    stdscr.refresh()

    menu_items = ["0.018 m", "0.023 m", "0.028 m", "0.033 m", "0.038 m", "Return to Select Individual Analysis Menu"]
    port_lengths = [0.018, 0.023, 0.028, 0.033, 0.038, None]  # None corresponds to Return option
    current_row = 0

    while True:
        stdscr.clear()
        stdscr.addstr("Select Port Length:\n")
        
        # Display menu items
        for idx, item in enumerate(menu_items):
            if idx == current_row:
                stdscr.addstr(f" {item}\n", curses.A_REVERSE)
            else:
                stdscr.addstr(f" {item}\n")

        key = stdscr.getch()

        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(menu_items) - 1:
            current_row += 1
        elif key == curses.KEY_ENTER or key in [10, 13]:
            if current_row == len(menu_items) - 1:
                return None  # Return None to go back to previous menu
            else:
                return port_lengths[current_row]

        stdscr.refresh()

def select_port_radius(stdscr):
    curses.curs_set(0)  # Hide cursor
    stdscr.clear()
    stdscr.refresh()

    menu_items = ["0.0079 m", "0.0105 m", "0.0133 m", "0.0175 m", "0.0205 m", "Return to Select Individual Analysis Menu"]
    port_radius = [0.0079, 0.0105, 0.0133, 0.0175, 0.0205, None]  # None corresponds to Return option
    current_row = 0

    while True:
        stdscr.clear()
        stdscr.addstr("Select Port Radius:\n")
        
        # Display menu items
        for idx, item in enumerate(menu_items):
            if idx == current_row:
                stdscr.addstr(f" {item}\n", curses.A_REVERSE)
            else:
                stdscr.addstr(f" {item}\n")

        key = stdscr.getch()

        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(menu_items) - 1:
            current_row += 1
        elif key == curses.KEY_ENTER or key in [10, 13]:
            if current_row == len(menu_items) - 1:
                return None  # Return None to go back to previous menu
            else:
                return port_radius[current_row]

        stdscr.refresh()

def select_edge_length(stdscr):
    curses.curs_set(0)  # Hide cursor
    stdscr.clear()
    stdscr.refresh()

    menu_items = [
        "0.10 m", "0.11 m", "0.12 m", "0.13 m", "0.14 m", 
        "0.15 m", "0.16 m", "0.17 m", "0.18 m", "0.19 m", 
        "Return to Select Individual Analysis Menu"
    ]
    edge_lengths = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, None]  # None corresponds to Return option
    current_row = 0

    while True:
        stdscr.clear()
        stdscr.addstr("Select Pentagon Edge Length:\n")
        
        # Display menu items
        for idx, item in enumerate(menu_items):
            if idx == current_row:
                stdscr.addstr(f" {item}\n", curses.A_REVERSE)
            else:
                stdscr.addstr(f" {item}\n")

        key = stdscr.getch()

        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(menu_items) - 1:
            current_row += 1
        elif key == curses.KEY_ENTER or key in [10, 13]:
            if current_row == len(menu_items) - 1:
                return None  # Return None to go back to previous menu
            else:
                return edge_lengths[current_row]

        stdscr.refresh()
            
def select_loudspeaker(stdscr):
    curses.curs_set(0)  # Hide cursor
    stdscr.clear()
    stdscr.refresh()

    loudspeakers = parameters.list_of_loudspeakers
    menu_items = [loudspeaker['name'] for loudspeaker in loudspeakers]
    menu_items.append("Return to Select Individual Analysis Menu")
    current_row = 0

    while True:
        stdscr.clear()
        stdscr.addstr("Select Loudspeaker:\n")
        
        # Display menu items
        for idx, item in enumerate(menu_items):
            if idx == current_row:
                stdscr.addstr(f" {item}\n", curses.A_REVERSE)
            else:
                stdscr.addstr(f" {item}\n")

        key = stdscr.getch()

        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(menu_items) - 1:
            current_row += 1
        elif key == curses.KEY_ENTER or key in [10, 13]:
            if current_row == len(menu_items) - 1:
                return None  # Return None to go back to previous menu
            else:
                return menu_items[current_row]

        stdscr.refresh()

    
def select_plot_type(enclosure_type, stdscr):
    curses.curs_set(0)  # Hide cursor
    stdscr.clear()
    stdscr.refresh()

    if enclosure_type == 'bass reflex':
        plot_types = ['Impedance', 'Sound Power Lw', 'Port and Diaphragm Response', 'Return to Select Individual Analysis Menu']
        plot_type_mappings = {
            '1': 'impedance',
            '2': 'power',
            '3': 'port and diaphragm',
            '4': 'return to select plot style Menu'
        }
    else:
        plot_types = ['Impedance', 'Sound Power LW', 'Return to Individual Analysis Menu']
        plot_type_mappings = {
            '1': 'impedance',
            '2': 'power',
            '3': 'return to select plot style Menu'
        }
    
    current_row = 0  # Initialize current row

    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "Select Plot Type:\n")

        # Display menu items
        for idx, item in enumerate(plot_types):
            if idx == current_row:
                stdscr.addstr(f" {item}\n", curses.A_REVERSE)
            else:
                stdscr.addstr(f" {item}\n")
                
        stdscr.refresh()
        
        key = stdscr.getch()
        
        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(plot_types) - 1:
            current_row += 1
        elif key == curses.KEY_ENTER or key in [10, 13]:
            if current_row == len(plot_types) - 1:
                return None  # Return None to go back to previous menu
            else:
                choice = str(current_row + 1)
                if choice in plot_type_mappings:
                    return plot_type_mappings[choice]

                
                

def select_plot_style(stdscr):
    curses.curs_set(0)  # Hide cursor
    stdscr.clear()
    stdscr.refresh()

    plot_styles = ['1/3 Octave Bands', 'Octave Bands', 'Logarithmic Scale', 'Return to Select Plot Type Menu']
    plot_style_mappings = {
        '1': '1/3 octave',
        '2': 'octave',
        '3': 'logarithmic',
    }

    current_row = 0

    while True:
        stdscr.clear()
        stdscr.addstr("Select Plot Style:\n")

        # Display menu items
        for idx, item in enumerate(plot_styles):
            if idx == current_row:
                stdscr.addstr(f" {item}\n", curses.A_REVERSE)
            else:
                stdscr.addstr(f" {item}\n")
                
        stdscr.refresh()
        
        key = stdscr.getch()
        
        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(plot_styles) - 1:
            current_row += 1
        elif key == curses.KEY_ENTER or key in [10, 13]:
            if current_row == len(plot_styles) - 1:
                return None  # Return None to go back to previous menu
            else:
                choice = str(current_row + 1)
                if choice in plot_style_mappings:
                    return plot_style_mappings[choice]
                



# Function to set up frequency bands
def setup_frequencies(plot_style, plot_type):
    if plot_type == 'power':
        if plot_style == '1/3 octave':
            central_frequencies = [
                20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
                1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000
            ]
            frequencies = third_octave_bands(20, 8000, 3)
            frequencies = np.array(frequencies)
            frequencies = frequencies[(frequencies >= 20) & (frequencies <= 8000)]
        elif plot_style == 'octave':
            central_frequencies = [
                31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000
            ]
            frequencies = octave_bands(31.5, 8000, 1)
            frequencies = np.array(frequencies)
            frequencies = frequencies[(frequencies >= 31.5) & (frequencies <= 8000)]        
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
    ax2.set_yscale('linear')
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
    x_ticks = [25, 50, 100, 200, 400, 800, 1600, 3150]
    tick_indices = [central_freqs.index(x) for x in x_ticks]

    ax.bar(np.arange(len(power_1_3_octave)), power_1_3_octave, width=bar_width, align="center")
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(x_ticks)
    ax.set_title("Sound Power Lw in 1/3 Octave Bands")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("dB rel. 1pW")
    ax.grid(which="both", axis="y")
    ax.set_ylim(40, 100)
    plt.show()

    
def plot_power_octave(frequencies, power, central_freqs):
    power_octave = np.zeros(len(central_freqs) - 1)

    for j in range(len(central_freqs) - 1):
        start_idx = np.argmax(frequencies >= central_freqs[j])
        stop_idx = np.argmax(frequencies >= central_freqs[j + 1])

        power_octave[j] = np.mean(power[start_idx:stop_idx])

    fig, ax = plt.subplots()
    bar_width = 0.8
    x_ticks = [31.5, 63, 125, 250, 500, 1000, 2000, 4000]
    tick_indices = [central_freqs.index(x) for x in x_ticks]

    ax.bar(np.arange(len(power_octave)), power_octave, width=bar_width, align="center")
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(x_ticks)
    ax.set_title("Sound Power Lw in Octave Bands")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("dB rel. 1pW")
    ax.grid(which="both", axis="y")
    ax.set_ylim(40, 100)
    plt.show()

    
# Function to plot power in logarithmic scale
def plot_power_logarithmic(frequencies, power):
    fig, ax = plt.subplots()
    ax.plot(frequencies, power, 'b')
    ax.set_xscale('log')
    ax.set_yscale('linear')
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
    ax.set_yscale('linear')
    ax.set_title("Port and Diaphragm Response")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Response (dB)")
    ax.legend()
    ax.grid(True, which="both", linestyle='--')
    plt.show()



def main(stdscr):
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
    main_menu(stdscr)

def suppress_specific_warnings():
    warnings.filterwarnings("ignore")

if __name__ == "__main__":
    suppress_specific_warnings()
    curses.wrapper(main)