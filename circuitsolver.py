import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sys import argv, exit

CAPACITOR_RESISTANCE = 1e12  # Very high resistance for open circuit
INDUCTOR_RESISTANCE = 1e-12  # Very low resistance for short circuit
CIRCUIT_START = '.circuit'
CIRCUIT_END = '.end'

class CircuitElement:
    def __init__(self, name, node1, node2, value, initial_condition=0):
        self.name = name
        self.node1 = node1
        self.node2 = node2
        self.value = float(value)
        self.initial_condition = initial_condition
        self.current = 0.0  # Store current through element
        self.voltage = 0.0  # Store voltage across element

def parse_circuit(file_path):
    """Enhanced circuit parsing with robust error handling."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find circuit definition boundaries
        start = next((i for i, line in enumerate(lines) if line.startswith(CIRCUIT_START)), -1)
        end = next((i for i, line in enumerate(lines) if line.startswith(CIRCUIT_END)), -1)

        if start == -1 or end == -1 or start >= end:
            raise ValueError("Invalid circuit definition.")
        
        elements = []
        for line in lines[start + 1:end]:
            # Skip comments and empty lines
            line = line.split('#')[0].strip()
            if not line:
                continue
            
            tokens = line.split()
            if len(tokens) < 4:
                continue
            
            name, node1, node2, value = tokens[:4]
            # Normalize ground node
            node1 = '0' if node1 == 'GND' else node1
            node2 = '0' if node2 == 'GND' else node2
            
            initial_condition = float(tokens[4]) if len(tokens) > 4 else 0
            
            elements.append(CircuitElement(name, node1, node2, value, initial_condition))
        
        print("Parsed Elements:")
        for elem in elements:
            print(f"{elem.name}: {elem.__dict__}")
        return elements

    except (IOError, ValueError) as e:
        print(f"Error parsing circuit: {e}")
        exit(1)

def steady_state_analysis(elements):
    """Perform steady-state DC circuit analysis."""
    # Create node list and count voltage sources
    node_list = []
    vsrc_count = 0
    for element in elements:
        node_list.append(element.node1)
        node_list.append(element.node2)
        if element.name[0] == 'V':
            vsrc_count += 1

    # Get unique nodes
    node_set = set(node_list)
    if '0' in node_set:
        node_set.remove('0')
    
    # Size of the system matrix
    size = len(node_set) + vsrc_count
    
    # Create system matrices
    M = np.zeros((size, size))
    b = np.zeros(size)
    
    # Convert nodes to indices (GND = 0 is not included in matrix)
    node_map = {node: idx for idx, node in enumerate(node_set, 1)}
    
    vsrc_idx = len(node_set)
    # Fill matrices for each element
    for element in elements:
        n1 = 0 if element.node1 == '0' else node_map[element.node1]
        n2 = 0 if element.node2 == '0' else node_map[element.node2]
        
        if element.name[0] == 'R':  # Resistor
            conductance = 1/float(element.value)
            if n1 > 0:
                M[n1-1][n1-1] += conductance
                if n2 > 0:
                    M[n1-1][n2-1] += -conductance
            if n2 > 0:
                M[n2-1][n2-1] += conductance
                if n1 > 0:
                    M[n2-1][n1-1] += -conductance
        
        elif element.name[0] == 'C':  # Capacitor (in steady state = open circuit)
            conductance = 1/CAPACITOR_RESISTANCE
            if n1 > 0:
                M[n1-1][n1-1] += conductance
                if n2 > 0:
                    M[n1-1][n2-1] += -conductance
            if n2 > 0:
                M[n2-1][n2-1] += conductance
                if n1 > 0:
                    M[n2-1][n1-1] += -conductance

        elif element.name[0] == 'L':  # Inductor (in steady state = short circuit)
            conductance = 1/INDUCTOR_RESISTANCE
            if n1 > 0:
                M[n1-1][n1-1] += conductance
                if n2 > 0:
                    M[n1-1][n2-1] += -conductance
            if n2 > 0:
                M[n2-1][n2-1] += conductance
                if n1 > 0:
                    M[n2-1][n1-1] += -conductance

        elif element.name[0] == 'V':  # Voltage Source
            if n1 > 0:
                M[n1-1][vsrc_idx] += 1
                M[vsrc_idx][n1-1] += 1
            if n2 > 0:
                M[n2-1][vsrc_idx] += -1
                M[vsrc_idx][n2-1] += -1
            b[vsrc_idx] = float(element.value)
            vsrc_idx += 1
        
        elif element.name[0] == 'I':  # Current Source
            if n1 > 0:
                b[n1-1] -= float(element.value)
            if n2 > 0:
                b[n2-1] += float(element.value)

    # Solve the system
    try:
        x = np.linalg.solve(M, b)
        
        # Calculate voltages and currents for each element
        for element in elements:
            n1 = 0 if element.node1 == '0' else node_map[element.node1]
            n2 = 0 if element.node2 == '0' else node_map[element.node2]
            
            # Calculate voltage across element
            v1 = 0 if n1 == 0 else x[n1-1]
            v2 = 0 if n2 == 0 else x[n2-1]
            element.voltage = v1 - v2
            
            # Calculate current through element
            if element.name[0] == 'R':
                element.current = element.voltage / float(element.value)
            elif element.name[0] == 'C':
                element.current = element.voltage / CAPACITOR_RESISTANCE
            elif element.name[0] == 'L':
                element.current = element.voltage / INDUCTOR_RESISTANCE
            elif element.name[0] == 'V':
                vsrc_num = sum(1 for e in elements if e.name[0] == 'V' and e.name <= element.name)
                element.current = x[len(node_set) + vsrc_num - 1]
            elif element.name[0] == 'I':
                element.current = float(element.value)

        print("\n=== Steady-State Circuit Analysis Results ===")
        print("\nNode Voltages:")
        print("--------------")
        for node in sorted(node_set):
            print(f"V{node} = {x[node_map[node]-1]:.4f}V")
        
        print("\nElement Analysis:")
        print("----------------")
        for element in elements:
            print(f"\n{element.name}:")
            print(f"  Voltage: {element.voltage:.4f}V")
            print(f"  Current: {element.current:.4f}A")
            if element.name[0] in ['R', 'C', 'L']:
                power = element.voltage * element.current
                print(f"  Power: {power:.4f}W")
        
        return x, node_set, node_map
                
    except np.linalg.LinAlgError:
        print("Error: Circuit matrix is singular. Check if the circuit is valid.")
        return None, None, None
def transient_analysis(elements):
    """Perform transient analysis of RLC circuits."""
    s = sp.Symbol('s')  # Laplace variable
    t = sp.Symbol('t', positive=True)  # Time variable
    
    # Identify nodes and components
    nodes = set()
    voltage_sources = []
    
    for elem in elements:
        nodes.update([elem.node1, elem.node2])
        if elem.name.startswith('V'):
            voltage_sources.append(elem)
    
    # Remove ground node
    nodes.discard('0')
    
    # Create node mapping
    node_map = {node: idx for idx, node in enumerate(sorted(nodes), start=1)}
    
    # Symbolic matrices
    num_nodes = len(nodes)
    num_sources = len(voltage_sources)
    matrix_size = num_nodes + num_sources
    
    M = sp.zeros(matrix_size, matrix_size)
    b = sp.zeros(matrix_size, 1)
    
    # Helper functions
    def get_node_index(node):
        """Get matrix index for a node."""
        return 0 if node == '0' else node_map[node]
    
    # Process circuit elements
    vsrc_offset = num_nodes
    
    for elem in elements:
        n1 = get_node_index(elem.node1)
        n2 = get_node_index(elem.node2)
        
        # Resistor: G matrix contribution
        if elem.name.startswith('R'):
            conductance = 1 / elem.value
            if n1 > 0:
                M[n1-1, n1-1] += conductance
                if n2 > 0:
                    M[n1-1, n2-1] -= conductance
            if n2 > 0:
                M[n2-1, n2-1] += conductance
                if n1 > 0:
                    M[n2-1, n1-1] -= conductance
        
        # Inductor: s*L in Laplace domain with initial condition
        elif elem.name.startswith('L'):
            impedance = s * elem.value
            initial_current = elem.initial_condition
            if n1 > 0:
                M[n1-1, n1-1] += 1/impedance
                if n2 > 0:
                    M[n1-1, n2-1] -= 1/impedance
            if n2 > 0:
                M[n2-1, n2-1] += 1/impedance
                if n1 > 0:
                    M[n2-1, n1-1] -= 1/impedance
            
            # Add initial condition contribution
            if initial_current:
                b[n1-1 if n1 > 0 else n2-1] += initial_current
        
        # Capacitor: 1/(s*C) in Laplace domain with initial voltage
        elif elem.name.startswith('C'):
            impedance = 1 / (s * elem.value)
            initial_voltage = elem.initial_condition
            if n1 > 0:
                M[n1-1, n1-1] += 1/impedance
                if n2 > 0:
                    M[n1-1, n2-1] -= 1/impedance
            if n2 > 0:
                M[n2-1, n2-1] += 1/impedance
                if n1 > 0:
                    M[n2-1, n1-1] -= 1/impedance
            
            # Add initial condition contribution
            if initial_voltage:
                b[n1-1 if n1 > 0 else n2-1] += initial_voltage * elem.value * s
        
        # Voltage Source
        elif elem.name.startswith('V'):
            if n1 > 0:
                M[n1-1, vsrc_offset] = 1
                M[vsrc_offset, n1-1] = 1
            if n2 > 0:
                M[n2-1, vsrc_offset] = -1
                M[vsrc_offset, n2-1] = -1
            
            # Laplace transform of DC source
            b[vsrc_offset] = elem.value / s
            vsrc_offset += 1
    
    # Solve the circuit
    try:
        solution = sp.linsolve((M, b))
        
        if not solution:
            print("No solution found for the circuit.")
            return {}
        
        # Extract node voltages
        node_voltages = list(solution)[0][:num_nodes]
        node_voltage_map = {f"V{idx+1}": volt for idx, volt in enumerate(node_voltages)}
        
        print("\nNode Voltages (Laplace Domain):")
        for node, voltage in node_voltage_map.items():
            print(f"{node}: {voltage}")
        
        return node_voltage_map
    
    except Exception as e:
        print(f"Circuit analysis error: {e}")
        return {}

def plot_transient_responses(transient_responses):
    """Advanced transient response plotting with extended time range for saturation."""
    t = sp.Symbol('t', positive=True)
    s = sp.Symbol('s')
    
    plt.figure(figsize=(12, 7))
    
    for label, response in transient_responses.items():
        try:
            # Dynamically determine the time range
            time_points = np.logspace(-3, 2, 1000)  # Extend time range to cover saturation
            
            if isinstance(response, sp.Mul) and response.args[1] == 1/s:
                # Constant DC voltage
                dc_value = response.args[0]
                numeric_response = dc_value * np.ones_like(time_points)
                plt.semilogx(time_points, numeric_response, label=label)
            else:
                # Attempt inverse Laplace transform with error handling
                try:
                    simplified_response = sp.simplify(response)
                    inverse_response = sp.inverse_laplace_transform(simplified_response, s, t)
                    
                    # Compute numeric response
                    numeric_func = sp.lambdify(t, inverse_response, modules=['numpy'], dummify=True)
                    numeric_response = numeric_func(time_points)
                    
                    plt.semilogx(time_points, numeric_response, label=label)
                except Exception as e:
                    print(f"Inverse transform failed for {label}: {e}")
                    continue
        
        except Exception as e:
            print(f"Failed to process {label}: {e}")
    
    plt.title("RLC Circuit Transient Responses")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    if len(argv) != 2:
        print("Usage: python comprehensive_circuit_solver.py <input_file>")
        exit(1)

    # Parse circuit elements
    elements = parse_circuit(argv[1])

    # Perform steady-state analysis
    steady_state_result = steady_state_analysis(elements)
    
    # Perform transient analysis
    transient_responses = transient_analysis(elements)
    
    # Plot transient responses if available
    if transient_responses:
        plot_transient_responses(transient_responses)
    else:
        print("No transient responses available.")

if __name__ == "__main__":
    main()