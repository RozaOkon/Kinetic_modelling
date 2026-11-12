# Kinetic Modeling of Prebiotic Reaction Networks
This repository contains Python code for kinetic modeling of prebiotic reaction networks using free energy data obtained from quantum-chemical calculations.

# Contents
- `kinetic_modelling.py` – calculates reaction rate constants and simulates the time evolution of species concentrations.
- `input_activation_network.txt` – input file containing reaction data, protonation equilibria, diffusion-controlled processes, initial concentrations, and solution pH.

# Methodology
Reaction rate constants are calculated using the Eyring–Polanyi equation. The model also accounts for diffusion-controlled processes, acid–base equilibria, and conformational effects. The resulting system of kinetic differential equations is numerically integrated using `scipy.integrate.odeint`.

# Requirements
- Python 3
- NumPy
- SciPy
- Matplotlib

Install the required packages using:
pip install numpy scipy matplotlib

# Usage
Place the Python scripts and input file in the same directory and run:
python kinetic_model.py

The main settings can be adjusted directly in the main block of kinetic_model.py:
- input_file – name of the input file containing the reaction network and simulation parameters.
- simulation_time – total simulation time in seconds (default: 43200 s).
- n_points – number of time points used for numerical integration (default: 1000).
- products_to_print – list of products for which final yields are printed. Species names must match those defined in the input file.
- diff_barrier – diffusion-controlled activation barrier in kcal/mol (default: 4.0 kcal/mol).

The script generates concentration profiles as a function of time and prints the final yields of the selected products.

For a sanity check, the complete system of ordinary differential equations (ODEs) generated automatically from the input reaction network can also be printed by uncommenting the following line:

print_kinetic_equations(reactions, species)

These equations are subsequently integrated numerically by the solver.

# Reproducibility
The results depend on the reaction network, free energy values, initial concentrations, pH, temperature, and simulation settings specified in the input file and Python scripts.
