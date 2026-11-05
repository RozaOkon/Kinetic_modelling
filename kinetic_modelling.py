import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Defining global variables
kB = 1.380649e-23  # J/K
h = 6.62607015e-34  # J*s
R = 8.314  # J/mol/K

def strip_comment(line):
    """Remove comments starting with #."""
    return line.split("#", 1)[0].strip()

def read_reactions_block(f):
    ''' Read elementary reactions and compute activation barriers. '''
    # Reading the file line by line
    lines = []
    for line in f:
        stripped = strip_comment(line)
        if stripped.lower() == "end":
            break
        if stripped:
            lines.append(stripped)

    # Separating the header from the data line
    header = [c.strip() for c in lines[0].split(",")]
    data_lines = lines[1:]

    # Checking if the input is complete and no columns are missing
    header_l = [h.lower() for h in header]
    required = ["substrates", "products", "g_sub", "g_ts", "g_prod"]

    for req in required:
        if req not in header_l:
            raise ValueError(f"Missing data column: '{req}'.")

    # Mapping lowercase column names to their original header names (case-insensitive access)
    header_map = {h.lower(): h for h in header}

    # Processing each reaction and saving it to the output list
    reactions = []
    for line in data_lines:
        parts = [c.strip() for c in line.split(",")]
        row = dict(zip(header, parts))

        def get(col):
            return row[header_map[col]]

        reactants = {s.strip(): 1 for s in get("substrates").split("+")}
        products  = {s.strip(): 1 for s in get("products").split("+")}

        G_sub  = float(get("g_sub"))
        G_TS   = float(get("g_ts"))
        G_prod = float(get("g_prod"))

        reactions.append({
            "reactants": reactants,
            "products": products,
            "barrier_fwd": G_TS - G_sub,
            "barrier_rev": G_TS - G_prod})
    return reactions

def read_protonation_equilibria(f):
    ''' Read protonation equilibria and free energies of proton transfer.'''
    lines = []
    for line in f:
        stripped = strip_comment(line)
        if stripped.lower() == "end":
            break
        if stripped:
            lines.append(stripped)

    header = [c.strip() for c in lines[0].split(",")]
    data_lines = lines[1:]

    header_l = [h.lower() for h in header]
    required = ["deprotonated form", "protonated form", "deltag_protonation"]

    for req in required:
        if req not in header_l:
            raise ValueError(f"Missing data column: '{req}'.")

    header_map = {h.lower(): h for h in header}
    protonation = []

    for line in data_lines:
        parts = [c.strip() for c in line.split(",")]
        row = dict(zip(header, parts))

        def get(col):
            return row[header_map[col]]

        protonation.append({
            "deprotonated": get("deprotonated form"),
            "protonated": get("protonated form"),
            "deltaG_protonation": float(get("deltag_protonation"))})

    return protonation

def read_diffusion_block(f):
    ''' Read diffusion - controlled (acid–base) reactions.'''
    # Reading the file line by line
    lines = []
    for line in f:
        stripped = strip_comment(line)
        if stripped.lower() == "end":
            break
        if stripped:
            lines.append(stripped)

    # Separating the header from the data line
    header = [c.strip() for c in lines[0].split(",")]
    data_lines = lines[1:]

    # Processing each reaction and saving it to the output list
    diffusion = []
    for line in data_lines:
        parts = [c.strip() for c in line.split(",")]
        row = dict(zip(header, parts))

        reactants = {row["deprotonated form"]: 1}
        products  = {row["protonated form"]: 1}

        diffusion.append({
            "reactants": reactants,
            "products": products,
            "pKa": row["pKa"]})
    return diffusion


def read_initial_conc_block(f):
    ''' Read initial species concentrations. '''
    conc = {}
    for line in f:
        stripped = strip_comment(line)
        if stripped.lower() == "end":
            break
        parts = [p.strip() for p in stripped.split(",")]
        species_name = parts[0]
        conc[species_name] = float(parts[1])
    return conc

def read_pH_block(f):
    ''' Read solution pH value. '''
    line = strip_comment(next(f))
    pH_value = float(line)
    for line in f:
        if strip_comment(line).lower() == "end":
            break
    return pH_value


def read_input(file_in):
    ''' Process input data and save it to relevant lists containing reactions,
    species, initial contentrations and pH. '''
    reactions = []
    protonation_equilibria = []
    diffusion_control = []
    conc_block = {}
    pH = None

    # Locates data blocks in the input starting with the * symbol and calls relevant functions to read them
    with open(file_in, "r") as f_in:
        for line in f_in:
            stripped = strip_comment(line)
            if stripped.startswith("*"):
                _, block_title = stripped.split("*", 1)
                block_title = block_title.strip().lower()
                if block_title == "reactions":
                    reactions = read_reactions_block(f_in)
                elif block_title == "protonation_equilibria":
                    protonation_equilibria = read_protonation_equilibria(f_in)
                elif block_title == "diffusion_control":
                    diffusion_control = read_diffusion_block(f_in)
                elif block_title == "initial_conc":
                    conc_block = read_initial_conc_block(f_in)
                elif block_title == "ph":
                    pH = read_pH_block(f_in)
                else:
                    print(f"Incorrect name of a block: {block_title}")

    # Creates a sorted list with species participating all the inputed reactions
    species_set = set()
    for r in reactions:
        species_set.update(r["reactants"].keys())
        species_set.update(r["products"].keys())
    for r in protonation_equilibria:
        species_set.add(r["deprotonated"])
        species_set.add(r["protonated"])
    for r in diffusion_control:
        species_set.update(r["reactants"].keys())
        species_set.update(r["products"].keys())
    species = sorted(list(species_set))

    # For all the species assigns their initial concentrations
    initial_conc = {}
    for s in species:
        initial_conc[s] = conc_block.get(s, 0.0)

    return reactions, protonation_equilibria, diffusion_control, species, initial_conc, pH



def compute_protonation_barrier(deltaG_protonation, diff_barrier, pKa_amph2=3.8, T=298.15):
    '''Compute the intrinsic barriers for protonation and deprotonation.'''

    # Free energy of deprotonation of AMPH2 in kcal/mol
    dG_diss_amph2 = 2.303 * R * T * pKa_amph2 / 4184

    # Free energy of A -> AH, obtained from the proton-transfer
    # reaction by Hess' law
    dG_protonation = deltaG_protonation - dG_diss_amph2

    # Free energy of the reverse reaction AH -> A
    dG_deprotonation = -dG_protonation

    # For an uphill reaction, add the kinetic barrier to the
    # thermodynamic free-energy difference.
    # For a downhill reaction, the barrier is diffusion-controlled.
    if dG_protonation > 0:
        barrier_protonation = dG_protonation + diff_barrier
    else:
        barrier_protonation = diff_barrier

    if dG_deprotonation > 0:
        barrier_deprotonation = dG_deprotonation + diff_barrier
    else:
        barrier_deprotonation = diff_barrier

    #print(diff_barrier)

    return (
        dG_protonation,
        dG_deprotonation,
        barrier_protonation,
        barrier_deprotonation
    )

def compute_effective_protonation_rates(deprotonated, protonated, deltaG_protonation, diff_barrier, pH, pKa_amph2=3.8, T=298.15):

    (dG_protonation, dG_deprotonation, barrier_protonation, barrier_deprotonation) = compute_protonation_barrier(deltaG_protonation, diff_barrier, pKa_amph2=pKa_amph2, T=T)

    # Intrinsic rate constant for A -> AH
    k_protonation = (kB * T / h * np.exp(-(barrier_protonation * 4184) / (R * T))
    )

    # Protonation is bimolecular with H3O+,
    # so convert the intrinsic rate to an effective pseudo-first-order rate.
    h3o_conc = 10 ** (-pH)
    k_eff_protonation = k_protonation * h3o_conc

    # Intrinsic rate constant for AH -> A
    # No H3O+ factor: this is the reverse deprotonation process.
    k_deprotonation = (kB * T / h * np.exp(-(barrier_deprotonation * 4184) / (R * T)))
    return [{"reactants": {deprotonated: 1}, "products": {protonated: 1}, "k": k_eff_protonation},
            {"reactants": {protonated: 1}, "products": {deprotonated: 1}, "k": k_deprotonation}]

def calculate_rate_constants(reactions, protonation_equilibria, diffusion, pH, diff_barrier = 4.0, cutoff=24.0, T = 298.15):
    ''' Compute kinetic rate constants from free energy barriers. '''
    solver_reactions = []
    for r in reactions:
        # forward barrier from the Eyring-Polanyi equation
        dGf_J = r["barrier_fwd"] * 4184 # converting kcal to J
        k_fwd = (kB * T / h) * np.exp(-dGf_J / (R * T))
        # solving a forward reaction with its rate to a list
        solver_reactions.append({
            "reactants": r["reactants"],
            "products": r["products"],
            "k": k_fwd})

        # calculating the reverse reaction rate if its barrier is lower than a defined cutoff
        if r["barrier_rev"] <= cutoff:
            dGr_J = r["barrier_rev"] * 4184
            k_rev = (kB * T / h) * np.exp(-dGr_J / (R * T))
            # solving a reverse reaction with its rate to a list
            solver_reactions.append({
                "reactants": r["products"],
                "products": r["reactants"],
                "k": k_rev})
        else:
            # if the reverse reaction is too slow -> ignore
            pass

    # Protonation/deprotonation reactions
    for r in protonation_equilibria:
        protonation_reactions = compute_effective_protonation_rates(
            r["deprotonated"],
            r["protonated"],
            r["deltaG_protonation"],
            diff_barrier,
            pH,
            pKa_amph2=3.8,
            T=T)
        solver_reactions.extend(protonation_reactions)

    # getting forward and reverse reaction rates for diffusion-controlled reactions
    for r in diffusion:
        dGf_J = diff_barrier * 4184
        k_fwd = (kB * T / h) * np.exp(-dGf_J / (R * T))
        k_eff = k_fwd * 10**(-pH)
        # add forward reaction
        solver_reactions.append({
            "reactants": r["reactants"],
            "products": r["products"],
            "k": k_eff})

        k_rev = k_fwd * 10**(-float(r["pKa"]))
        # add reverse reaction
        solver_reactions.append({
            "reactants": r["products"],
            "products": r["reactants"],
            "k": k_rev})
    return solver_reactions


def dCdt(conc, reactions, species_list):
    ''' Evaluate time derivatives of species concentrations. '''
    dcdt = {s: 0.0 for s in species_list}
    for r in reactions:
        rate = r["k"]
        for reactant, stoich in r["reactants"].items():
            rate *= conc[reactant] ** stoich
        for reactant, stoich in r["reactants"].items():
            dcdt[reactant] -= stoich * rate
        for product, stoich in r["products"].items():
            dcdt[product] += stoich * rate
    return [dcdt[s] for s in species_list]


def solver(reactions, species, initial_conc, simulation_time = 43200, n_points = 1000):
    ''' Integrate the kinetic ODE system. '''
    t_plot = np.linspace(0, simulation_time, n_points)
    t_hours = [x / 3600 for x in t_plot]

    # defining the function for odeint: f(y, t)
    def rhs(y, t):
        conc_dict = dict(zip(species, y))
        return dCdt(conc_dict, reactions, species)

    # odeint(Y0, t)
    y = odeint(rhs, [initial_conc[s] for s in species], t_plot, rtol=1e-8, atol=1e-12, mxstep=50000).T

    return t_hours, y

def plotter(species, t_plot, y):
    ''' Plot concentration profiles versus time. '''
    plt.figure(figsize=(8, 5))
    # selected_species = {"MeNC", "MI"}
    # for i, s in enumerate(species):
    #     if s in selected_species:
    #         plt.plot(t_plot, y[i])
    for i, s in enumerate(species):
        plt.plot(t_plot, y[i], label=s)
    plt.xlabel("Time [h]")
    plt.ylabel("Concentration")
    plt.legend()
    plt.grid(True)
    # plt.ylim(0, 0.01)
    # plt.xlim(left=0.01)

def print_yields(species, y, products, precision = 2):
    for i, s in enumerate(species):
        if s in products:
            print(f"{s} yield: {y[species.index(s)][-1]/0.01 * 100:.{precision}f}%")

def print_kinetic_equations(reactions, species_list):
    ''' Print symbolic kinetic equations for sanity-check. '''
    for s in species_list:
        terms = []
        reactants = []
        for idx, r in enumerate(reactions, 1):
            if s in r["reactants"]:
                for i in r["reactants"].keys():
                    reactants.append(f"[{i}]")
                react_str = " * ".join(reactants)
                terms.append(f"-k{idx} {react_str}")
                reactants = []
            if s in r["products"]:
                for i in r["reactants"].keys():
                    reactants.append(f"[{i}]")
                react_str = " * ".join(reactants)
                terms.append(f"k{idx} {react_str}")
                reactants = []
        eq_str = " + ".join(terms).replace("+ -", "- ")
        print(f"d[{s}]/dt = {eq_str}")


if __name__ == '__main__':
    # reading the data from the input
    input_file = "input_activation_network.txt"
    reactions, protonation_equilibria, diffusion_control, species, initial_conc, pH = read_input(input_file)
    reactions = calculate_rate_constants(reactions, protonation_equilibria, diffusion_control, pH, diff_barrier = 4.0)

    # running the solver
    simulation_time = 43200
    n_points = 1000
    t_hours, y =solver(reactions, species, initial_conc, simulation_time, n_points)

    # plotting the results
    plotter(species, t_hours, y)
    plt.show()

    # printing the yields of the selected products, names need to match the ones from the input file
    products_to_print = ["23-cAMP", "2-Ac-3-AMPH"]
    print_yields(species, y, products_to_print, precision = 6)

    # printing the full system of ODEs to integrate (for sanity check)
    # print_kinetic_equations(reactions, species)
