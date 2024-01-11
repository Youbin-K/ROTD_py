import numpy as np
import rotd_py.rotd_math as rotd_math
from scipy.integrate import simps

# This file is used to parse the calculated flux
# mainly from the surface_X.dat


def get_thermal(file_name, t_size, pes_index):
    # read through the file and return the total thermal flux
    # for potential energy surface index with pes_index
    # this function does not separate the contribution from each facet
    thermal_flux = np.zeros(t_size)
    lines = open(file_name).readlines()
    for i, line in enumerate(lines):
        if line.startswith('Canonical'):
            for j in range(0, t_size):
                thermal_flux[j] += float(lines[i+1+j].split()[1 + pes_index])
    return thermal_flux


def get_micro(file_name, e_size, pes_index):
    # read through the file and return the total thermal flux
    # this function does not separate the contribution from each facet
    e_flux = np.zeros(e_size)
    lines = open(file_name).readlines()
    for i, line in enumerate(lines):
        if line.startswith('Microcanonical'):
            for j in range(0, e_size):
                e_flux[j] += float(lines[i+1+j].split()[1 + pes_index])
    return e_flux


def get_ej_flux(file_name, e_size, j_size, pes_index):
    # read through the file and return the total thermal flux
    # this function does not separate the contribution from each facet
    ej_flux = np.zeros(e_size, j_size)
    lines = open(file_name).readlines()
    for k, line in enumerate(lines):
        if line.startswith('E-J resolved'):
            for i in range(0, e_size):
                for j in range(0, j_size):
                    ej_flux[i][j] += float(lines[k+i*j_size+j].split()[pes_index])

    return ej_flux


def integrate_micro(e_flux, energy_grid, temperature_grid, dof_num):
    """This function integrate the e_flux to thermal flux based on the
    e_flux and temperature_grid.

    Parameters
    ----------
    e_flux : 1_D numpy array
    energy_grid : with unit Kelv, same dimension with e_flux
    temperature_grid : with unit Kelv
    dof_num : the degree of freedom of the whole system

    Returns: thermal flux
    -------
    """

    if len(energy_grid) != len(e_flux):
        raise ValueError("mc_flux and energy dimension INVALID")
    energy_grid = energy_grid * rotd_math.Kelv
    temperature_grid = temperature_grid * rotd_math.Kelv
    temper_fac = np.power(temperature_grid, dof_num//2)
    if dof_num % 2:
        temper_fac *= np.sqrt(temperature_grid)
    mc_rate = np.zeros(len(temperature_grid))

    for t, temperature in enumerate(temperature_grid):

        enint = np.zeros(len(energy_grid))
        for e in range(0, len(energy_grid)):
            enint[e] = np.exp(-energy_grid[e]/temperature) * e_flux[e]

        # do the the integral
        # in the original rotd, a davint integral in slatec is used, which is a
        # overlapping parabolas fitting, thus here, we use simpson's integral in
        # scipy
        mc_rate[t] = simps(enint, energy_grid)
    mc_rate *= 4. * np.pi / temper_fac
    mc_rate *= rotd_math.conv_fac

    return mc_rate

def create_matplotlib_graph(x_lists=[[0., 1.]], data=[[1., 1.]], name="mtpltlb", x_label="x", y_label="y",\
                            data_legends=["y0"], comments=[""], exponential=False, splines=None, title=None):
    """Function that create the input for a 2D matplotlib plot.
    x_lists: List of lists of floats.
    data: List of lists of floats.
    name: String.
    x_label: String.
    y_label: String.
    data_legends: List of strings, same length as data.
    comments: List of strings.
    exponential: Boolean. Change the y-axis to exponential scale.
    """

    if splines == None:
        splines = [False for i in range(len(data))]

    if x_lists == None or not isinstance(x_lists, list):
        print("No x data found")
        return
    if not isinstance(data, list):
        return 
    
    content = """import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline\n\n"""

    for comment in comments:
        content += f"# {comment}\n"
    
    content += "ymin = np.inf\n"
    content += "ymax = -np.inf\n"

    for index, x in enumerate(x_lists):
        content += f"x{index} = {x}\n"
        content += f"x{index}_spln = np.arange(min(x{index}), max(x{index}), 0.01)\n\n"

    for index, y in enumerate(data):
        content += f"y{index} = {list(y)}\n"
        content += f"ymin = min(ymin, min(y{index}))\n"
        content += f"ymax = min(ymax, max(y{index}))\n"
        if splines[index]:
            content += f"spln{index} = make_interp_spline(x{index}, y{index})\n"
            content += f"y_spln{index} = spln{index}(x{index}_spln)\n"

    content += "\nfig, ax = plt.subplots()\n"

    for index, legend in enumerate(data_legends):
        if splines[index]:
            #content += f"ax.scatter(x{index}, y{index}, marker='x')\n"
            content += f"ax.plot(x{index}_spln, y_spln{index}, label='spln_{legend}')\n"
        else:
            content += f"ax.scatter(x{index}, y{index}, label='{legend}', marker='.')\n"

    if exponential:
        content += "ax.set_yscale('symlog')\n"

    if title != None and isinstance(title, str):
        content += f"ax.set_title('{title}')"

    content += f"""
ax.legend(loc='lower right')
ax.set_ylim([ymin*0.95, ymax*1.02])
ax.set_xlabel(r'{x_label}')
ax.set_ylabel(r'{y_label}')
plt.show()"""

    with open(f"{name}_plt.py", "w") as plt_file:
                plt_file.write(content)
