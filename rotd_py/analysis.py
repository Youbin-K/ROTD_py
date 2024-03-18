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
                thermal_flux[j] += float(lines[i+2+j].split()[1 + pes_index])
    return thermal_flux


def get_micro(file_name, e_size, pes_index):
    # read through the file and return the total thermal flux
    # this function does not separate the contribution from each facet
    e_flux = np.zeros(e_size)
    lines = open(file_name).readlines()
    for i, line in enumerate(lines):
        if line.startswith('Microcanonical'):
            for j in range(0, e_size):
                e_flux[j] += float(lines[i+2+j].split()[1 + pes_index])
    return e_flux


def get_ej_flux(file_name, e_size, j_size, pes_index, face=None):
    # read through the file and return the total thermal flux
    # this function separates the contribution from each facet
    ej_flux = np.zeros((e_size, j_size))
    lines = open(file_name).readlines()
    for k, line in enumerate(lines):
        if face != None:
            if 'Face:' in line:
                file_face = int(line.split()[1])
            if file_face == face:
                if line.startswith('E-J resolved'):
                    for i in range(0, e_size):
                        for j in range(0, j_size):
                            ej_flux[i][j] += float(lines[k+i*j_size+j+2].split()[2 + pes_index])
        else:
            if line.startswith('E-J resolved'):
                for i in range(0, e_size):
                    for j in range(0, j_size):
                        ej_flux[i][j] += float(lines[k+i*j_size+j+2].split()[2 + pes_index])

    return ej_flux


def integrate_micro(e_flux, energy_grid, temperature_grid, dof_num, return_contrib=False):
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
    energy_grid = energy_grid * rotd_math.Kelv # Convert from K to Hartree
    temperature_grid = temperature_grid * rotd_math.Kelv
    temper_fac = np.power(temperature_grid, dof_num//2)
    if dof_num % 2:
        temper_fac *= np.sqrt(temperature_grid)
    mc_rate = np.zeros(len(temperature_grid))
    mc_rate_contrib = np.zeros(len(temperature_grid))

    for t, temperature in enumerate(temperature_grid):

        enint = np.zeros(len(energy_grid))
        for e in range(0, len(energy_grid)):
            enint[e] = np.exp(-energy_grid[e]/temperature) * e_flux[e]

        # do the the integral
        # in the original rotd, a davint integral in slatec is used, which is a
        # overlapping parabolas fitting, thus here, we use simpson's integral in
        # scipy
            #SUBROUTINE DAVINT (X, Y, N, XLO, XUP, ANS, IERR)
            #     Description of Parameters
# C         The user must dimension all arrays appearing in the call list
# C              X(N), Y(N)
# C
# C         Input--
# C      X    - DOUBLE PRECISION array of abscissas, which must be in
# C             increasing order.
# C      Y    - DOUBLE PRECISION array of function values. i.e.,
# C                Y(I)=FUNC(X(I))
# C      N    - The integer number of function values supplied.
# C                N .GE. 2 unless XLO = XUP.
# C      XLO  - DOUBLE PRECISION lower limit of integration
# C      XUP  - DOUBLE PRECISION upper limit of integration.  Must have
# C              XLO.LE.XUP
# C
# C         Output--
# C      ANS  - Double Precision computed approximate value of integral
# C      IERR - A status code
# C           --Normal Code
# C                =1 Means the requested integration was performed.
# C           --Abnormal Codes
# C                =2 Means XUP was less than XLO.
# C                =3 Means the number of X(I) between XLO and XUP
# C                   (inclusive) was less than 3 and neither of the two
# C                   special cases described in the abstract occurred.
# C                   No integration was performed.
# C                =4 Means the restriction X(I+1).GT.X(I) was violated.
# C                =5 Means the number N of function values was .lt. 2.
# C                   ANS is set to zero if IERR=2,3,4,or 5.
        mc_rate[t] = simps(enint, energy_grid)
        mc_rate_contrib[t] = int(list(enint).index(max(enint)))


    mc_rate *= 4. * np.pi / temper_fac
    mc_rate *= rotd_math.conv_fac * 1e-11

    if return_contrib:
        return mc_rate, mc_rate_contrib
    else:
        return mc_rate

def integrate_ej(ej_mlux, amom_grid, ener_grid, return_contrib=False):
    """This function integrate the ej_mlux to microcanonical e_flux based on the
    angular momentum grid amom_grid.

    Parameters
    ----------
    ej_mlux : 2_D numpy array of dimension ener_grid*j_grid.
              Contains minimum ej_flux.
    ener_grid : with unit Kelv, same dimension with e_flux
    amom_grid : a.u., same dimension with ej_mlux axis=1

    Returns: e_flux
    -------
    """
    if (len(ener_grid),len(amom_grid)) != np.shape(ej_mlux):
        raise ValueError("ej_mlux and J*energy dimensions INVALID")
    e_flux = np.zeros(len(ener_grid))
    e_flux_contrib = np.zeros(len(ener_grid))
    ener_grid = ener_grid * rotd_math.Kelv # Convert from K to Hartree
    for en_ind, energy in enumerate(ener_grid):
        amint = np.zeros(len(amom_grid))
        for am_ind in range(0, len(amom_grid)):
            amint[am_ind] = amom_grid[am_ind]**2 * ej_mlux[en_ind, am_ind]
        #Integration
        e_flux[en_ind] = simps(amint, amom_grid)
        e_flux_contrib[en_ind] = int(list(amint).index(max(amint)))
    
    if return_contrib:
        return e_flux, e_flux_contrib
    else:
        return e_flux

def get_Ne_from_e(e_flux, fragments):
    """Convert the microcanonical flux into number of states"""
    reduced_mass = fragments[0].get_total_mass()*fragments[1].get_total_mass()/\
                  (fragments[0].get_total_mass()+fragments[1].get_total_mass())
    nfac = 2 * np.sqrt(2.*np.pi) * np.power(reduced_mass, 1.5)
    for frag in fragments:
        nfac *= frag.get_stat_sum()

    N_e = e_flux * nfac

    return N_e

def create_matplotlib_graph(x_lists=[[0., 1.]], data=[[1., 1.]],\
                            name="mtpltlb", x_label="x", y_label="y",\
                            data_legends=["y0"], comments=[""],\
                            xexponential=False, yexponential=False,\
                            splines=None, title=None,\
                            user_ymax=None, user_ymin=None,\
                            plot_type='scatter'):
    """Function that create the input for a 2D matplotlib plot.
    x_lists: List of lists of floats.
    data: List of lists of floats.
    name: String.
    x_label: String.
    y_label: String.
    data_legends: List of strings, same length as data.
    comments: List of strings.
    exponential: Boolean. Change corresponding axis to exponential scale.
    """

    if splines == None:
        splines = [False for i in range(len(data))]

    if x_lists == None or not isinstance(x_lists, list):
        print("No x data found")
        return
    if not isinstance(data, list):
        return 
    
    if '/' in name:
        for index, character in enumerate(name):
            if character == '/':
                tmp = list(name)
                tmp[index] = '_'
                tmp_lst = ''
                for char in tmp:
                    tmp_lst += char
                name = tmp_lst

    content = """import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline\n\n"""

    for comment in comments:
        content += f"# {comment}\n"
    
    content += "xmin = np.inf\n"
    content += "xmax = -np.inf\n"
    content += "ymin = np.inf\n"
    content += "ymax = -np.inf\n"

    for index, x in enumerate(x_lists):
        content += f"x{index} = {x}\n"
        content += f"xmin = min(xmin, min(x{index}))\n"
        content += f"xmax = max(xmax, max(x{index}))\n"
        content += f"x{index}_spln = np.arange(min(x{index}), max(x{index}), 0.01)\n\n"

    for index, y in enumerate(data):
        content += f"y{index} = {list(y)}\n"
        content += f"ymin = min(ymin, min(y{index}))\n"
        content += f"ymax = max(ymax, max(y{index}))\n"
        if splines[index]:
            content += f"spln{index} = make_interp_spline(x{index}, y{index})\n"
            content += f"y_spln{index} = spln{index}(x{index}_spln)\n"

    content += "\nfig, ax = plt.subplots()\n"

    for index, legend in enumerate(data_legends):
        if splines[index]:
            #content += f"ax.scatter(x{index}, y{index}, marker='x')\n"
            content += f"ax.plot(x{index}_spln, y_spln{index}, label='spln_{legend}')\n"
        else:
            if plot_type == 'scatter':
                content += f"ax.{plot_type}(x{index}, y{index}, label='{legend}', marker='.')\n"
            else:
                content += f"ax.{plot_type}(x{index}, y{index}, label='{legend}')\n"

    if xexponential:
        content += "ax.set_xscale('log')\n"
    if yexponential:
        content += "ax.set_yscale('log')\n"

    if title != None and isinstance(title, str):
        content += f"ax.set_title('{title}')"

    content += "\n"
    if user_ymin != None:
        content += f"ymin = {user_ymin}"
        content += "\n"
    if user_ymax != None:
        content += f"ymax = {user_ymax}"
        content += "\n"    

    content += f"""
ax.legend(loc='lower right')
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
ax.set_xlabel(r'{x_label}')
ax.set_ylabel(r'{y_label}')
plt.show()"""

    with open(f"{name}_plt.py", "w") as plt_file:
                plt_file.write(content)
