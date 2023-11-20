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
