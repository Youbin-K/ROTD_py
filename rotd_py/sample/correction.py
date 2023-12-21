import numpy as np

import rotd_py
import rotd_py.rotd_math as rotd_math
from scipy.interpolate import make_interp_spline

class Correction():
    """Class that creates correction objects, which provide a correction potential for each sample"""
    def __init__(self, name:str, parameters:dict, sample):
        self.name = name
        self.sample = sample
        self.type = parameters["type"]
        self.default_energy = None
        parameter_missing = False
        match self.type:
            case "1d":
                necessary_data = ["r_sample", "r_trust", "e_sample", "e_trust", "scan_ref"]
                for key in necessary_data:
                    if key not in parameters:
                        self.logger.warning(f"1d correction: {key} is not set. 1d correction will not be applied")
                        self.default_energy = 0.
                        parameter_missing = True
                if not parameter_missing:
                    self.set_1d_correction(parameters)

    def set_1d_correction(self, parameters:dict):
        """Function that takes scan relative energies in Kcal
        and returns a spline corresponding to the 1d correction in eV."""
        self.set_scan_ref(parameters["scan_ref"])  # atom number of pivots, list of lists
        if parameters["r_sample"] == None or not isinstance(parameters["r_sample"], list):
            self.logger.warning("Error setting r_sample")
            self.default_energy = 0.
        else:
            self.r_sample = parameters["r_sample"]
        if parameters["e_sample"] == None or not isinstance(parameters["e_sample"], list):
            self.logger.warning("Error setting e_sample")
            self.default_energy = 0.
        else:
            self.e_sample = parameters["e_sample"]
        if parameters["r_trust"] == None or not isinstance(parameters["r_trust"], list):
            self.logger.warning("Error setting r_trust")
            self.default_energy = 0.
        else:
            self.r_trust = parameters["r_trust"]
        if parameters["e_trust"] == None or not isinstance(parameters["e_trust"], list):
            self.logger.warning("Error setting e_trust")
            self.default_energy = 0.
        else:
            self.e_trust = parameters["e_trust"]
        
        x_spln_1d_correction = np.arange(min(self.r_sample + self.r_trust), max(self.r_sample + self.r_trust), 0.01)
        
        spln_sample = make_interp_spline(self.r_sample, np.asarray(self.e_sample)*rotd_math.Kcal/rotd_math.Hartree)
        spln_trust = make_interp_spline(self.r_trust, np.asarray(self.e_trust)*rotd_math.Kcal/rotd_math.Hartree)

        y_spln_sample = spln_sample(x_spln_1d_correction)
        y_spln_trust = spln_trust(x_spln_1d_correction)

        y_1d_correction = np.subtract(np.asarray(y_spln_sample), np.asarray(y_spln_trust))

        self._1d_correction = make_interp_spline(x_spln_1d_correction, y_1d_correction)

    def set_scan_ref(self, scan_ref):
        self.scan_ref = []
        for scr in scan_ref:
            self.scan_ref.append([scr[0], scr[1]+len(self.sample.fragments[0].positions)])

        def energy(self, configuration):
            if self.default_energy != None:
                return self.default_energy
            match self.type:
                case "1d":
                    distance = np.inf
                    for scr in self.scan_ref:
                        distance = min(distance, np.absolute(np.linalg.norm(configuration.positions[scr[0]] -\
                                                            configuration.positions[scr[1]])))
                    return self._1d_correction(distance)

        