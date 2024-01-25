import numpy as np

import rotd_py.rotd_math as rotd_math
from scipy.interpolate import make_interp_spline
from rotd_py.analysis import create_matplotlib_graph

class Correction():
    """Class that creates correction objects, which provide a correction potential for each sample"""
    def __init__(self, name:str, parameters:dict, sample):
        self.name = name
        self.sample = sample
        self.type = parameters["type"].casefold()
        self.default_energy = None
        parameter_missing = False
        match self.type:
            case "1d":
                necessary_data = ["r_sample", "r_trust", "e_sample", "e_trust", "scan_ref"]
                for key in necessary_data:
                    if key not in parameters:
                        # self.logger.warning(f"1d correction: {key} is not set. 1d correction will not be applied")
                        self.default_energy = 0.
                        parameter_missing = True
                if not parameter_missing:
                    self.set_1d_correction(parameters)
            case "relaxed":
                pass
            case "counterpoise":
                # self.logger.warning(f"Counterpoise: correction is only implemented for gaussian calculator")
                self.set_counterpoise()
            case "custom":
                pass

    def set_counterpoise(self):
        self.fragments_indexes = []
        self.fragments_indexes.append([index for index in range(len(self.sample.fragments[0].symbols))])
        self.fragments_indexes.append([index+len(self.sample.fragments[0].symbols) for index in range(len(self.sample.fragments[1].symbols))])
        self.fragment1_charge = int(sum(self.sample.fragments[0].get_initial_charges()))
        self.fragment2_charge = int(sum(self.sample.fragments[1].get_initial_charges()))
        self.fragment1_mult = 1
        self.fragment2_mult = 1

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
        
        spln_sample = make_interp_spline(self.r_sample, np.asarray(self.e_sample)*rotd_math.Kcal*rotd_math.Hartree)
        spln_trust = make_interp_spline(self.r_trust, np.asarray(self.e_trust)*rotd_math.Kcal*rotd_math.Hartree)

        y_spln_sample = spln_sample(x_spln_1d_correction)
        y_spln_trust = spln_trust(x_spln_1d_correction)

        y_1d_correction = np.subtract(np.asarray(y_spln_trust), np.asarray(y_spln_sample))

        self._1d_correction = make_interp_spline(x_spln_1d_correction, y_1d_correction)

    def set_scan_ref(self, scan_ref):
        self.scan_ref = []
        for scr in scan_ref:
            self.scan_ref.append([scr[0], scr[1]+len(self.sample.fragments[0].positions)])
    
    def basis_set_superposition_error(self, configuration):
        with open(f'{configuration.calc.label}.log', 'r') as f:
            lines = f.readlines()
        for line in reversed(lines):
            if 'BSSE energy' in line:
                bsse = float(line.split()[3])*rotd_math.Hartree
                return bsse

        return 0.0

    def energy(self, configuration=None, distance=None):
        if self.default_energy != None:
            return self.default_energy
        match self.type:
            case "1d":
                if distance != None:
                    if distance > min(max(self.r_trust), max(self.r_sample)):
                        return 0.
                    elif distance < max(min(self.r_trust), min(self.r_sample)):
                        return 0.
                    else:
                        return self._1d_correction(distance)
                distance = np.inf
                for scr in self.scan_ref:
                    distance = min(distance, np.absolute(np.linalg.norm(configuration.positions[scr[0]] -\
                                                        configuration.positions[scr[1]])))
                    if distance > min(max(self.r_trust), max(self.r_sample)):
                        return 0.
                    elif distance < max(min(self.r_trust), min(self.r_sample)):
                        return 0.
                    else:
                        return self._1d_correction(distance)
            case 'counterpoise':
                return self.basis_set_superposition_error(configuration)
            case "relaxed":
                pass
        
                    
    def plot(self, xmin=0., xmax=20.):
        """Function that create a matplotlib plot of the correction"""
        x = np.arange(xmin, xmax, 0.01)
        y = [self.energy(distance=distance)/rotd_math.Hartree/rotd_math.Kcal for distance in x]

        comments = [f"Default energy: {self.default_energy}"]

        create_matplotlib_graph(x_lists=[x.tolist()], data=[y], name=f"{self.sample.name}_1d_{self.name}",\
                        x_label=f"{self.sample.configuration.symbols[self.scan_ref[0][0]]}{self.scan_ref[0][0]} to {self.sample.configuration.symbols[self.scan_ref[0][1]]}{self.scan_ref[0][1]} distance ($\AA$)",
                        y_label="Energy (Kcal/mol)", data_legends=[f"Correction {self.name}"],\
                        exponential=False, comments=comments, title=f"{self.name}({self.type}) energy correction")

        