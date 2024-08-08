from rotd_py.corrections.correction import Correction
from rotd_py.sample.sample import Sample
import rotd_py.rotd_math as rotd_math
from scipy.interpolate import make_interp_spline
import numpy as np
from ase.atoms import Atoms


class OneDimensional(Correction):
    def __init__(self,
                 name: str,
                 sample: Sample):

        super(OneDimensional, self).__init__(name,
                                             sample)

    @property
    def necessary_keys(self) -> list[str]:
        return ["r_sample", "r_trust", "e_sample", "e_trust", "scan_ref"]

    def generate(self,
                 parameters: dict):
        """Function that takes scan relative energies in Kcal
        and returns a spline corresponding to the 1d correction in eV."""

        self.type = parameters["type"].casefold()

        # atom number of pivots, list of lists
        self.set_scan_ref(parameters["scan_ref"])
        if parameters["r_sample"] is None or\
           not isinstance(parameters["r_sample"], list):
            raise AttributeError("1D correction needs 'r_sample' list.")
        else:
            self.r_sample = parameters["r_sample"]
        if parameters["e_sample"] is None or\
           not isinstance(parameters["e_sample"], list):
            raise AttributeError("1D correction needs 'e_sample' list.")
        else:
            self.e_sample = parameters["e_sample"]
        if parameters["r_trust"] is None or\
           not isinstance(parameters["r_trust"], list):
            raise AttributeError("1D correction needs 'r_trust' list.")
        else:
            self.r_trust = parameters["r_trust"]
        if parameters["e_trust"] is None or\
           not isinstance(parameters["e_trust"], list):
            raise AttributeError("1D correction needs 'e_trust' list.")
        else:
            self.e_trust = parameters["e_trust"]

        x_spln_1d_correction = np.arange(min(self.r_sample + self.r_trust),
                                         max(self.r_sample + self.r_trust),
                                         0.01)

        spln_sample = make_interp_spline(self.r_sample,
                                         np.asarray(self.e_sample) *
                                         rotd_math.Kcal*rotd_math.Hartree)
        spln_trust = make_interp_spline(self.r_trust,
                                        np.asarray(self.e_trust) *
                                        rotd_math.Kcal*rotd_math.Hartree)

        y_spln_sample = spln_sample(x_spln_1d_correction)
        y_spln_trust = spln_trust(x_spln_1d_correction)

        y_1d_correction = np.subtract(np.asarray(y_spln_trust),
                                      np.asarray(y_spln_sample))

        self._1d_correction = make_interp_spline(x_spln_1d_correction,
                                                 y_1d_correction)

    def set_scan_ref(self, scan_ref):
        self.scan_ref = []
        for scr in scan_ref:
            self.scan_ref.append([scr[0],
                                  scr[1] +
                                  len(self.sample.fragments[0].positions)])

    def energy(self,
               configuration: Atoms | None = None,
               distance: float = np.inf) -> float:
        if distance != np.inf:
            if distance > min(max(self.r_trust), max(self.r_sample)):
                return 0.
            elif distance < max(min(self.r_trust), min(self.r_sample)):
                return self._1d_correction(max(min(self.r_trust),
                                               min(self.r_sample)))
            else:
                return self._1d_correction(distance)
        for scr in self.scan_ref:
            distance = min(distance,
                           np.absolute(
                             np.linalg.norm(configuration.positions[scr[0]] -
                                            configuration.positions[scr[1]])))
        if distance > min(max(self.r_trust), max(self.r_sample)):
            return 0.
        elif distance < max(min(self.r_trust), min(self.r_sample)):
            return 0.
        else:
            return self._1d_correction(distance)
