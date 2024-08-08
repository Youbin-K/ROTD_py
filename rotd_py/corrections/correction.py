import numpy as np

import rotd_py.rotd_math as rotd_math
from rotd_py.analysis import create_matplotlib_graph
#from rotd_py.sample.sample import Sample
import abc
from ase.atoms import Atoms


class Correction:
    """Parent class for all the corrections.
    All methods should be overwriten for a new correction.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 name: str,
                 parameters: dict,
                 sample):
        self.name = name
        self.sample = sample

    @property
    def necessary_keys(self) -> list:
        """Return list of necessary keywords
        for creating the correction object.

        Returns:
            list: contains necessary keywords (list of str)
        """
        return []

    @abc.abstractmethod
    def generate(self) -> None:
        """Method specific to internally build the correction object.
        """
        pass

    @abc.abstractmethod
    def energy(self,
               configuration: Atoms | None = None,
               distance: float = np.inf) -> float:
        """Return the correction energy (Hartree) for a given sample.

        Args:
            configuration (Sample): Configuration being sampled
            distance (float): distance between reactive atoms for
                              the facet being sampled (Angstrom).

        Returns:
            float: energy of the correction only (Hartree)
        """
        return 0.0

    def plot(self, xmin=0., xmax=20.):
        """Function that create a matplotlib plot of the correction"""
        x = np.arange(xmin, xmax, 0.01)
        y = [self.energy(distance=distance) /
             rotd_math.Hartree/rotd_math.Kcal for distance in x]

        lbl1 = "{}{}".format(
                self.sample.configuration.symbols[self.scan_ref[0][0]],
                self.scan_ref[0][0])
        lbl2 = "{}{}".format(
                self.sample.configuration.symbols[self.scan_ref[0][1]],
                self.scan_ref[0][1])

        create_matplotlib_graph(x_lists=[x.tolist()],
                                data=[y],
                                name=f"{self.sample.name}_1d_{self.name}",
                                x_label="{} to {} distance ($\AA$)".format(
                                        lbl1,
                                        lbl2),
                                y_label="Energy (Kcal/mol)",
                                data_legends=[f"Correction {self.name}"],
                                comments=[],
                                title=f"{self.name}({self.type}) energy correction")
