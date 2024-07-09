import sys
import importlib
from rotd_py.corrections.correction import Correction
from rotd_py.sample.sample import Sample
import numpy as np
from ase.atoms import Atoms


class Custom(Correction):
    """Custom correction class that execute the user's
    python code to calculate a correction energy."""

    def __init__(self,
                 name: str,
                 sample: Sample):

        super(Custom, self).__init__(name,
                                     sample)

    @property
    def necessary_keys(self) -> list:
        return ['file']

    def generate(self,
                 parameters: dict):
        """Check if the user's defined method exists.
        """

        self.type = parameters["type"].casefold()

        custom_path = parameters["file"]
        if not sys.path.isfile(custom_path):
            raise AttributeError('User defined file was not found.\
                                 Try providind absolute path.')
        
        custom_file = custom_path.split('/')[-1].rstrip('.py')
        custom_folder = custom_path.split(custom_file)[0].rstrip('/')
        sys.path.append(custom_folder)
        self.custom_module = importlib.import_module(custom_file)

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
        energy: float = self.custom_module(configuration,
                                           distance)
        return energy
