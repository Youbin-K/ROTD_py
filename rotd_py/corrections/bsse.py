from rotd_py.corrections.correction import Correction
from rotd_py.sample.sample import Sample
import rotd_py.rotd_math as rotd_math
import numpy as np
from ase.atoms import Atoms


class BSSE(Correction):
    def __init__(self,
                 name: str,
                 sample: Sample):

        super(BSSE, self).__init__(name,
                                   sample)

    @property
    def necessary_keys(self):
        return []

    def generate(self,
                 parameters: dict):

        self.type = parameters["type"].casefold()

        self.fragments_indexes = []
        self.fragments_indexes.append([index 
                                       for index in range(len(self.sample.fragments[0].symbols))])
        self.fragments_indexes.append([index+len(self.sample.fragments[0].symbols)
                                       for index in range(len(self.sample.fragments[1].symbols))])
        self.fragment1_charge = int(sum(self.sample.fragments[0].get_initial_charges()))
        self.fragment2_charge = int(sum(self.sample.fragments[1].get_initial_charges()))
        self.fragment1_mult = 1
        self.fragment2_mult = 1

    def energy(self,
               configuration: Atoms | None = None,
               distance: float = np.inf) -> float:
        with open(f'{configuration.calc.label}.log', 'r') as f:
            lines = f.readlines()
        for line in reversed(lines):
            if 'BSSE energy' in line:
                bsse = float(line.split()[3])*rotd_math.Hartree
                return bsse

        return 0.0
