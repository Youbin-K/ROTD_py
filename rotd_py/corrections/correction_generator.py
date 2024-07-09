from rotd_py.corrections.correction import Correction
from rotd_py.sample.sample import Sample
from rotd_py.corrections.onedimensional import OneDimensional
from rotd_py.corrections.bsse import BSSE


class CorrectionGenerator:
    def __init__(self,
                 name: str,
                 parameters: dict,
                 sample: Sample) -> None:

        self.name = name
        self.sample = sample
        self.init_correction()
        self.param = parameters
        self.corr: Correction

    def init_correction(self) -> None:
        if 'type' not in self.param:
            raise AttributeError("Correction dictionary needs a 'type' entry.")
        type = self.param["type"].casefold()

        if type == '1d':
            self.corr = OneDimensional(self.name,
                                       self.sample)
        elif self.type == "counterpoise":
            self.corr = BSSE(self.name,
                             self.sample)

    def valid_parameters(self) -> bool:
        for param in self.corr.necessary_keys:
            if param not in self.param:
                return False
        return True

    def generate(self) -> Correction:
        if self.valid_parameters():
            self.corr.generate(self.param)
        else:
            raise AttributeError("All necessary parameters where\
                                  not given for correction\
                                 {}.".format(self.name))
