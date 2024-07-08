from rotd_py.corrections.correction import Correction
from rotd_py.sample.sample import Sample


class CorrectionGenerator:
    def __init__(self,
                 name:str,
                 parameters:dict,
                 sample:Sample) -> None:
        
        self.name = name
        self.sample = sample
        self.init_correction()
        self.param = parameters
        self.default_energy = None
        self.corr: Correction

    def init_correction(self) -> None:
        if 'type' not in self.param:
            raise AttributeError("Correction dictionary needs a 'type' entry.")
        type = self.param["type"].casefold()

        if type == '1d':
            self.corr = 

    def valid_parameters(self) -> bool:
        return True

    def generate(self) -> Correction:
        pass
    