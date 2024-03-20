from rotd_py.sample.sample import Sample, prerprocess
from ase.atoms import Atoms
from rotd_py.system import SampTag
import numpy as np
import rotd_py.rotd_math as rotd_math


class DummySample(Sample):
    """Inherit class of Sample. Used for testing the calculation between the
    original varecof and current code.
    Could be useful for development, but not for actual usage of the code.

    Parameters
    ----------
    fragments : List of fragments
    dividing_surface : type
    Attributes
    ----------
    geometries : type
        Results of preprocess of results from original rotd.
    iter : type
        Iterate the geometry in the stored sample.

    """

    def __init__(self, fragments=None, dividing_surface=None):
        # I want to create a list of geometry that I can used as a iterator
        # for the geometry class, it will include the weight, the geometry(atoms)
        # the energy
        self.geometries = preprocess('ch3ch3_test.txt')
        self.iter = iter(self.geometries)
        super(DummySample, self).__init__(fragments=fragments,
                                          dividing_surface=dividing_surface)

    def generate_configuration(self):
        geom = next(self.iter)
        self.configuration = geom.get_atoms()
        tag = SampTag.SAMP_SUCCESS
        self.weight = geom.get_weight()
        self.energy = np.array([geom.get_energy() * rotd_math.Kcal])
        self.tot_im = geom.get_tot_im()
        return tag


# first only add one configuration here and return the
# then maybe a iterator could be helpful.
