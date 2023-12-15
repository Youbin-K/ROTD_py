from rotd_py.fragment.fragment import Fragment
from rotd_py.system import MolType
import numpy as np


class Monoatomic(Fragment):
    """This class is the rotation manipulation correspondent to Monoatomic molecule."""

    def set_molecule_type(self):

        self.frag_array['mol_type'] = MolType.MONOATOMIC
        self.frag_array['ang_size'] = 0
        self.frag_array['stat_sum'] = 1.0

    def lf2mf(self, lf_vector):

        if self.molecule_type != MolType.MONOATMIC:
            raise ValueError("Wrong molecule type")
        return lf_vector

    def mf2lf(self, mf_vector):

        if self.molecule_type != MolType.MONOATMIC:
            raise ValueError("Wrong molecule type")
        return mf_vector

    def set_mfo(self):

        # There is no mfo for monoatomic molecule
        if self.molecule_type != MolType.MONOATMIC:
            raise ValueError("Wrong molecule type")

    def set_labframe_positions(self):

        for i in range(0, 3):
            self.frag_array['lab_frame_positions'][i] = self.get_labframe_com()[i]

    def get_labframe_imm(self, i, j):

        if self.molecule_type != MolType.MONOATMIC:
            raise ValueError("Wrong molecule type")
        return 0.0

    def get_inertia_moments(self):

        return np.array([0.0]*3)
