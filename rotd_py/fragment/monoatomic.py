from rotd_py.fragment.fragment import Fragment
from rotd_py.system import MolType
import numpy as np


class Monoatomic(Fragment):
    """This class is the rotation manipulation correspondent to Monoatomic molecule."""

    def set_molecule_type(self):

        self.frag_array['mol_type'] = MolType.MONOATOMIC

        self.frag_array['ang_size'] = 0 # originally 0

        self.frag_array['stat_sum'] = 1.0

    def lf2mf(self, lf_vector):


        if self.molecule_type != MolType.MONOATOMIC:
            raise ValueError("Wrong molecule type")
        print ("MONO lf_vector: ", lf_vector)

        return lf_vector

    def mf2lf(self, mf_vector):


        if self.molecule_type != MolType.MONOATOMIC:
            raise ValueError("Wrong molecule type")
        print ("MONOATOMIC mf_vector: ", mf_vector)

        return mf_vector

    def set_mfo(self):

        # There is no mfo for monoatomic molecule

        if self.molecule_type != MolType.MONOATOMIC:

            raise ValueError("Wrong molecule type")

    def set_labframe_positions(self):

        orig_mf_pos = self.get_molframe_positions()
        #print ("POSITIONS: ", orig_mf_pos) ## This is original molframe positions for Pt slab
        new_com = self.get_labframe_com()
        #print ("NEW COM", new_com) 

        for i in range(0, self.get_global_number_of_atoms()):
            for j in range(0, 3):
                self.frag_array['lab_frame_positions'][i][j] = self.get_labframe_com()[j]

    def get_labframe_imm(self, i, j):

        if self.molecule_type != MolType.MONOATOMIC:

            raise ValueError("Wrong molecule type")
        return 0.0

    def get_inertia_moments(self):

        return np.array([0.0]*3)
