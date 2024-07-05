from rotd_py.fragment.fragment import Fragment
from rotd_py.system import MolType
import numpy as np
import rotd_py.rotd_math as rotd_math


class Nonlinear(Fragment):
    """This class is the rotation manipulation correspondent to Nonlinear molecule."""

    def set_molecule_type(self):

        self.frag_array['mol_type'] = MolType.NONLINEAR
        self.frag_array['ang_size'] = 4
        self.frag_array['stat_sum'] = 2.0 * \
            np.sqrt(2.0 * np.pi * np.prod(self.get_inertia_moments()))

    def lf2mf(self, lf_vector):

        if self.molecule_type != MolType.NONLINEAR:
            raise ValueError("Wrong molecule type")
        return np.dot(self.get_rotation_matrix(), lf_vector.reshape(3, 1)).reshape(3,)

    def mf2lf(self, mf_vector):

        if self.molecule_type != MolType.NONLINEAR:
            raise ValueError("Wrong molecule type")
        return np.dot(mf_vector, self.get_rotation_matrix()).reshape(3,)

    def get_labframe_imm(self, i, j):

        rotation_matrix = self.get_rotation_matrix()
        inertia_mom = self.get_inertia_moments()
        res = 0
        for k in range(0, 3):
            res += inertia_mom[k] * rotation_matrix[k][i] * rotation_matrix[k][j]
        return res

    def set_rotation_matrix(self):
        """Convert the quaternion vector to a 3*3 rotation matrix.

        """
        rot_vec = self.get_ang_pos().copy()
        mfo = rotd_math.quaternion_matrix_transformation(rot_vec)
        for i in range(0, 3):
            for j in range(0, 3):
                self.frag_array['mfo'][i][j] = mfo[i][j]

    def set_labframe_positions(self):

        mfo = self.get_rotation_matrix()
        #print ("THIS IS MFOOO: ", mfo)
        orig_mf_pos = self.get_molframe_positions()
        new_com = self.get_labframe_com()
        for i in range(0, self.get_number_of_atoms()):
        #for i in range(0, self.get_global_number_of_atoms()):
            rel_pos = np.dot(orig_mf_pos[i], mfo)
            for j in range(0, 3):
                self.frag_array['lab_frame_positions'][i][j] = rel_pos[j] + new_com[j]

    def get_inertia_moments(self):

        return self.frag_array['inertia_moments'].copy()
