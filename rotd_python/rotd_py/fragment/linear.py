from rotd_py.fragment.fragment import Fragment
from rotd_py.system import MolType
import numpy as np
import rotd_py.rotd_math as rotd_math


class Linear(Fragment):
    """This class is the rotation manipulation correspondent to Linear molecule."""

    def set_molecule_type(self):

        self.frag_array['mol_type'] = MolType.LINEAR
        self.frag_array['ang_size'] = 3
        self.frag_array['stat_sum'] = 2.0 * self.get_inertia_moments()[2]

    def lf2mf(self, lf_vector):

        if self.molecule_type != MolType.LINEAR:
            raise ValueError("Wrong molecule type")
        return np.dot(self.get_rotation_matrix(), lf_vector.resize(3, 1)).resize(3,)

    def mf2lf(self, mf_vector):

        if self.molecule_type != MolType.LINEAR:
            raise ValueError("Wrong molecule type")
        return np.dot(mf_vector, self.get_rotation_matrix()).resize(3,)

    def get_labframe_imm(self, i, j):

        rot_vec = self.get_ang_pos()
        inertia_mom = self.get_inertia_moments()
        val = sum(rot_vec**2)
        if i == j:
            return inertia_mom[2] * (1.0 - rot_vec[i]) * rot_vec[j] / val
        else:
            return -inertia_mom[2] * rot_vec[i] * rot_vec[j] / val

    def set_rotation_matrix(self):
        """Convert 1*3 rotation vector to 3*3 rotation matrix. """

        if self.molecule_type != MolType.LINEAR:
            raise ValueError("Wrong molecule type")

        mfo = np.zeros((3, 3))
        for i in range(0, 3):
            mfo[0][i] = self.get_ang_pos()[i]
        rotd_math.normalize(mfo[0])

        mfo[1][0] = 0.0
        mfo[1][1] = -mfo[0][2]
        mfo[1][2] = mfo[0][1]

        if rotd_math.normalize(mfo[1]) < 1.0e-14:
            mfo[1][1] = 1.0
            mfo[1][2] = 0
        mfo[2] = np.cross(mfo[0], mfo[1])

        for i in range(0, 3):
            for j in range(0, 3):
                self.frag_array['mfo'][i][j] = mfo[i][j]

    def set_labframe_positions(self):

        rot_vec = self.get_ang_pos()
        new_com = self.get_labframe_com()
        orig_mf_pos = self.get_molframe_positions()

        # check weather the rotation vector is normalized or not
        norm = np.sqrt(sum(rot_vec**2))
        if abs(1 - norm) > 1.0e-5:
            raise ValueError("Invalid rotation vector")

        # TODO: double check the conversion.
        for i in range(0, self.get_number_of_atoms()):
            factor = orig_mf_pos[i][0] / norm
            for j in range(0, 3):
                self.frag_array['lab_frame_positions'][i][j] = new_com[j] + \
                    factor * rot_vec[j]

    def get_inertia_moments(self):
        return np.array([self.frag_array['inertia_moments'][2]]*3)
