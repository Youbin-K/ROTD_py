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
#        return np.dot(self.get_rotation_matrix(), lf_vector.resize(3, 1)).resize(3,)
        return np.dot(self.get_rotation_matrix(), lf_vector.reshape(3, 1)).reshape(3,)        

    def mf2lf(self, mf_vector):

        if self.molecule_type != MolType.LINEAR:
            raise ValueError("Wrong molecule type")
        # return np.dot(mf_vector, self.get_rotation_matrix()).resize(3,)
        return np.dot(mf_vector, self.get_rotation_matrix()).reshape(3,)

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
        for i in range(0, self.get_global_number_of_atoms()):
            factor = orig_mf_pos[i][0] / norm
            for j in range(0, 3):
                self.frag_array['lab_frame_positions'][i][j] = new_com[j] + \
                    factor * rot_vec[j]

    def set_labframe_positions_for_surface_rotd(self):
        # molframe 가져와서 랜덤 로테이션 매트릭스만큼 닷 프로덕트 하고 나온 포지션에 센터오브매스 더해줌
        # molframe 의 경우, 내가 준 포지션에대해서 evecs(principal axis) 곱해서 정해짐
        mfo = self.get_rotation_matrix() # random
        # print ("THIS IS MFOOO: ", mfo)
        orig_mf_pos = self.get_molframe_positions()
        new_com = self.get_labframe_com()
        # before_setting_lf_position_Pt = self.get_positions()
        for i in range(0, self.get_number_of_atoms()): # for ase ==3.13.0
        #for i in range(0, self.get_global_number_of_atoms()): # for ase ==3.19.0 and over
            rel_pos = np.dot(orig_mf_pos[i], mfo)
            # print ('I', i)
            for j in range(0, 3):
                # print ('J',j)
                self.frag_array['lab_frame_positions'][i][j] = rel_pos[j] + new_com[j]

    def get_inertia_moments(self):
        return np.array([self.frag_array['inertia_moments'][2]]*3)
    
    def get_inertia_moments_for_surface(self):
        # print ('full inertia moments', self.frag_array['inertia_moments'])
        # print ('round work?', np.round(self.frag_array['inertia_moments']))
        # print ('what is returned', np.array([self.frag_array['inertia_moments'][2]]*3))
        
        # return np.array([self.frag_array['inertia_moments'].copy()])
        return np.array([self.frag_array['inertia_moments'][2]]*3)
