from rotd_py.fragment.fragment import Fragment
from rotd_py.system import MolType
import numpy as np


class Slab(Fragment):
    ###This class is the rotation manipulation correspondent to surface Slab.

    def set_molecule_type(self):

        self.frag_array['mol_type'] = MolType.SLAB
        self.frag_array['ang_size'] = 1
        self.frag_array['stat_sum'] = 1.0

    def lf2mf(self, lf_vector):

        if self.molecule_type != MolType.SLAB:
            raise ValueError("Wrong molecule type")

        A = np.dot(self.get_rotation_matrix(), lf_vector.reshape(3, 1)).reshape(3,) 
        #print ("SLAB lf2mf test: ", A)
        #print ("SLAB lf_vector: ", lf_vector)
        #return np.dot(self.get_rotation_matrix(), lf_vector.reshape(3, 1)).reshape(3,) 
        return lf_vector

    def mf2lf(self, mf_vector):

        if self.molecule_type != MolType.SLAB:
            raise ValueError("Wrong molecule type")

        B = np.dot(mf_vector, self.get_rotation_matrix()).reshape(3,)
        #print ("SLAB mf2lf test: ", B)  # 이건 프린트 되는데 위의 lf2mf 는 프린트가 안되네??
        #print ("SLAB mf_vector: ", mf_vector)  #이것도 마찬가지

        #return np.dot(mf_vector, self.get_rotation_matrix()).reshape(3,)
        return mf_vector

########################################################################################################
# 이게 지금 ang_size 가 3 개여서 문제.. 1개 아니면 0개로 바꿔야함.
########################################################################################################
    """
    def set_rotation_matrix(self):
        #Convert 1*3 rotation vector to 3*3 rotation matrix.
        if self.molecule_type != MolType.SLAB:
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
    """    

    def set_rotation_matrix(self):
        #Convert 1*3 rotation vector to 3*3 rotation matrix.
        if self.molecule_type != MolType.SLAB:
            raise ValueError("Wrong molecule type")

    def set_mfo(self):

        # There is no mfo for monoatomic molecule
        if self.molecule_type != MolType.SLAB:
            raise ValueError("Wrong molecule type")
        return


    """
    def set_labframe_positions(self):
        mfo = self.get_rotation_matrix()
        orig_mf_pos = self.get_molframe_positions()
        new_com = self.get_labframe_com()
        for i in range(0, self.get_number_of_atoms()):
            rel_pos = np.dot(orig_mf_pos[i], mfo)
            for j in range(0, 3):
                self.frag_array['lab_frame_positions'][i][j] = rel_pos[j] + new_com[j]
    """


######## TEST for SLAB
    def set_labframe_positions(self):
        #mfo = self.get_rotation_matrix()
        orig_mf_pos = self.get_molframe_positions()
        #print ("SLAB orig_mf_pos: ", orig_mf_pos) ##THIS IS FINE
        new_com = self.get_labframe_com()
        #print ("SLAB new_com: ", new_com) ##THIS IS ALSO FINE

        rel_pos = orig_mf_pos.copy()
        #for i in range(0, self.get_number_of_atoms()):
        for i in range(0, self.get_global_number_of_atoms()):
            rel_pos[i] = orig_mf_pos[i]
            #print ("SLAB rel_pos: ", rel_pos)
            self.frag_array['lab_frame_positions'][i] = rel_pos[i] + new_com

    def get_labframe_imm(self, i, j):

        if self.molecule_type != MolType.SLAB:
            raise ValueError("Wrong molecule type")
        return 0.0

    def get_inertia_moments(self):

        return np.array([0.0]*3)

