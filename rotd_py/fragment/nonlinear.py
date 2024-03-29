from rotd_py.fragment.fragment import Fragment
from rotd_py.system import MolType
import numpy as np
import rotd_py.rotd_math as rotd_math
from ase.constraints import FixAtoms

class Nonlinear(Fragment):
    """This class is the rotation manipulation correspondent to Nonlinear molecule."""
    print ("Welcome to Nonlinear molecule")
    def set_molecule_type(self):

        self.frag_array['mol_type'] = MolType.NONLINEAR
        self.frag_array['ang_size'] = 4
#        self.frag_array['stat_sum'] = 2.0 * \
#            np.sqrt(2.0 * np.pi * np.prod(self.get_inertia_moments()))
        self.frag_array['stat_sum'] = 1
 #           np.sqrt(2.0 * np.pi * np.prod(self.get_inertia_moments()))
        print ("setting molecule_type_in_non_linear 1st")

    def lf2mf(self, lf_vector):

        if self.molecule_type != MolType.NONLINEAR:
            raise ValueError("Wrong molecule type")
        #print ("NON lf2mf: ",np.dot(self.get_rotation_matrix(), lf_vector.reshape(3, 1)).reshape(3,))
        #print ("self.get_rotation_matrix: ", self.get_rotation_matrix())
        #print ("lf_vector", lf_vector)
        #print ("reshape(3,1) ", lf_vector.reshape(3,1))
        #print ("(3,1).reshape(3,)", lf_vector.reshape(3,1).reshape(3,))
        #print ("nonlinear lf2mf")
        return np.dot(self.get_rotation_matrix(), lf_vector.reshape(3, 1)).reshape(3,)

    def mf2lf(self, mf_vector):

        if self.molecule_type != MolType.NONLINEAR:
            raise ValueError("Wrong molecule type")
        #print ("NON mf2lf: ", np.dot(mf_vector, self.get_rotation_matrix()).reshape(3,))
        #print ("lf2mf nonlinear")
        #print ("88888")
        # At first, after set_rotation_matrix, it is called out as number of the set_rotation_matrix. However, right after the next step (mf2lf), it is called out multiple times.
        return np.dot(mf_vector, self.get_rotation_matrix()).reshape(3,)

    def get_labframe_imm(self, i, j):

        rotation_matrix = self.get_rotation_matrix()
        inertia_mom = self.get_inertia_moments()
        res = 0
        for k in range(0, 3):
            res += inertia_mom[k] * rotation_matrix[k][i] * rotation_matrix[k][j]
        return res

    
    def set_rotation_matrix(self):
        #Convert the quaternion vector to a 3*3 rotation matrix.

        #print ("nonlinear setting rotation matrix")
        #print ("77777")
        # Called out as much as number of get_microcanonical factor + get_ej_factor
        rot_vec = self.get_ang_pos().copy()
        #print ("frag.nonlin rot_vec: ", rot_vec)
        mfo = rotd_math.quaternion_matrix_transformation(rot_vec)
        #print ("frag.nonlin mfo: ", mfo)
        for i in range(0, 3):
            for j in range(0, 3):
                self.frag_array['mfo'][i][j] = mfo[i][j]
    
    def set_labframe_positions(self):

        mfo = self.get_rotation_matrix()
        #print ("THIS IS MFOOO: ", mfo)
        orig_mf_pos = self.get_molframe_positions()
        new_com = self.get_labframe_com()
        #for i in range(0, self.get_number_of_atoms()):
        for i in range(0, self.get_global_number_of_atoms()):
            rel_pos = np.dot(orig_mf_pos[i], mfo)
            for j in range(0, 3):
                self.frag_array['lab_frame_positions'][i][j] = rel_pos[j] + new_com[j]

    def get_inertia_moments(self):

        return self.frag_array['inertia_moments'].copy()


"""
class Slab(Nonlinear):
    #This class is the rotation manipulation correspondent to Slab surface.

    def set_labframe_positions(self):
        #Do not move positions of Slab
        mfo = self.get_rotation_matrix()
        #print ("THIS IS MFOOO: ", mfo)
        orig_mf_pos = self.get_molframe_positions()
        #new_com = self.get_labframe_com()
        for i in range(0, self.get_number_of_atoms()):
            rel_pos = np.dot(orig_mf_pos[i], mfo)
            for j in range(0, 3):
                self.frag_array['lab_frame_positions'][i][j] = rel_pos[j]
"""
