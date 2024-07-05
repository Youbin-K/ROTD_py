from rotd_py.fragment.fragment import Fragment
from rotd_py.system import MolType
import numpy as np
import rotd_py.rotd_math as rotd_math


class Slab(Fragment):
    ###This class is the rotation manipulation correspondent to surface Slab.
    def __init__(self, orig_mfo=None, *args, **kwargs):
        self.tossed_mfo = orig_mfo
        #print ('tossed_mfo ', self.tossed_mfo)
        super().__init__(*args, **kwargs) # super가 parent class(여기서는 fragment 불러와서) 그걸 돌림.

    def set_molframe_positions(self):
        """Method used to set up the matrix convert the input Cartesian
        coordinates to molecular frame coordinates and the molecule frame
        positions.        
        """
        rel_pos = self.get_relative_positions() / rotd_math.Bohr
        #print ('molframe position for slab', rel_pos)
        self.frag_array['inertia_moments'] = np.zeros(3)
        self.frag_array['orig_mfo'] = self.tossed_mfo 
        self.frag_array['mol_frame_positions'] = np.dot(rel_pos, self.tossed_mfo) # matrix X matrix 여서 지금은 그냥 곱셈이나 마찬가지


    def set_molecule_type(self):

        self.frag_array['mol_type'] = MolType.SLAB
        self.frag_array['ang_size'] = 0 # or 1?
        self.frag_array['stat_sum'] = 1.0

    def lf2mf(self, lf_vector):

        if self.molecule_type != MolType.SLAB:
            raise ValueError("Wrong molecule type")

        #A = np.dot(self.get_rotation_matrix(), lf_vector.reshape(3, 1)).reshape(3,) 

        return np.dot(self.tossed_mfo, lf_vector.reshape(3, 1)).reshape(3,) 
        #return lf_vector

    def mf2lf(self, mf_vector):

        if self.molecule_type != MolType.SLAB:
            raise ValueError("Wrong molecule type")
        
        #print ('mf2lf tossed_mfo: ', self.tossed_mfo)

        return np.dot(mf_vector, self.tossed_mfo).reshape(3,)

        
    
    def unit_cell_mf(self):
        """Method used to set up the matrix convert the input Cartesian
        coordinates to molecular frame coordinates and the molecule frame
        positions.        
        """

        unit_cell_334_pt = np.array([[8.3, 0.000, 0.000],
                                    [4.15, 7.188, 0.000],
                                    [0.000, 0.000, 26.777]])
        
        test_pos_com = self.get_center_of_mass()
        #print ('unit_cell_com', test_pos_com)
        unit_cell_334_pt -= test_pos_com 
        unit_cell_334_pt /= rotd_math.Bohr
        #print ('unit_cell_orig_pos', unit_cell_334_pt)

        # self.frag_array['orig_mfo'] = self.tossed_mfo 
        # self.frag_array['unit_cell_mf_position'] = np.dot(unit_cell_334_pt, self.tossed_mfo)
        
        unit_cell_334_pt = np.dot(unit_cell_334_pt, self.tossed_mfo) # matrix X matrix 여서 지금은 그냥 곱셈이나 마찬가지
        unit_cell_334_pt *= rotd_math.Bohr
        unit_cell_334_pt += test_pos_com
        unit_cell_334_pt /= rotd_math.Bohr
        self.frag_array['unit_cell_mf_position'] = unit_cell_334_pt


    def unit_cell_mf2lf(self, mf_vector):

        if self.molecule_type != MolType.SLAB:
            raise ValueError("Wrong molecule type")
        
        return np.dot(mf_vector, self.tossed_mfo)

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
        orig_mf_pos = self.get_molframe_positions() # 고정이어야하고
        #orig_mf_pos = self.get_relative_positions() / rotd_math.Bohr

        #orig_mf_pos /= [1.5,1.5,1.5]
        #orig_mf_pos_test = orig_mf_pos.copy() - [1.5,1.5,1.5]
        orig_mf_pos_test = orig_mf_pos.copy()
        #np.transpose(orig_mf_pos_test)

        new_com = self.get_labframe_com()
        #print ("SLAB new_com: ", new_com) ##THIS IS ALSO FINE
        
        #rel_pos = orig_mf_pos.copy()
        rel_pos = orig_mf_pos_test.copy()
        for i in range(0, self.get_number_of_atoms()): # for ase ==3.13.0
        #for i in range(0, self.get_global_number_of_atoms()): # for ase ==3.19.0 and over
            rel_pos[i] = orig_mf_pos[i]
            
            self.frag_array['lab_frame_positions'][i] = rel_pos[i] + new_com
            #self.frag_array['lab_frame_positions'][i] = rel_pos[i] 

        

    def get_labframe_imm(self, i, j):

        if self.molecule_type != MolType.SLAB:
            raise ValueError("Wrong molecule type")
        return 0.0

    def get_inertia_moments(self):

        return np.array([0.0]*3)

