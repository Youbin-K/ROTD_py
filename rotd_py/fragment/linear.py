from rotd_py.fragment.fragment import Fragment
from rotd_py.system import MolType
import numpy as np
import rotd_py.rotd_math as rotd_math


class Linear(Fragment):
    """This class is the rotation manipulation correspondent to Linear molecule."""

    def set_molecule_type(self):

        self.frag_array['mol_type'] = MolType.LINEAR
        self.frag_array['ang_size'] = 3
        # Turned off since is not used..
        #print ('this is original linear.py')
        #self.frag_array['stat_sum'] = 2.0 * self.get_inertia_moments()[2]

    def lf2mf(self, lf_vector):

        if self.molecule_type != MolType.LINEAR:
            raise ValueError("Wrong molecule type")
        # Print below does not work Since the linear does not go through this method

        #print ("Linear lf2mf: ", np.dot(self.get_rotation_matrix(), lf_vector.resize(3, 1)).resize(3,))
        #print ("LINEAR lf2mf: ", np.dot(self.get_rotation_matrix(), lf_vector.resize(3, 1)).resize(3,))
        #print ("LINEAR lf_vector: ", lf_vector)
        
        # Below is original
        # return np.dot(self.get_rotation_matrix(), lf_vector.resize(3, 1)).resize(3,)
    
        #한번에 바꿔야함 #1
        return np.dot(self.get_rotation_matrix(), lf_vector.reshape(3, 1)).reshape(3,)        
    

    

        



    def mf2lf(self, mf_vector):

        if self.molecule_type != MolType.LINEAR:
            raise ValueError("Wrong molecule type")
        # This also does not print since it does not go through this pass.


        #print ("Linear  : ",np.dot(mf_vector, self.get_rotation_matrix()).resize(3,))
        
        # Below is original
        # return np.dot(mf_vector, self.get_rotation_matrix()).resize(3,)        
    
        # 한번에 바꿔야함 #1
        # print ('rotation matrix ',self.get_rotation_matrix()) # 무수히 많은 종류의 값. get_rotation_matrix의 베이스가 get_ang_pos이기 때문에
        return np.dot(mf_vector, self.get_rotation_matrix()).reshape(3,)



    """
    def get_labframe_imm(self, i, j):

        #rot_vec = self.get_ang_pos()
        rot_vec = self.get_rotation_matrix()
        print ("Linear rot_vec: ", rot_vec)
        inertia_mom = self.get_inertia_moments()
        val = sum(rot_vec**2)
        if i == j:
            return inertia_mom[2] * (1.0 - rot_vec[i]) * rot_vec[j] / val
        else:
            return -inertia_mom[2] * rot_vec[i] * rot_vec[j] / val
    """

    def set_rotation_matrix(self):
        """Convert 1*3 rotation vector to 3*3 rotation matrix. """

        if self.molecule_type != MolType.LINEAR:
            raise ValueError("Wrong molecule type")

        mfo = np.zeros((3, 3))
        for i in range(0, 3):
            mfo[0][i] = self.get_ang_pos()[i]
        
        #print ("LINEAR mfo[0] after ang_pos: ", mfo[0])
        rotd_math.normalize(mfo[0])
        
        #print ("LINEAR mfo after normal: ", mfo)        

        mfo[1][0] = 0.0
        mfo[1][1] = -mfo[0][2]
        mfo[1][2] = mfo[0][1]
        #print ("LINEAR mfo after garbage1: ", mfo)
     
        if rotd_math.normalize(mfo[1]) < 1.0e-14:
            mfo[1][1] = 1.0
            mfo[1][2] = 0
        mfo[2] = np.cross(mfo[0], mfo[1])
        #print ("LINEAR mfo after garbage2: ", mfo)

        for i in range(0, 3):
            for j in range(0, 3):
                self.frag_array['mfo'][i][j] = mfo[i][j]
        #print ("END: ",mfo)

    def set_labframe_positions(self):

        rot_vec = self.get_ang_pos() # random value
        #print ('rot_vec in lf_pos, ', rot_vec) # Same as frag.get_ang_pos() in labframe_pivot_points
        new_com = self.get_labframe_com()
        #print ('linear lf_com', new_com)
        orig_mf_pos = self.get_molframe_positions()
        # check weather the rotation vector is normalized or not
        norm = np.sqrt(sum(rot_vec**2)) 
        #print ('original set labframe called out which should not be')
        if abs(1 - norm) > 1.0e-5: # Norm 은 항상 1
            raise ValueError("Invalid rotation vector")

        # TODO: double check the conversion.
        for i in range(0, self.get_number_of_atoms()): # for ase==3.13.0 
        #for i in range(0, self.get_global_number_of_atoms()): # for ase==3.19.0 and over
            # Original
            factor = orig_mf_pos[i][0] / norm # x 좌표 가져옴
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

        # print ('Nonlin method value: ', self.frag_array['testing_lab_frame_positions'])
        # print ('Linear method value: ', self.frag_array['lab_frame_positions'])
             
    def get_inertia_moments(self):
        # print ('full inertia moments', self.frag_array['inertia_moments'])
        # print ('what is returned', np.array([self.frag_array['inertia_moments'][2]]*3))
        
        return np.array([self.frag_array['inertia_moments'][2]]*3)
    
    def get_inertia_moments_for_surface(self):
        # print ('full inertia moments', self.frag_array['inertia_moments'])
        # print ('what is returned', np.array([self.frag_array['inertia_moments'][2]]*3))
        
        return self.frag_array['inertia_moments'].copy()
