from rotd_py.fragment.fragment import Fragment
from rotd_py.system import MolType
import numpy as np
import rotd_py.rotd_math as rotd_math
from ase.constraints import FixAtoms
from ase import Atoms
from ase.io import Trajectory



class Slab(Fragment):
    """This class is the rotation manipulation correspondent to Nonlinear molecule."""
    #print ("Welcome to Nonlinear molecule")
    def set_molecule_type(self):

        self.frag_array['mol_type'] = MolType.SLAB
        self.frag_array['ang_size'] = 4
#        self.frag_array['stat_sum'] = 2.0 * \
#            np.sqrt(2.0 * np.pi * np.prod(self.get_inertia_moments()))
        self.frag_array['stat_sum'] = 1
 #           np.sqrt(2.0 * np.pi * np.prod(self.get_inertia_moments()))
        #print ("setting molecule_type_in_non_linear 1st")

    def lf2mf(self, lf_vector):

        if self.molecule_type != MolType.SLAB:
            raise ValueError("Wrong molecule type")

        return np.dot(self.get_rotation_matrix(), lf_vector.reshape(3, 1)).reshape(3,)

    def mf2lf(self, mf_vector):

        if self.molecule_type != MolType.SLAB:
            raise ValueError("Wrong molecule type")
    
        #print ( 'slab mf2lf',self.get_rotation_matrix()) # Same as mfo

        return np.dot(mf_vector, self.get_rotation_matrix()).reshape(3,)

    def get_labframe_imm(self, i, j):

        if self.molecule_type != MolType.SLAB:
            raise ValueError("Wrong molecule type")
        return 0.0

    
    def set_rotation_matrix(self):
        #Convert the quaternion vector to a 3*3 rotation matrix.

        rot_vec = self.get_ang_pos().copy() #(4x4) matrix
        #print ('slab rot_vec', rot_vec)
        mfo = rotd_math.quaternion_matrix_transformation(rot_vec)
        #mfo = np.zeros((3, 3))

        #print ("frag.nonlin mfo: ", mfo)
        for i in range(0, 3):
            for j in range(0, 3):
                self.frag_array['mfo'][i][j] = mfo[i][j]
    
    # def set_labframe_positions(self):
    #     # molframe 가져와서 랜덤 로테이션 매트릭스만큼 닷 프로덕트 하고 나온 포지션에 센터오브매스 더해줌
    #     # molframe 의 경우, 내가 준 포지션에대해서 evecs(principal axis) 곱해서 정해짐
    #     #orig_mf_pos = self.get_molframe_positions()
    #     orig_mf_pos = self.get_original_positions() / rotd_math.Bohr # This sets slab to be in molframe position
    #     new_com = self.get_labframe_com()
    #     rel_pos = orig_mf_pos.copy()
    #     for i in range(0, self.get_number_of_atoms()): # for ase ==3.13.0
    #     #for i in range(0, self.get_global_number_of_atoms()): # for ase ==3.19.0 and over
    #         rel_pos[i] = orig_mf_pos[i]
        
    #         self.frag_array['lab_frame_positions'][i] = rel_pos[i] #+ new_com
        
    def set_labframe_positions(self):
        # molframe 가져와서 랜덤 로테이션 매트릭스만큼 닷 프로덕트 하고 나온 포지션에 센터오브매스 더해줌
        # molframe 의 경우, 내가 준 포지션에대해서 evecs(principal axis) 곱해서 정해짐
        mfo = self.get_rotation_matrix() # random
        #print ("THIS IS MFOOO: ", mfo) # Same as self.get_rotation_matrix()
        orig_mf_pos = self.get_molframe_positions()
        pure_orig_pos = self.get_original_positions()

        test_orig_mf_pos= np.array(orig_mf_pos)  
         
        new_test_orig_mf_pos = Atoms('Pt36', positions=test_orig_mf_pos)
        #print ('After Atoms',pure_orig_pos)
        slab_com = self.get_center_of_mass()

        new_com = self.get_labframe_com()
        #print ('labframe COM', new_com) # 랜덤 하지만 거의 0,0,0 에 인접한 값...
        for i in range(0, self.get_number_of_atoms()): # for ase ==3.13.0
        #for i in range(0, self.get_global_number_of_atoms()): # for ase ==3.19.0 and over
            rel_pos = np.dot(orig_mf_pos[i], mfo)
            for j in range(0, 3):
                self.frag_array['lab_frame_positions'][i][j] = rel_pos[j] + new_com[j]

    def slab_straightening_matrix_calculation(self):
        labframe_position = self.get_labframe_positions()

        # z축 평행 29, 
        # x축 평행 4
        # 원점 2
        # y축 평행 
        # 3 Normal vectors
        # x, y, z 각각 돌린후 평행할 친구들
        index_shift = 2

        to_be_parallel_x_after_rotation = labframe_position[4-index_shift] - labframe_position[2-index_shift]
        to_be_parallel_z_after_rotation = labframe_position[29-index_shift] - labframe_position[2-index_shift]

        to_be_parallel_y_after_rotation = np.cross(to_be_parallel_z_after_rotation, to_be_parallel_x_after_rotation)

        to_be_parallel_x_after_rotation /= np.linalg.norm(to_be_parallel_x_after_rotation)
        to_be_parallel_y_after_rotation /= np.linalg.norm(to_be_parallel_y_after_rotation)
        to_be_parallel_z_after_rotation /= np.linalg.norm(to_be_parallel_z_after_rotation)

        total_rotation_matrix = np.array([to_be_parallel_x_after_rotation, to_be_parallel_y_after_rotation, to_be_parallel_z_after_rotation])

        moving_vector = labframe_position[2-index_shift].copy()

        return moving_vector, total_rotation_matrix


    def get_inertia_moments(self):

        #print ('slab moments of inertia')

        return np.array([0.0]*3)
    

    # def set_slab_principal_axis_for_visualization(self): 
    #     rel_pos = self.get_relative_positions() / rotd_math.Bohr
    #     masses = self.get_masses() * rotd_math.mp
    #     I11 = I22 = I33 = I12 = I13 = I23 = 0.0
    #     for i in range(len(rel_pos)):
    #         x, y, z = rel_pos[i]
    #         m = masses[i]

    #         I11 += -m * (x ** 2)
    #         I22 += -m * (y ** 2)
    #         I33 += -m * (z ** 2)
    #         I12 += -m * x * y
    #         I13 += -m * x * z
    #         I23 += -m * y * z

    #     I = np.array([[I11, I12, I13, ],
    #                   [I12, I22, I23],
    #                   [I13, I23, I33]])

    #     trace = I.trace()
    #     I = I - np.array([[trace, 0, 0],
    #                       [0, trace, 0],
    #                       [0, 0, trace]])
    #     new_evals, new_evecs = np.linalg.eigh(I)
        
    #     #self.frag_array['inertia_moments'] = evals # used for get_inertia_moments, which is used for rc_vec calculation # Principal moments of inertia
    #     self.frag_array['slab_evecs'] = new_evecs # Used to set labframe_pivot for nonlinear # Principal axes for original, Identity matrix

    #     #print ('slab_evecs', self.frag_array['slab_evecs'])
