import numpy as np
from rotd_py.sample.sample import Sample, preprocess
import rotd_py.rotd_math as rotd_math
from rotd_py.system import MolType, SampTag
from ase import Atoms, atoms
from ase.constraints import FixAtoms
from ase.constraints import FixBondLength
from ase.io.trajectory import Trajectory
import math


class MultiSample(Sample):

    """
    This subclass is used for generating configuration based on dividing surface
    For MultiSample, there is no sample constraint area. In another word,
    both center of mass vector and the fragments themselves can rotate in any
    direction.
    For current usage, this class only consider 2 fragments system.


    """

    def generate_configuration(self):
        """Generate the random rotational configuration.

        Returns
        -------
        SampTag
            Return SampTag to indicate whether the sample is valid or not.

        """
        #print ("Let's start with generating the config")
        # define essential parameters:
        total_mass = 0
        for frag in self.fragments:
            total_mass += frag.get_total_mass()
        mfactor = np.zeros(2)
        mfactor[0] = self.fragments[1].get_total_mass()/total_mass # 거의 1 
        mfactor[1] = -self.fragments[0].get_total_mass()/total_mass # 거의 0 
        emass = self.fragments[0].get_total_mass() * mfactor[0] # This is reduced mass

        # random orient a vector between the center of mass of two fragments
        com_vec = rotd_math.random_orient(3)  # 1st step
        #print ("com_vec", com_vec) # 3 array

        # random rotate each fragment based on random generated solid angle.
 
        for frag in self.fragments:
            orient = rotd_math.random_orient(frag.get_ang_size()) 
            frag.set_ang_pos(orient)
            frag.set_rotation_matrix()

        # get the pivot points coordinates for current face in the laboratory frame
        lfactor = 1.0
        lf_pivot = np.zeros((2, 3))
        molframe_pivot = np.zeros((2, 3))
        empty_com = np.zeros((1, 3))
        #print ("lf_pivot_1, np.zeros(2,3)", lf_pivot)
        curr_face = self.div_surface.get_curr_face()
        if all(frag.molecule_type == MolType.LINEAR for frag in self.fragments):
            # for linear molecule, the pivot point could only sit on x-axis?
            for i in range(0, len(self.fragments)):
                lf_pivot[i][:] = (self.fragments[i].get_ang_pos() *
                                  self.div_surface.get_pivot_point(i, curr_face)[0])
        else:
            for i in range(0, len(self.fragments)):
                sample_info, lfactor = self.labframe_pivot_point(
                    i, com_vec, lf_pivot[i], molframe_pivot[i], lfactor) 
                # com_vec = rotd_math.random_orient(3)

                if sample_info == SampTag.SAMP_FACE_OUT:
                    return SampTag.SAMP_FACE_OUT
                
        #print ("0 Pivot: ", lf_pivot[0]) # 0,0,0 nonlin 일때, 0,0,0
        #print ("1 Pivot: ", lf_pivot[1]) # 0,0,0 nonlin 일때, 0,0,0 # 2,2,2 일때 계속 바뀌네..? 심지어 처음부터 2,2,2는 불리지도 않고 이상한값 나옴. 
                                            # 왜냐면 labframe_pivot에서 디파인 될때부터 랜덤 밸류가 곱해지기에..
        #now set the reaction coordinate vector
        # lf_com 의 경우 check_sample 에서와 아래 set_labframe_COM 일때 쓰임
        lf_com = com_vec * self.div_surface.get_dist(curr_face) - lf_pivot[0] + lf_pivot[1] # lf_pivot 에 get_rotation_matrix 들어가있음. 
        #print ('com_vec, random ', com_vec) # 당연하게 랜덤.
        #print ('yangsik for lf_com',lf_com) # 당연하게도 랜덤이 곱해지니 랜덤
        #print ('distance', self.div_surface.get_dist(curr_face)) # 내가 initial 로 준 값 # 내가 준 거리 값 잘 나옴.
        # print ('lf_pivot[0] ', lf_pivot[0]) # 바뀜
        # print ('lf_pivot[1]', lf_pivot[1]) # 바뀜
        
        new_positions = []
        for_surface_lf_visual = []

        if any(frag.molecule_type == MolType.SLAB for frag in self.fragments):

            for i in range(0, 2):
            # update COM positions

                self.fragments[i].set_labframe_com(lf_com * mfactor[i]) # set_labframe_COM 의 경우 아래에 set_labframe_position 을 가져올때만 쓰임 
                #self.fragments[i].set_labframe_com(lf_com) # set_labframe_COM 의 경우 아래에 set_labframe_position 을 가져올때만 쓰임 
                #print ('lf_com * mfactor, ', self.fragments[1].set_labframe_com(lf_com * mfactor[1]))
                # set up the final laboratory frame of two fragment
                # self.fragments[i].set_labframe_positions()

                self.fragments[i].set_labframe_positions_for_surface_rotd()
                # self.fragments[i].set_labframe_position_only_for_visualization()


                # print ('testing_co_com')
         
                for pos in self.fragments[i].get_labframe_positions():
                    new_positions.append(pos)
                    for_surface_lf_visual.append(pos)

            # Viewing real position of CO pivot
            testing_co_com = self.fragments[0].get_labframe_com()


            testing_Pt_com = self.fragments[1].get_labframe_com()

            view_CO_pivot = lf_pivot[0] + (lf_com * mfactor[0])
            view_Pt_pivot = lf_pivot[1] + (lf_com * mfactor[1])
            for_surface_lf_visual.append(view_CO_pivot) # Boron
            for_surface_lf_visual.append(view_Pt_pivot) # Nitrogen
            for_surface_lf_visual.append(testing_co_com)      # Florine
            for_surface_lf_visual.append(testing_Pt_com)     # Helium

            for_visual = []
            for i in range(0, 2): # This allows the labframe to be shown like the ASE initial. 
                #!!!!!!!Warning WARNING 두가지 경우에 바뀌어야함 1. slab의 모양이 바뀌거나, 더이상 Pt36 이 아닐때. 2. Adsorbate이 CO 같은게 아니고 복잡해질때
                for pos in self.fragments[i].get_labframe_positions():
                    moving_vector, total_rotation_matrix = self.fragments[1].slab_straightening_matrix_calculation()
                    pos_visual = pos - moving_vector
                    pos_visual = np.dot(total_rotation_matrix, pos_visual)
                    for_visual.append(pos_visual)

            lf_pivot_zero_moving = view_CO_pivot - moving_vector
            lf_pivot_zero_moved = np.dot(total_rotation_matrix, lf_pivot_zero_moving)

            lf_pivot_one_moving = view_Pt_pivot - moving_vector
            lf_pivot_one_moved = np.dot(total_rotation_matrix, lf_pivot_one_moving)

            lf_com_moving = lf_com -moving_vector
            lf_com_moved = np.dot(total_rotation_matrix, lf_com_moving)

            #print ('lf_pivot notouch', lf_pivot[0])
            #print ('lf_pivot Ang change', lf_pivot[0] * rotd_math.Bohr) #여기 까지 오케이
            testing_unit =lf_pivot_zero_moved * rotd_math.Bohr
            for_visual.append(lf_pivot_zero_moved) # B 
            #print ('lf_pivot_zero_moved', testing_unit)
            for_visual.append(lf_pivot_one_moved) # N 
            for_visual.append(lf_com_moved) # F




        else:
            for_gas_visual = []
            labframe_gas_visual = []
            for i in range(0, 2):
            # update COM positions
                self.fragments[i].set_labframe_com(lf_com * mfactor[i]) # set_labframe_COM 의 경우 아래에 set_labframe_position 을 가져올때만 쓰임 
                # if i == 0: # 아래 적히는것들은 0,0,0 기준으로 Nonlinear가 두개있을때 비교한 값임
                #     #print ('mfo 0 ', self.fragments[0].testing_rotational_matrix()) # mfo, 즉 rotation matrix는 매번 다르게 나옴
                #     #print ('labframe com for 0', self.fragments[i].testing_lf_com()) # 1일때와 부호만 바꿔서 나옴, 왜냐면 mfactor가 각각 0.5와 -0.5 이기 때문에 for two nonlinear
                #     #print ('relative position 0', self.fragments[i].testing_rel_pos()) # 1일때와 아예 다름, molframe dot random 이기 때문에 랜덤값임. 
                #     #print ('mf position 0', self.fragments[i].testing_mf_pos()) # 이거는 1과 같게나옴
                #     print ('relative positions for frag 0', self.fragments[i].testing_real_rel_pos)
                # else:
                #     #print ('mfo 1 ', self.fragments[1].testing_rotational_matrix())
                #     # print ('labframe com for 1', self.fragments[i].testing_lf_com())
                #     # print ('relative position 1', self.fragments[i].testing_rel_pos())
                #     #print ('mf position 1', self.fragments[i].testing_mf_pos())
                #     print ('relative positions for frag 1', self.fragments[i].testing_real_rel_pos)

                ## ->>>> 결국 중요한건 initial position에 따라서 evec값이 다르게 나와서 이로인해 mf_pos가 다르게나옴. 거기서 기안하여,
                ## evec 을 프린트한뒤 기존값은 바꾸고 강제로 evec 바꿔서 테스트 해보셈
                    
                # print ('mfactor 0, ', mfactor[0]) # 1
                # print ('mfactor 1, ', mfactor[1]) # 0
                # set up the final laboratory frame of two fragment
                self.fragments[i].set_labframe_positions()
                #print ('linear lf position', self.fragments[i].set_labframe_positions() )
                for pos in self.fragments[i].get_labframe_positions():
                    new_positions.append(pos)
                    for_gas_visual.append(pos)
                    labframe_gas_visual.append(pos)

            for_gas_visual.append(lf_pivot[0])
            for_gas_visual.append(lf_pivot[1])
            for_gas_visual.append(lf_com)
            for_gas_visual.append([0,0,0])     # Helium


            #print ('gas_visual', for_gas_visual)
            labframe_gas_visual.append(lf_pivot[0])
            labframe_gas_visual.append(lf_pivot[1])             
            labframe_gas_visual.append(lf_com)            
            labframe_gas_visual.append([0,0,0])     # Helium
        if any(frag.molecule_type == MolType.SLAB for frag in self.fragments):
        
            first_carbon_position_after_labframe = new_positions[0]
            top_middle_platinum_position_after_labframe = new_positions[33] # For Surface
            length_of_carbons_and_platinum = np.linalg.norm(first_carbon_position_after_labframe - top_middle_platinum_position_after_labframe)
            rounded_length_of_carbons_and_platinum = np.round(length_of_carbons_and_platinum, 1)
            #print ('lengths of Pt-C: ',rounded_length_of_carbons_and_platinum)

            input_molecule = Atoms(self.configuration.get_chemical_symbols()[:2], # 이렇게하면 CO 만 되고 [2:] 로 하면 CO 빼고되고 [-9:] 로 하면 맨위 Pt만 선택됨.
                               self.configuration.get_positions()[:2]) 
            input_molecule_com = input_molecule.get_center_of_mass()


        else:
            first_carbon_position_after_labframe = new_positions[0]
            second_carbon_position_after_labframe = new_positions[4]
            length_of_carbons = np.linalg.norm(second_carbon_position_after_labframe - first_carbon_position_after_labframe)
            rounded_length_of_carbons = np.round(length_of_carbons, 1)

        # For gas
        # print ('[',repr(lf_pivot[0]),end=',')
        # print (repr(lf_pivot[1]),end=',') 
        # print (repr(first_carbon_position_after_labframe),end=',') 
        # print (repr(second_carbon_position_after_labframe),'],') # 이거는 gas 일때만 잘 키도록하자.
        # print (repr(top_middle_platinum_position_after_labframe),'],') # 이거는 surface 일때만 키도록 하자.

        # For Surface
        # 변환 한뒤의 좌표
        # print ('[',repr(lf_pivot_zero_moved),end=',')
        # print (repr(lf_pivot_one_moved),end=',') 
        # print (repr(first_carbon_position_after_labframe),end=',') 
        # print (repr(top_middle_platinum_position_after_labframe),'],') 

        # 변환 전의 좌표
        # print ('[',repr(lf_pivot[0]),end=',')
        # print (repr(lf_pivot[1]),end=',') 
        # print (repr(first_carbon_position_after_labframe),end=',') 
        # print (repr(top_middle_platinum_position_after_labframe),',') # 이거는 surface 일때만 키도록 하자.
        # print (repr(lf_com),'],') 

        # Length of Carbon gases

        #print ('lengths of carbon: ',rounded_length_of_carbons)

        # Lengths of Surface + Carbon gas
       
        what_is_lf_com = np.linalg.norm(lf_com)
        rounded_what_is_lf_com = np.round(what_is_lf_com, 1)
        #print ('lf_com size: ', rounded_what_is_lf_com) # 이거 위에 길이랑 gas의 경우 같음. Surface 의 경우 무엇을 기준으로 하길래 다르지?
            

        # check whether the atoms of two fragments are too close:
        # if any(frag.molecule_type == MolType.SLAB for frag in self.fragments):
        #     if self.if_fragments_too_close_for_surface():
        #         return SampTag.SAMP_ATOMS_CLOSE
        # else:
        #     if self.if_fragments_too_close():
        #         return SampTag.SAMP_ATOMS_CLOSE
            
        if self.if_fragments_too_close():
                return SampTag.SAMP_ATOMS_CLOSE

        # check the distance between pivot points of other faces
        for face in range(0, self.div_surface.get_num_faces()):
            if face == curr_face:
                continue
            else:
                # print ('checking sample') # 거의 항상 checking sample 로 들어옴
                if self.check_sample(face, lf_com.copy()) == SampTag.SAMP_FACE_OUT:
                    return SampTag.SAMP_FACE_OUT


        # until now, the sampling is valid, set up the configuration
        new_positions = np.array(new_positions)  # in units of Bohr, Currently in Labframe      

        if any(frag.molecule_type == MolType.SLAB for frag in self.fragments):
            for_visual = np.array(for_visual)
            for_visual *= rotd_math.Bohr
            self.visual_configuration.set_positions(for_visual) # For visual, in Ang      

            for_surface_lf_visual = np.array(for_surface_lf_visual)  # in units of Bohr
            # for_surface_lf_visual *= rotd_math.Bohr
            self.surface_labframe_configuration.set_positions(for_surface_lf_visual)

        else:
            for_gas_visual = np.array(for_gas_visual)
            for_gas_visual *= rotd_math.Bohr
            self.gas_visual_configuration.set_positions(for_gas_visual)

            test_gas_labframe_positions = np.array(labframe_gas_visual)
            self.gas_labframe_configuration.set_positions(test_gas_labframe_positions)


     
        
        #test_labframe_positions *= rotd_math.Bohr
     
        # Original convert back to angstrom
        new_positions *= rotd_math.Bohr # in units of Ang
        #print ('Ang position ', new_positions)
        # if SampTag.SAMP_SUCCESS:
        #     print ('success in Angstrom: ', new_positions)

        #TEMP: Move C and O by 20 angstrom in z-direction
        # 이게 없으면 NO Sampling found!! 이건 오리지널로 여기서 작동하는거 확인됨
        # if frag.molecule_type == MolType.SLAB:
        #     new_positions[0][2] += 1. # Ang
        #     new_positions[1][2] += 1.
        #     test_labframe_positions[0][2] += 2.35 # Bohr
        #     test_labframe_positions[1][2] += 2.35

        #TEST: Move C and O by 20 angstrom in z-direction = 37.7945 Bohr
        # print ('checking complete')

        self.configuration.set_positions(new_positions) # Labframe, in Ang

        #Check the distance of slab and molecule  
        # 여기여기
        # if frag.molecule_type == MolType.SLAB and self.check_in_rhombus(): # This works fine 240605
        #     return SampTag.SAMP_ATOMS_CLOSE
                
        # if frag.molecule_type == MolType.SLAB and self.check_molecule_z_coordinate():
        #     return SampTag.SAMP_ATOMS_CLOSE

        if frag.molecule_type == MolType.SLAB and self.check_in_surface():
            # print ('is working')
            return SampTag.SAMP_ATOMS_CLOSE

        
        # calculate the kinematic weight (Phi in the equation) << I'm sure! 0702/2024
        rc_vec = com_vec.copy()  # orbital coordinates (reaction coordinates)
        #print ("rc_vec initial", rc_vec)
        rc_vec /= np.sqrt(emass)  # This means  a = a/b (i.e. rc_vec = rc_vec/np.sqrt(emass) emass = reduced mass
        #print ("rc_vec with mass", rc_vec)
        test_rc_vec = com_vec.copy()
        test_rc_vec /= np.sqrt(emass)
        # internal coordinates
        # C++ 설명으론,
        # transform a vector between ref. points to molecular frame 
        there_is_slab = False
        for f in self.fragments:
            if f.molecule_type == MolType.SLAB:
                there_is_slab = True
                break
        
        if there_is_slab:
            for index, frag in enumerate(self.fragments):
                if frag.molecule_type == MolType.MONOATOMIC:
                    continue
                elif frag.molecule_type == MolType.LINEAR:
                    # Original # C++ 코드에서도 이거로 했음..
                    if index == 0:
                        rc_vec = np.append(rc_vec, np.cross(lf_pivot[index, :], com_vec)
                                           / np.sqrt(frag.get_inertia_moments_for_surface())) # 랩프레임
                    else:
                        rc_vec = np.append(rc_vec, np.cross(com_vec, lf_pivot[index, :])
                                        / np.sqrt(frag.get_inertia_moments_for_surface()))
                    
                elif frag.molecule_type == MolType.NONLINEAR: # 이거 molframe 으로 바꾸기!!!!!
                    mf_com = frag.lf2mf(com_vec)
                    #print ("multisample lf2mf internal coord: ", mf_com)
                    if index == 0:
                        rc_vec = np.append(rc_vec, np.cross(molframe_pivot, mf_com)/np.sqrt(frag.get_inertia_moments())) # np.cross = multiplication of vectors
                    #print ('rc_vec pivot point ', self.div_surface.get_pivot_point(index, curr_face))
                    #print ("rc_vec index ==0", rc_vec)
                    else: # index == 1
                        rc_vec = np.append(rc_vec, np.cross(mf_com, molframe_pivot) /
                                           np.sqrt(frag.get_inertia_moments()))
                    
                elif frag.molecule_type == MolType.SLAB:
                # mf_com = frag.lf2mf(com_vec)
                # if index == 0:
                #     rc_vec = np.append(rc_vec, np.cross(self.div_surface.get_pivot_point(index, curr_face),
                #                                         mf_com)/np.sqrt(frag.get_inertia_moments())) # np.cross = multiplication of vectors
                #     #print ("rc_vec index ==0", rc_vec)
                # else: # index == 1
                #     rc_vec = np.append(rc_vec, np.cross(mf_com,
                #                                         self.div_surface.get_pivot_point(index, curr_face)) /
                #                        np.sqrt(frag.get_inertia_moments()))
                #print ('rc_vec initial', rc_vec)
                    continue
                    #print ("rc_vec else", rc_vec)

        else:
            for index, frag in enumerate(self.fragments):
                if frag.molecule_type == MolType.MONOATOMIC:
                    continue
                elif frag.molecule_type == MolType.LINEAR:
                    # Original # C++ 코드에서도 이거로 했음..
                    if index == 0:
                        rc_vec = np.append(rc_vec, np.cross(lf_pivot[index, :], com_vec)
                                           / np.sqrt(frag.get_inertia_moments())) # 랩프레임
                    else:
                        rc_vec = np.append(rc_vec, np.cross(com_vec, lf_pivot[index, :])
                                        / np.sqrt(frag.get_inertia_moments()))
                elif frag.molecule_type == MolType.NONLINEAR: 
                    mf_com = frag.lf2mf(com_vec)
                    if index == 0:
                        rc_vec = np.append(rc_vec, np.cross(self.div_surface.get_pivot_point(index, curr_face),
                                                            mf_com)/np.sqrt(frag.get_inertia_moments())) # np.cross = multiplication of vectors
                    else: # index == 1
                        rc_vec = np.append(rc_vec, np.cross(mf_com,
                                                            self.div_surface.get_pivot_point(index, curr_face)) /
                                           np.sqrt(frag.get_inertia_moments()))
                        
        # print ('lfactor')
        #print ('div_surface.get_dist(curr_face)^2: ', self.div_surface.get_dist(curr_face) ** 2)
        #print ('normalized rc_vec: ', rotd_math.normalize(rc_vec))
        #print ('two multiplied: ', self.div_surface.get_dist(curr_face) ** 2 *rotd_math.normalize(rc_vec))
        #print ('rc_vec after calc, ', rc_vec)
        self.weight = lfactor * self.div_surface.get_dist(curr_face) ** 2 * \
                      rotd_math.normalize(rc_vec)
        
        #print ('end rc_vec', rc_vec) # 왜 위랑 다른값이지..?
        #print ('self.weight,: ', self.weight)
        #print ("self.weight", self.weight) This is real weight. Not always 1
        
        #print ("dist_cur_face: ", self.div_surface.get_dist(curr_face)) # This is the value given in the input file
        
        #print ("lfactor in weight: ", lfactor) 

        #test = rotd_math.normalize(rc_vec)
        #print ("normalized rc_vec ", test) # returns 1 or 0.999999

      
        #print ('frag.whole labframe', labframe_new_positions ) # This includes both frag[0] and frag[1]
        #print ('Inside Angstrom position',self.configuration.get_positions()) #위랑 같은데 형식이 다름
        #print ('lf_pivot', lf_pivot)
        return SampTag.SAMP_SUCCESS

    # Defined as #reference points in laboratory frame in C++
    def labframe_pivot_point(self, frag_index, com_vec, lf_pivot, molframe_pivot, lfactor):
        """Converting the pivot point coordinates in the lab frame coordinates.

        Parameters
        ----------
        frag_index : int
            the index of the current fragment
        com_vec : 1D numpy array
            the reaction coordinate vector for the current sampling
        lf_pivot : 1D numpy array
            the pivot point vector in the molecular frame
        lfactor : float
            factor will be used in weight calculation

        Returns
        -------
        type Boolean
            whether this sample is valid or not

        """

        frag = self.fragments[frag_index] #fragment.index 0 & 1
        temp_com = com_vec.copy()  # in case modify the original com_vec
        face = self.div_surface.get_curr_face()
        
        #check if there is any slab in fragments
        frag_molecule_types = np.array([f.molecule_type for f in self.fragments])
        # print ('frag molecule type', frag_molecule_types)
        # above line is equivalent to:
        # frag_molecule_types = []
        # for f in self.fragments:
        #     frag_molecule_types.append(f.molecule_type)
        # there_is_slab = np.any(frag_molecule_types) #bool
        # if there_is_slab:     
        there_is_slab = False
        for f in self.fragments:
            if f.molecule_type == MolType.SLAB:
                there_is_slab = True
                break
        
        if there_is_slab:
            #print ('there is slab')
            if frag.molecule_type == MolType.MONOATOMIC:
                lf_pivot = np.zeros(3)
                molframe_pivot = np.zeros(3)
                #print ("New, MONOATOM labframe pivot point: np.zeros(3)", lf_pivot)
                return [SampTag.SAMP_SUCCESS, lfactor]

            elif frag.molecule_type == MolType.NONLINEAR:

                nonlinear_com = frag.get_center_of_mass()
                nonlinear_pivot_rel_pos = self.div_surface.get_pivot_point(frag_index, face) - nonlinear_com
                nonlinear_pivot_rel_pos /= rotd_math.Bohr
                nonlinear_mf_temp = np.dot(nonlinear_pivot_rel_pos, frag.frag_array['orig_mfo']) #nonlinear_mf_temp = molframe 상태
                rotated_nonlinear_lf_temp = frag.mf2lf(nonlinear_mf_temp)
                
                for i in range(0, len(lf_pivot)):
                    lf_pivot[i] = rotated_nonlinear_lf_temp[i]
                    molframe_pivot[i] = nonlinear_mf_temp
               
                return [SampTag.SAMP_SUCCESS, lfactor]

            elif frag.molecule_type == MolType.LINEAR: # For slab case, we should consider it as a spherical.
                # reorient the pivot point according to the fragment reorientation

                #This is original linear.. /fragment/linear.py original 과 밑에 check_sample 까지 한번에 바꿔야함 #1
                # lf_temp = frag.get_ang_pos() * self.div_surface.get_pivot_point(frag_index, face)[0]
                # # print ('linear pivot [0] ', self.div_surface.get_pivot_point(frag_index, face)[0]) #내가 준 값
                # #print ('angpos multiplied ', lf_temp) # 랜덤함 x,y,z  
                # for i in range(0, len(lf_temp)):
                #     lf_pivot[i] = lf_temp[i]
                

                # Test2
                # linear_com = frag.get_center_of_mass()
                # linear_pivot_rel_pos = self.div_surface.get_pivot_point(frag_index, face)- linear_com
                # linear_pivot_rel_pos /= rotd_math.Bohr
                # lf_temp_original = frag.get_ang_pos() * linear_pivot_rel_pos[0]
                # lf_temp = np.dot(linear_pivot_rel_pos, frag.frag_array['orig_mfo']) #orig_mfo = evecs
                # lf_temp = frag.mf2lf(lf_temp)

                # 하... 8/9 테스트 3
                # testing_pivot = self.div_surface.get_pivot_point(frag_index, face)
                # testing_pivot /= rotd_math.Bohr
                # lf_temp = np.dot(testing_pivot, frag.frag_array['orig_mfo']) #orig_mfo = evecs
                # lf_temp = frag.mf2lf(lf_temp)
                # # #print ('nonlinear orig_mfo: ', frag.frag_array['orig_mfo'])
                # # #print ('nonlinear get_pivot: ',self.div_surface.get_pivot_point(frag_index, face) )
                # for i in range(0, len(lf_pivot)):
                #     lf_pivot[i] = lf_temp[i] 

                # print ('linear pivot [0] ', self.div_surface.get_pivot_point(frag_index, face)[0]) #내가 준 값
                #print ('angpos multiplied ', lf_temp) # 랜덤함 x,y,z  
                # print ('original', lf_temp_original) #COM 일때는 아래 두개가 같을거임
                # print ('new lf_temp', lf_temp)

                # My test 
                linear_com = frag.get_center_of_mass()
                linear_pivot_rel_pos = self.div_surface.get_pivot_point(frag_index, face)- linear_com
                linear_pivot_rel_pos /= rotd_math.Bohr
                lf_temp = np.dot(linear_pivot_rel_pos, frag.frag_array['orig_mfo']) #orig_mfo = evecs
                #print ('orig_mfo', frag.frag_array['orig_mfo']) # [[-1.0000000e+00 -6.5523478e-31  0.0000000e+00][-6.5523478e-31  1.0000000e+00  0.0000000e+00][-0.0000000e+00  0.0000000e+00  1.0000000e+00]]
                # print ('lf_temp after first dot')
                lf_temp = frag.mf2lf(lf_temp)
                #print ('lf_temp after mf2lf', lf_temp) #무수히 많은 랜덤 1x3 array
                
                for i in range(0, len(lf_pivot)):
                    lf_pivot[i] = lf_temp[i]

                return [SampTag.SAMP_SUCCESS, lfactor]
            
    #######################################################################################################
    # Test for SLAB
            elif frag.molecule_type == MolType.SLAB:
                # 바꿔야하는데...
                top_slab = Atoms(self.configuration.get_chemical_symbols()[-9:],
                            self.configuration.get_positions()[-9:]) 
                #print ('top_slab',top_slab)
                top_slab_com = top_slab.get_center_of_mass()
                #print ('com',top_slab_com)

                # origin_slab = Atoms(self.configuration.get_chemical_symbols()[-36:],
                #                   self.configuration.get_positions()[-36:]) 
                # origin_slab_com = origin_slab.get_center_of_mass()

                mol_pos = frag.get_molframe_positions()            
                slab_com = frag.get_center_of_mass()

                # 슬랩과 똑같이 돌리기
                slab_pivot_rel_pos = self.div_surface.get_pivot_point(frag_index, face) - slab_com
                slab_pivot_rel_pos /= rotd_math.Bohr
                lf_temp = np.dot(slab_pivot_rel_pos, frag.frag_array['orig_mfo']) #orig_mfo = evecs

                # 새롭게 돌려보기
                # slab_pivot_test_pos = self.div_surface.get_pivot_point(frag_index, face)
                # slab_pivot_test_pos /= slab_pivot_test_pos
                # lf_temp = np.dot(slab_pivot_test_pos, frag.frag_array['orig_mfo']) #orig_mfo = evecs             

                # print ('lf_temp: ', lf_temp)
                #print ('slab pivot', self.div_surface.get_pivot_point(frag_index, face))
                # print ('slab orig_mfo: ', frag.frag_array['orig_mfo']) #[[ 0.85716015 -0.05286034 -0.51233023][0.49902208 -0.16098632  0.85150477][-0.12748899 -0.98554005 -0.11161258]]
                #test_position = Atoms(self.configuration.get_chemical_symbols()[-1:], # -1로 마지막인 37 가져옴
                #                      self.configuration.get_positions()[-1:])
                # test_position = np.array([8.29966584, 4.79181431, 6.7766488])
                # dot_test_position = np.dot(test_position, frag.frag_array['orig_mfo']) # 이거로 기존 위치에서 mf 상태로 됨.
                #print ('test_dot_pos', dot_test_position)

                #print ('test_position', test_position) 
                #test_position_com = test_position.get_center_of_mass()
                #print ('test_position com', test_position_com) # 37번 com 값 가져옴. 근데 처음 두번을 제외하곤 다른 값이 나옴.

                # Original
                # lf_temp = np.dot(self.div_surface.get_pivot_point(frag_index, face), frag.frag_array['orig_mfo']) #orig_mfo = evecs

                # 여기는 노터치 어쩌면 for loop 터치
                lf_temp = frag.mf2lf(lf_temp) 
                #labframe_dot_test_position = frag.mf2lf(dot_test_position) #rotation matrix에서 가져온 랜덤값 만큼 mf 을 돌려서 lf로 만듬.
                for i in range(0, len(lf_temp)):
                    lf_pivot[i] = lf_temp[i]
                # print ('lf_pivot of slab',lf_pivot) # 랜덤값이 나오고있음.

                
                #print ('slab lf_pivot: ', lf_pivot) # 내가 준 pivot point 그대로 왜 mf2lf 거쳤는데 그대로지?
                # if any(frag.molecule_type == MolType.LINEAR):
                #     lf_temp = frag[0].get_ang_pos() * self.div_surface.get_pivot_point(frag_index, face)[0]
                #     print ('slab get_ang_pos: ', frag.get_ang_pos())
                #     print ('slab lf_temp: ', lf_temp) # [0,0,7]

                #     for i in range(0, len(lf_temp)): # 이거 lf_pivot 아니면 lf_temp 임 아직 뭔지 모름
                #         lf_pivot[i] = lf_temp[i]

                # elif any(frag.molecule_type == MolType.NONLINEAR):
                #     lf_temp = np.dot(self.div_surface.get_pivot_point(frag_index, face), frag.frag_array['orig_mfo']) #orig_mfo = evecs
                #     # return pivot_point_vector in molecule frame * rotation matrix
                #     print ('wrong thing printing')
                #     lf_temp = frag.mf2lf(self.div_surface.get_pivot_point(frag_index, face))
                #     for i in range(0, len(lf_pivot)):
                #         lf_pivot[i] = lf_temp[i]

                return [SampTag.SAMP_SUCCESS, lfactor]
            

            else:
                raise ValueError("MolType is invalid")
        else: # 만약 슬랩이 없다면 여기로
            #print ("This is printed since there is no slab fragment!")
            if frag.molecule_type == MolType.MONOATOMIC:
                lf_pivot = np.zeros(3)
                return [SampTag.SAMP_SUCCESS, lfactor]

            elif frag.molecule_type == MolType.NONLINEAR:
                #print ('you are in nonlinear lf_pivot without slab')

                # Original
                # lf_temp = frag.mf2lf(self.div_surface.get_pivot_point(frag_index, face))

                # Clement & Me
                lf_temp = np.dot(self.div_surface.get_pivot_point(frag_index, face), frag.frag_array['orig_mfo']) #orig_mfo = evecs
                lf_temp = frag.mf2lf(lf_temp)
                #print ('nonlinear orig_mfo: ', frag.frag_array['orig_mfo'])
                #print ('nonlinear get_pivot: ',self.div_surface.get_pivot_point(frag_index, face) )
                for i in range(0, len(lf_pivot)):
                    lf_pivot[i] = lf_temp[i]
                
                # input_molecule = Atoms(self.configuration.get_chemical_symbols(),
                #                        self.configuration.get_positions()) 
                # #print ('input molecule',input_molecule) # CH3CH3
                # #print ('position', self.configuration.get_positions())
                # input_molecule_com = input_molecule.get_center_of_mass()
                # #print ('com',input_molecule_com)
                # empty_com = input_molecule_com.copy()

                return [SampTag.SAMP_SUCCESS, lfactor]

            elif frag.molecule_type == MolType.LINEAR:
                if self.div_surface.get_pivot_point(frag_index, face)[1] > 0.0:  # toroidal surface
                    #print ('linear pivot 3',self.div_surface.get_pivot_point(frag_index, face))
                    #print ('linear pivot 3 [1] ',self.div_surface.get_pivot_point(frag_index, face)[1]) #정말 y coordinate printed
                    temp_com = rotd_math.orthogonalize(temp_com, frag.get_ang_pos())
                    dtemp = rotd_math.normalize(temp_com)

                    # get random number
                    itemp = 0  # external part of the torus
                    if np.random.random_sample() > 0.5:
                        itemp = 1  # internal part of the torus

                    u = self.div_surface.pivot_point(frag_index)[1]
                    if itemp:
                        u -= dtemp*self.div_surface.get_dist(face)
                    else:
                        u += dtemp*self.div_surface.get_dist(face)

                    if u <= 0.0:
                        return [SampTag.SAMP_FACT_OUT, lfactor]
                    if dtemp > 1.0e-14:
                        lfactor = 2. * u/dtemp/self.div_surface.get_dist(face)
                        #print ("Since nonlinear, this should not be printed") Not printed.

                    else:
                        lfactor = 0.0
                        #print ("Since nonlinear, this should not be printed") Not printed.

                    lf_pivot = frag.get_ang_pos() * self.div_surface.get_pivot_point(frag_index, face)[0]
                    #print ("ang_pos*pivot", lf_pivot)
                    if (itemp+frag_index) % 2:
                        lf_pivot += temp_com * self.div_surface.get_pivot_point(frag_index, face)[1]
                    else:
                        lf_pivot -= temp_com * self.div_surface.get_pivot_point(frag_index, face)[1]
                    #print ('linear pivot toridal: ', lf_pivot)
                    return [SampTag.SAMP_SUCCESS, lfactor]

                else:  # spherical surface
                    # 여기여기 오리지널
                    # 1. 왜 [0] 인지 
                    # 2. 왜 nonlinear 랑 다르게 get_ang_pos 곱하는지
                    lf_temp = frag.get_ang_pos() * self.div_surface.get_pivot_point(frag_index, face)[0]
                    #lf_temp = frag.get_ang_pos() * self.div_surface.get_pivot_point(frag_index, face)
                    
                    #print ('linear lf_temp: ', lf_temp) #[0,0,0]
                    #lf_temp = np.dot(self.div_surface.get_pivot_point(frag_index, face), frag.frag_array['orig_mfo']) #orig_mfo = evecs
                    #print ('orig_mfo: ', frag.frag_array['orig_mfo']) # [1,1,1] diagonal matrix
                    #print ('get_pivot: ', self.div_surface.get_pivot_point(frag_index, face)[0])
                    # print ('linear get_ang_pos: ', frag.get_ang_pos()) # random vector

                    #print ('co lf_temp: ', lf_temp) #[0,0,0]
                    for i in range(0, len(lf_temp)):
                        lf_pivot[i] = lf_temp[i]
                    
                    # print ('linear lf_pivot spherical: ', lf_pivot) #[0,0,0]

                    #print ('lf_temp (0): ',self.div_surface.get_pivot_point(frag_index, face)[0]) # 둘다 0
                    #print ('lf_temp (1): ',self.div_surface.get_pivot_point(frag_index, face)[1]) # 둘다 0

                    return [SampTag.SAMP_SUCCESS, lfactor]
            else:
                raise ValueError("MolType is invalid")



    def check_sample(self, face, lf_com):
        """Validate the current sample based on the face with index "face".

        Parameters
        ----------
        face : int
            The index of the target face
        lf_com : 1*3 numpy array
            The center of mass labframe vector

        Returns
        -------
        SampTag

        """

        frag_molecule_types = np.array([f.molecule_type for f in self.fragments])
        # print ('frag molecule type', frag_molecule_types)
        # above line is equivalent to:
        # frag_molecule_types = []
        # for f in self.fragments:
        #     frag_molecule_types.append(f.molecule_type)
        # there_is_slab = np.any(frag_molecule_types) #bool
        # if there_is_slab:     
        there_is_slab = False
        for f in self.fragments:
            if f.molecule_type == MolType.SLAB:
                there_is_slab = True
                break
        
        if all(frag.molecule_type == MolType.LINEAR for frag in self.fragments):
            for i in range(0, 2):
                lf_com[i] = self.div_surface.get_pivot_point(0, face)[0] * \
                    self.fragments[0].get_ang_pos()[i] - \
                    self.div_surface.get_pivot_point(1, face)[0] * \
                    self.fragments[1].get_ang_pos()[i]
        else:                 
            if there_is_slab:
                #print ('there is slab')
                for i, frag in enumerate(self.fragments):
                    mf_pivot_point = self.div_surface.get_pivot_point(i, face)

                    if frag.molecule_type == MolType.NONLINEAR:
                        lf_pivot_point = frag.mf2lf(mf_pivot_point)
                        if i == 1:
                            lf_com -= lf_pivot_point
                        else:
                            lf_com += lf_pivot_point

                    elif frag.molecule_type == MolType.SLAB:
                        lf_pivot_point = frag.mf2lf(mf_pivot_point)
                        #print ('lf_pivot_point input, aka mf_pivot_point ', mf_pivot_point)
                        #print ('lf_pivot_point in check_sample ', lf_pivot_point)
                        if i == 1:
                            lf_com -= lf_pivot_point
                        else:
                            lf_com += lf_pivot_point
                            #print ('else lf_com ', lf_com)

                    elif frag.molecule_type == MolType.LINEAR:                     
                        # original /fragment/linear.py original 과 위에 labframe_pivot 까지 한번에 바꿔야함 #1
                        # if i == 1: # 여기 x 좌표 가져옴 0으로
                        #     lf_com -= frag.get_ang_pos() * mf_pivot_point[0]
                        #     #print ('mffffffff') # 당연히 안나옴 현재 i==1 은 surface
                        # else:
                        #     lf_com -= frag.get_ang_pos() * mf_pivot_point[0]
                            # print ('mf_pivot[0]', mf_pivot_point[0]) #처음 copt.py 에서 내가 준 값
                            # print ('what is minus? ', frag.get_ang_pos() * mf_pivot_point[0])

                        # My test
                        lf_pivot_point = frag.mf2lf(mf_pivot_point)
                        if i == 1: # 여기 x 좌표 가져옴 0으로
                            lf_com -= lf_pivot_point
                        else:
                            lf_com -= lf_pivot_point

            else: # 슬랩이 없다면 여기로옴
                #print ('there is no slab')
                for i, frag in enumerate(self.fragments):
                    mf_pivot_point = self.div_surface.get_pivot_point(i, face)

                    if frag.molecule_type == MolType.NONLINEAR:
                        lf_pivot_point = frag.mf2lf(mf_pivot_point)
                        if i == 1:
                            lf_com -= lf_pivot_point
                        else:
                            lf_com += lf_pivot_point

                    elif frag.molecule_type == MolType.LINEAR:
                        if mf_pivot_point[1] > 1.0:  # torus
                            pos_temp = lf_com.copy()
                            p0 = np.dot(pos_temp, frag.get_ang_pos())
                            if i == 1:
                                p0 -= mf_pivot_point[0]
                            else:
                                p0 += mf_pivot_point[0]
                            rotd_math.orthogonalize(pos_temp, frag.get_ang_pos())
                            p1 = rotd_math.normalize(pos_temp)
                            p1 -= mf_pivot_point[1]

                            lf_com = p0 * frag.get_ang_pos() + p1 * pos_temp
                        # end torus
                        else:
                            if i == 1:
                                lf_com -= frag.get_ang_pos() * mf_pivot_point[0]
                            else:
                                lf_com += frag.get_ang_pos() * mf_pivot_point[0]

            # print ('norm lf_com ',np.linalg.norm(lf_com))
            # print ('face dist',  self.div_surface.get_dist(face)) # 내가 처음에 준 distance 값
            # now calculate the distance between the two pivot point
            if np.linalg.norm(lf_com) < self.div_surface.get_dist(face):
                # print('fail')
                return SampTag.SAMP_FACE_OUT
            else:
                # print ('at least succeded')
                #print ('face dist',  self.div_surface.get_dist(face)) # 내가 처음에 준 distance 값
                # 프린트해보니 이건 어지간하면 성공함.
                return SampTag.SAMP_SUCCESS            