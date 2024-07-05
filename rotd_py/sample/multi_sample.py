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
        ### 무조건 if 랑 else가 둘다 돌고있음
 
        for frag in self.fragments:
            orient = rotd_math.random_orient(frag.get_ang_size()) 
            frag.set_ang_pos(orient)
            frag.set_rotation_matrix()
            #print ('frag', frag)

        


            #print ('tonghap orient', orient)
            # if frag.molecule_type == MolType.SLAB:
            #     #print ("hm", frag)
                
            #     orient = rotd_math.random_orient(frag.get_ang_size()) 
            #     frag.set_ang_pos(orient)
            #     frag.set_rotation_matrix()

            #     #print ("slab orient", orient)
            #     # c = FixAtoms(indices = [atom.index for atom in frag if atom.symbol =='Pt'])
            #     # frag.set_constraint(c)
            #     #traj = Trajectory('slab.traj', 'a', frag)
            #     #traj.write()
            #     #traj.close()
            # else:    
            #     orient = rotd_math.random_orient(frag.get_ang_size())
            #     frag.set_ang_pos(orient)
            #     frag.set_rotation_matrix() # 키나 안키나 어차피 작동 안함 -> 왜냐면 linear 는 이걸 안쓰거든~ -> Nonlinear 랑 통일해서 이젠 쓰임.
            #     #print ('linear orient', orient)
            #     #traj2 = Trajectory('linear.traj', 'a', frag)
            #     #traj2.write()
            #     #traj2.close()

        # for frag in self.fragments:
        #     if frag.molecule_type == MolType.SLAB:
        #         frag.set_slab_principal_axis_for_visualization()
        #     else:
        #         frag.set_adsorbate_principal_axis_for_visualization()
    
        #print ('working????')
        # get the pivot points coordinates for current face in the laboratory frame
        lfactor = 1.0
        lf_pivot = np.zeros((2, 3))
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
                    i, com_vec, lf_pivot[i], lfactor) 
                # com_vec = rotd_math.random_orient(3)

                if sample_info == SampTag.SAMP_FACE_OUT:
                    return SampTag.SAMP_FACE_OUT
                
        #print ("0 Pivot: ", lf_pivot[0])
        #print ("1 Pivot: ", lf_pivot[1])
        #now set the reaction coordinate vector
        # lf_com 의 경우 check_sample 에서와 아래 set_labframe_COM 일때 쓰임
        lf_com = com_vec * self.div_surface.get_dist(curr_face) - lf_pivot[0] + lf_pivot[1] # lf_pivot 에 get_rotation_matrix 들어가있음. 
        #print ('com_vec, random ', com_vec)
        #print ('yangsik for lf_com',lf_com) # 바뀜
        #print ('distance', self.div_surface.get_dist(curr_face)) # 내가 initial 로 준 값
        #print ('lf_pivot[0]', lf_pivot[0]) # 바뀜
        #print ('lf_pivot[1]', lf_pivot[1]) # 바뀜
        
        new_positions = []
        if any(frag.molecule_type == MolType.SLAB for frag in self.fragments):

            for i in range(0, 2):
            # update COM positions

                self.fragments[i].set_labframe_com(lf_com * mfactor[i]) # set_labframe_COM 의 경우 아래에 set_labframe_position 을 가져올때만 쓰임 
                #self.fragments[i].set_labframe_com(lf_com) # set_labframe_COM 의 경우 아래에 set_labframe_position 을 가져올때만 쓰임 
                # print ('lf_com * mfactor, ', self.fragments[1].set_labframe_com(lf_com * mfactor[1]))
                #print ('mfactor 0 CO, ', mfactor[0]) # 1
                #print ('mfactor 1 Pt, ', mfactor[1]) # 0

                # set up the final laboratory frame of two fragment
                # self.fragments[i].set_labframe_positions_for_surface_rotd()
                self.fragments[i].set_labframe_positions()
                # self.fragments[i].set_labframe_position_only_for_visualization()


            
                for pos in self.fragments[i].get_labframe_positions():
                    new_positions.append(pos)   
                    # for_visual.append(pos_visual)                

                   
                # for pos in self.fragments[i].get_visualization_positions():
                #     for_visual.append(pos)

            for_visual = []

            for i in range(0, 2):
                for pos in self.fragments[i].get_labframe_positions():
                    moving_vector, total_rotation_matrix = self.fragments[1].slab_straightening_matrix_calculation()

                    pos_visual = pos - moving_vector
                    pos_visual = np.dot(total_rotation_matrix, pos_visual)
                    for_visual.append(pos_visual)

        else:
            for i in range(0, 2):
            # update COM positions
                self.fragments[i].set_labframe_com(lf_com * mfactor[i]) # set_labframe_COM 의 경우 아래에 set_labframe_position 을 가져올때만 쓰임 
                # set up the final laboratory frame of two fragment
                self.fragments[i].set_labframe_positions()
                #print ('linear lf position', self.fragments[i].set_labframe_positions() )
            
                for pos in self.fragments[i].get_labframe_positions():
                    new_positions.append(pos)
        # test_traj.close()

        # check whether the atoms of two fragments are too close:
        if self.if_fragments_too_close():
            return SampTag.SAMP_ATOMS_CLOSE
       
        """        
        if frag.molecule_type == MolType.SLAB and self.if_slab_and_molecule_too_close():
            print ("distance test working")
            #traj = Trajectory('wtf happening.traj', 'a', lb_pos_platinum)
            #traj.write()
            #traj.close()
            return SampTag.SAMP_ATOMS_CLOSE
        """

        # check the distance between pivot points of other faces
        for face in range(0, self.div_surface.get_num_faces()):
            if face == curr_face:
                continue
            else:
                #print ('checking sample') # 거의 항상 checking sample 로 들어옴
                if self.check_sample(face, lf_com.copy()) == SampTag.SAMP_FACE_OUT:
                    return SampTag.SAMP_FACE_OUT


        # until now, the sampling is valid, set up the configuration
        new_positions = np.array(new_positions)  # in units of Bohr, Currently in Labframe      
        for_visual = np.array(for_visual)
        # Testing Bohr
        # if frag.molecule_type == MolType.SLAB: # 6 seems to be minimum
        #     new_positions[0][2] += 20
        #     new_positions[1][2] += 20
        
        # if frag.molecule_type == MolType.SLAB:
        #     new_positions[0][2] += 37.7945
        #     new_positions[1][2] += 37.7945
        
        test_labframe_positions = np.array(new_positions)  # in units of Bohr
        #test_labframe_positions *= rotd_math.Bohr
     
        # Original convert back to angstrom
        new_positions *= rotd_math.Bohr # in units of Ang
        for_visual *= rotd_math.Bohr
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


        # Testing Bohr
        #new_positions *= rotd_math.Bohr # in units of Ang

        self.configuration.set_positions(new_positions) # Labframe, in Ang
        self.labframe_configuration.set_positions(test_labframe_positions)
        self.visual_configuration.set_positions(for_visual) # For visual, in Ang
     
        # if frag.molecule_type == MolType.SLAB:
        #     self.fragments[i].unit_cell_mf()       
        #     orig_unit_cell_pos = self.fragments[i].get_molframe_unit_cell()
        #     rotated_unit_cell_334_pt = self.fragments[i].unit_cell_mf2lf(orig_unit_cell_pos)
        #     rotated_unit_cell_334_pt *= rotd_math.Bohr
        #     self.configuration.set_celldisp(-np.array([30,30,30]))
        #     self.configuration.get_celldisp()
        #     self.configuration.set_cell(rotated_unit_cell_334_pt, scale_atoms=True)
            

        #Check the distance of slab and molecule  
        # 여기여기
        # if frag.molecule_type == MolType.SLAB and self.check_in_rhombus(): # This works fine 240605
        #     return SampTag.SAMP_ATOMS_CLOSE
                
        # if frag.molecule_type == MolType.SLAB and self.check_molecule_z_coordinate():
        #     return SampTag.SAMP_ATOMS_CLOSE

        if frag.molecule_type == MolType.SLAB and self.check_in_surface():
            # print ('is working')
            return SampTag.SAMP_ATOMS_CLOSE

        
        # calculate the kinematic weight (Phi in the equation) << I'm sure!
        rc_vec = com_vec.copy()  # orbital coordinates (reaction coordinates)
        #print ("rc_vec initial", rc_vec)
        rc_vec /= np.sqrt(emass)  # This means  a = a/b (i.e. rc_vec = rc_vec/np.sqrt(emass) emass = reduced mass
        #print ("rc_vec with mass", rc_vec)
        # internal coordinates
        for index, frag in enumerate(self.fragments):
            if frag.molecule_type == MolType.MONOATOMIC:
                continue
            elif frag.molecule_type == MolType.SLAB:
                #print ('index moleculetype = SLAB, ', index)
                continue
            elif frag.molecule_type == MolType.LINEAR:
                # Original
                if index == 0:
                    rc_vec = np.append(rc_vec, np.cross(lf_pivot[index, :], com_vec)
                                       / np.sqrt(frag.get_inertia_moments())) # 이거이거 이상하게 eval을 [2] 만 가져와서 *3 해주는데 맞아?
                else:
                    rc_vec = np.append(rc_vec, np.cross(com_vec, lf_pivot[index, :])
                                       / np.sqrt(frag.get_inertia_moments()))
                    
                # My test
                # mf_com = frag.lf2mf(com_vec)
                # if index == 0:
                #     rc_vec = np.append(rc_vec, np.cross(self.div_surface.get_pivot_point(index, curr_face),
                #                                         mf_com)/np.sqrt(frag.get_inertia_moments())) # np.cross = multiplication of vectors
                # else: # index == 1
                #     rc_vec = np.append(rc_vec, np.cross(mf_com,
                #                                         self.div_surface.get_pivot_point(index, curr_face)) /
                #                        np.sqrt(frag.get_inertia_moments()))

            elif frag.molecule_type == MolType.NONLINEAR:
                mf_com = frag.lf2mf(com_vec)
                #print ("multisample lf2mf internal coord: ", mf_com)
                if index == 0:
                    rc_vec = np.append(rc_vec, np.cross(self.div_surface.get_pivot_point(index, curr_face),
                                                        mf_com)/np.sqrt(frag.get_inertia_moments())) # np.cross = multiplication of vectors
                    #print ("rc_vec index ==0", rc_vec)
                else: # index == 1
                    rc_vec = np.append(rc_vec, np.cross(mf_com,
                                                        self.div_surface.get_pivot_point(index, curr_face)) /
                                       np.sqrt(frag.get_inertia_moments()))
                    #print ("rc_vec else", rc_vec)
        # print ('lfactor')
        #print ('div_surface.get_dist(curr_face)^2: ', self.div_surface.get_dist(curr_face) ** 2)
        #print ('normalized rc_vec: ', rotd_math.normalize(rc_vec))
        #print ('two multiplied: ', self.div_surface.get_dist(curr_face) ** 2 *rotd_math.normalize(rc_vec))
        #print ('rc_vec after calc, ', rc_vec)
        self.weight = lfactor * self.div_surface.get_dist(curr_face) ** 2 * \
                      rotd_math.normalize(rc_vec)
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

    def labframe_pivot_point(self, frag_index, com_vec, lf_pivot, lfactor):
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
        # above line is equivalent to:
        # frag_molecule_types = []
        # for f in self.fragments:
        #     frag_molecule_types.append(f.molecule_type)
        there_is_slab = np.any(frag_molecule_types) #bool
        if there_is_slab: 
            #print ('there is slab')
            if frag.molecule_type == MolType.MONOATOMIC:
                lf_pivot = np.zeros(3)
                #print ("New, MONOATOM labframe pivot point: np.zeros(3)", lf_pivot)
                return [SampTag.SAMP_SUCCESS, lfactor]

            elif frag.molecule_type == MolType.NONLINEAR:
                # reorient the pivot point according to the fragment reorientation
                #during set_molframe_positions()
                lf_temp = np.dot(self.div_surface.get_pivot_point(frag_index, face), frag.frag_array['orig_mfo']) #orig_mfo = evecs

                # return pivot_point_vector in molecule frame * rotation matrix
                #np.dot(mf_vector, self.get_rotation_matrix()).reshape(3,)
                lf_temp = frag.mf2lf(lf_temp)
                for i in range(0, len(lf_pivot)):
                    lf_pivot[i] = lf_temp[i]
                
                return [SampTag.SAMP_SUCCESS, lfactor]

            elif frag.molecule_type == MolType.LINEAR: # For slab case, we should consider it as a spherical.
                # reorient the pivot point according to the fragment reorientation

                #This is original linear.. /fragment/linear.py original 과 밑에 check_sample 까지 한번에 바꿔야함 #1
                lf_temp = frag.get_ang_pos() * self.div_surface.get_pivot_point(frag_index, face)[0]
                for i in range(0, len(lf_temp)):
                    lf_pivot[i] = lf_temp[i]
                

                # My test 
                # lf_temp = np.dot(self.div_surface.get_pivot_point(frag_index, face), frag.frag_array['orig_mfo']) #orig_mfo = evecs
                # lf_temp = frag.mf2lf(lf_temp)
                # for i in range(0, len(lf_pivot)):
                #     lf_pivot[i] = lf_temp[i]

                return [SampTag.SAMP_SUCCESS, lfactor]
            
    #######################################################################################################
    # Test for SLAB
            elif frag.molecule_type == MolType.SLAB:
                # 바꿔야하는데...
                top_slab = Atoms(self.configuration.get_chemical_symbols()[-9:],
                            self.configuration.get_positions()[-9:]) 
                #print ('top_slab',top_slab)
                top_slab_com = top_slab.get_center_of_mass()
                #print ('lab_pos slab: ',frag.get_labframe_positions())

                lf_temp = np.dot(self.div_surface.get_pivot_point(frag_index, face), frag.frag_array['orig_mfo']) #orig_mfo = evecs
                #print ('slab pivot', self.div_surface.get_pivot_point(frag_index, face))
                #print ('slab orig_mfo: ', frag.frag_array['orig_mfo']) #[1,1,1] diagonal matrix
                # lf_temp = frag.mf2lf(self.div_surface.get_pivot_point(frag_index, face)) # get_pivot_point의 경우 내가 example.py 에서 준거로 결정됨.
                lf_temp = frag.mf2lf(lf_temp) 
                for i in range(0, len(lf_temp)):
                    lf_pivot[i] = lf_temp[i]
                
                #print ('lf_pivot of slab',lf_pivot) # 랜덤값이 나오고있음.

                
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
            print ("This is printed since there is no slab and this is gas phase reaction")
            if frag.molecule_type == MolType.MONOATOMIC:
                lf_pivot = np.zeros(3)
                return [SampTag.SAMP_SUCCESS, lfactor]

            elif frag.molecule_type == MolType.NONLINEAR:
                lf_temp = np.dot(self.div_surface.get_pivot_point(frag_index, face), frag.frag_array['orig_mfo']) #orig_mfo = evecs
                lf_temp = frag.mf2lf(lf_temp)
                for i in range(0, len(lf_pivot)):
                    lf_pivot[i] = lf_temp[i]
                
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
        there_is_slab = np.any(frag_molecule_types) #bool
        if all(frag.molecule_type == MolType.LINEAR for frag in self.fragments):
            for i in range(0, 3):
                lf_com[i] = self.get_pivot_point(0, face)[0] * \
                    self.fragments[0].get_ang_pos()[i] - \
                    self.get_pivot_point(1, face)[0] * \
                    self.fragemnts[0].get_ang_pos()[i]
        else:                 
            if there_is_slab: 
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
                            #print ('i-==1 lf_com ', lf_com) # 항상 이게 프린트됨
                        else:
                            lf_com += lf_pivot_point
                            #print ('else lf_com ', lf_com)

                    elif frag.molecule_type == MolType.LINEAR:                     
                        # original /fragment/linear.py original 과 위에 labframe_pivot 까지 한번에 바꿔야함 #1
                        if i == 1: # 여기 x 좌표 가져옴 0으로
                            lf_com -= frag.get_ang_pos() * mf_pivot_point[0]
                            #print ('mffffffff') # 당연히 안나옴 현재 i==1 은 surface
                        else:
                            lf_com -= frag.get_ang_pos() * mf_pivot_point[0]
                            #print ('mf_pivot[0]', mf_pivot_point[0]) #처음 copt.py 에서 내가 준 값
                            #print ('what is minus? ', frag.get_ang_pos() * mf_pivot_point[0])

                        # My test
                        # lf_pivot_point = frag.mf2lf(mf_pivot_point)
                        # if i == 1: # 여기 x 좌표 가져옴 0으로
                        #     lf_com -= lf_pivot_point
                        # else:
                        #     lf_com -= lf_pivot_point

            else: # 슬랩이 없다면 여기로옴
                print ('there is no slab this should not be coming out')
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
            # now calculate the distance between the two pivot point
            if np.linalg.norm(lf_com) < self.div_surface.get_dist(face):
                #print('fail')
                return SampTag.SAMP_FACE_OUT
            else:
                #print ('at least succeded')
                #print ('face dist',  self.div_surface.get_dist(face)) # 내가 처음에 준 distance 값
                return SampTag.SAMP_SUCCESS            