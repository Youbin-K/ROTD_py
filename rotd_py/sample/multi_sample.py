import numpy as np
from rotd_py.sample.sample import Sample, preprocess
import rotd_py.rotd_math as rotd_math
from rotd_py.system import MolType, SampTag
from ase import Atoms, atoms
from ase.constraints import FixAtoms
from ase.constraints import FixBondLength
from ase.io.trajectory import Trajectory


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
        mfactor[0] = self.fragments[1].get_total_mass()/total_mass #시작은 이거로 하면서 결국 fragment로 들어가서 싹다 부르고 나올듯
        mfactor[1] = -self.fragments[0].get_total_mass()/total_mass
        emass = self.fragments[0].get_total_mass() * mfactor[0] # This is reduced mass

        # random orient a vector between the center of mass of two fragments
        com_vec = rotd_math.random_orient(3)  # 1st step
        #print ("com_vec", com_vec) # 3 array

        # random rotate each fragment based on random generated solid angle.
        
        ### 무조건 if 랑 else가 둘다 돌고있음
 
        for frag in self.fragments:
            if frag.molecule_type == MolType.SLAB:
                #print ("hm", frag)
                
                orient = rotd_math.random_orient(frag.get_ang_size()) 
                frag.set_ang_pos(orient)
                #frag.set_rotation_matrix()
                #print ("2222")
                #c = FixAtoms(indices = [atom.index for atom in frag if atom.symbol =='Pt'])
                #frag.set_constraint(c)
                
                #print ("multisample frag", frag) # Check whether nonlinear, linear & ch3 & pbc
                #print ("multisample orient", orient) # 4 array
                #print ("The position of angle set by ang_pos: ", frag.set_ang_pos(orient))
                #print ("rotation matrix: ", frag.set_rotation_matrix())
                #print ("iiiiii", i)

                #traj = Trajectory('slab.traj', 'a', frag)
                #traj.write()
                #traj.close()
            else:    
                orient = rotd_math.random_orient(frag.get_ang_size())
                frag.set_ang_pos(orient)
                frag.set_rotation_matrix()
                #print ("else")
                #print ("else frag", frag)
                #print ("iiiiii", i)
                #frag.translate([0.0, 0.0, 100.0])

                #traj2 = Trajectory('linear.traj', 'a', frag)
                #traj2.write()
                #traj2.close()

        #print ("Multisample #2 ")
        """

        for frag in self.fragments:
            orient = rotd_math.random_orient(frag.get_ang_size())
            frag.set_ang_pos(orient)
            frag.set_rotation_matrix()
            #print ("The position of angle set by ang_pos: ", frag.set_ang_pos(orient))
            #print ("GETT", frag.get_ang_pos(orient))        
        """ 
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
                #print ("lf pivot [i][:] = ", lf_pivot[i][:])
                #print ("aaaa") # Only running when all are linear.
                #print ("Whats happening") All rows change corresdpondent to the i
        else:
            for i in range(0, len(self.fragments)):
                sample_info, lfactor = self.labframe_pivot_point(
                    i, com_vec, lf_pivot[i], lfactor) # lfactor 두번 씌어지나??
                #print ("lfactor in the for loop: ", lfactor) This is alwyas 1

                if sample_info == SampTag.SAMP_FACE_OUT:
                    return SampTag.SAMP_FACE_OUT
       
        #print ("0 Pivot: ", lf_pivot[0])
        #print ("1 Pivot: ", lf_pivot[1])
        #now set the reaction coordinate vector
        lf_com = com_vec * self.div_surface.get_dist(curr_face) - lf_pivot[0] + lf_pivot[1]
        new_positions = []
        #print ("0 Pivot: ", lf_pivot[0])
        #print ("1 Pivot: ", lf_pivot[1])
        #print ("COMM", lf_com) ##

        # update the fragments' lab frame position
        # and get the configuration as a whole

        # 여기에 if 넣으면 슬랩이fix 안해도 고정될수도?
        
        for i in range(0, 2):
            # update COM positions
            self.fragments[i].set_labframe_com(lf_com * mfactor[i])

            # set up the final laboratory frame of two fragment
            self.fragments[i].set_labframe_positions()
            
            for pos in self.fragments[i].get_labframe_positions():
                new_positions.append(pos)
        
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

        #print ("Multisample Checking 3")
        #if frag.molecule_type == MolType.SLAB and self.if_slab_and_molecule_too_close():
        #    print ("Multisample_test")
        #    return SampTag.SAMP_ATOMS_CLOSE

        # check the distance between pivot points of other faces
        for face in range(0, self.div_surface.get_num_faces()):
            if face == curr_face:
                continue
            else:
                if self.check_sample(face, lf_com.copy()) == SampTag.SAMP_FACE_OUT:
                    return SampTag.SAMP_FACE_OUT

        # until now, the sampling is valid, set up the configuration
        new_positions = np.array(new_positions)  # in units of Bohr
        # convert back to angstrom
        new_positions *= rotd_math.Bohr # in units of Ang
        #TEMP: Move C and O by 20 angstrom in z-direction
        if frag.molecule_type == MolType.SLAB:
            #print ("testing") #works
            new_positions[0][2] += 20.
            new_positions[1][2] += 20.
        

        #Check the distance of slab and molecule
        """
        if frag.molecule_type == MolType.SLAB and self.if_slab_and_molecule_too_close():
            print ("distance test working")
            return SampTag.SAMP_ATOMS_CLOSE
        """

        # in order to use ASE, convert back to angstrom
        self.configuration.set_positions(new_positions)
        """ 
        if frag.molecule_type == MolType.SLAB and self.if_slab_and_molecule_too_far():
            print ("too far")
            return SampTag.SAMP_ATOMS_CLOSE

        if frag.molecule_type == MolType.SLAB and self.if_slab_and_molecule_too_close():
            print ("too close")
            return SampTag.SAMP_ATOMS_CLOSE

        #print ("I WANT THAT WAY")
        """

        """
        if frag.molecule_type == MolType.SLAB and self.check_molecule_x_coordinate():
            print ("X processing")
            return SampTag.SAMP_ATOMS_CLOSE
        
        
        if frag.molecule_type == MolType.SLAB and self.check_molecule_y_coordinate():
            print ("Y processing")
            return SampTag.SAMP_ATOMS_CLOSE
        """

        if frag.molecule_type == MolType.SLAB and self.check_molecule_z_coordinate():
            #print ("Low molecule removal")
            return SampTag.SAMP_ATOMS_CLOSE
        

        if frag.molecule_type == MolType.SLAB and self.check_in_rhombus():
            #print ("HATE RHOMBUS")
            return SampTag.SAMP_ATOMS_CLOSE

        # calculate the kinematic weight (Phi in the equation) << I'm not sure
        rc_vec = com_vec.copy()  # orbital coordinates (reaction coordinates)
        #print ("rc_vec initial", rc_vec)
        rc_vec /= np.sqrt(emass)  # This means  a = a/b (i.e. rc_vec = rc_vec/np.sqrt(emass) emass = reduced mass
        #print ("rc_vec with mass", rc_vec)
        # internal coordinates
        for index, frag in enumerate(self.fragments):
            if frag.molecule_type == MolType.MONOATOMIC:
                continue
            elif frag.molecule_type == MolType.SLAB:
                continue
            elif frag.molecule_type == MolType.LINEAR:
                if index == 0:
                    rc_vec = np.append(rc_vec, np.cross(lf_pivot[index, :], com_vec)
                                       / np.sqrt(frag.get_inertia_moments()))
                else:
                    rc_vec = np.append(rc_vec, np.cross(com_vec, lf_pivot[index, :])
                                       / np.sqrt(frag.get_inertia_moments()))
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

        self.weight = lfactor * self.div_surface.get_dist(curr_face) ** 2 * \
                      rotd_math.normalize(rc_vec)
        #print ("self.weight", self.weight) This is real weight. Not always 1
        
        #print ("dist_cur_face: ", self.div_surface.get_dist(curr_face)) # This is the value given in the input file
        
        #print ("lfactor in weight: ", lfactor) 

        test = rotd_math.normalize(rc_vec)
        #print ("normalized rc_vec ", test) # returns 1 or 0.999999

        #try printing weight!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # & also lfactor
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

        frag = self.fragments[frag_index] #fragment.index??
        temp_com = com_vec.copy()  # in case modify the original com_vec
        face = self.div_surface.get_curr_face()

        if frag.molecule_type == MolType.MONOATOMIC:
            lf_pivot = np.zeros(3)
            #print ("New, MONOATOM labframe pivot point: np.zeros(3)", lf_pivot)
            return [SampTag.SAMP_SUCCESS, lfactor]


#######################################################################################################
# Test for SLAB
        elif frag.molecule_type == MolType.SLAB:
            #lf_pivot = np.zeros(3)
            lf_temp = frag.mf2lf(self.div_surface.get_pivot_point(frag_index, face))
            for i in range(0, len(lf_pivot)):
                lf_pivot[i] = lf_temp[i]
          
                #print ("Slab lf_pivot: ", lf_pivot[i])
            return [SampTag.SAMP_SUCCESS, lfactor]

        elif frag.molecule_type == MolType.NONLINEAR:
            # return pivot_point_vector in molecule frame * rotation matrix
            lf_temp = frag.mf2lf(self.div_surface.get_pivot_point(frag_index, face))
            for i in range(0, len(lf_pivot)):
                lf_pivot[i] = lf_temp[i]
                #print ("Nonlinear pivot: ", lf_pivot[i])
            
            #print ("return of nonlinear",SampTag.SAMP_SUCCESS, lfactor) always 1 because it is returned
            return [SampTag.SAMP_SUCCESS, lfactor]

        elif frag.molecule_type == MolType.LINEAR:
            if self.div_surface.get_pivot_point(frag_index, face)[1] > 0.0:  # toroidal surface
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

                return [SampTag.SAMP_SUCCESS, lfactor]

            else:  # spherical surface
                lf_temp = frag.get_ang_pos() * self.div_surface.get_pivot_point(frag_index, face)[0]
                for i in range(0, len(lf_temp)):
                    lf_pivot[i] = lf_temp[i]
                    #print ("spherical pivot", lf_pivot[i])
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
        if all(frag.molecule_type == MolType.LINEAR for frag in self.fragments):
            for i in range(0, 3):
                lf_com[i] = self.get_pivot_point(0, face)[0] * \
                    self.fragments[0].get_ang_pos()[i] - \
                    self.get_pivot_point(1, face)[0] * \
                    self.fragemnts[0].get_ang_pos()[i]
        else:
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
            return SampTag.SAMP_FACE_OUT
        else:
            return SampTag.SAMP_SUCCESS
