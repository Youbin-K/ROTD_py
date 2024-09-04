from ase.atoms import Atoms
import rotd_py.rotd_math as rotd_math
import numpy as np
from abc import ABCMeta, abstractmethod
from rotd_py.system import MolType
from rotd_py.sample.sample import Sample, preprocess


class Fragment(Atoms):
    """Fragment Class used for describing the fragment
    Fragment could be atom, molecule or even a slab.
    This is a class inherits from the atoms object in ASE.
    It can be initialized by the element and position of them of the molecule,
    and all other information related to the molecule can be either initialized
    or calculated. Users are referred to the Atoms in ASE for more details.

    uniqe parameters for rotd_py:
    mol_type: the geometry information of the fragment, it could be NONATOMIC,
             LINEAR, and NONLINEAR
    mol_frame_positions: the molecule positions used as a reference for
                        rotating the molecule, with the unit of Bohr
    orig_mfo: initial molecular frame orientation matrix
    ang_size: dimension of the angular orientation vector related to the type
                of Atoms
    inertia_moments: moments of inertia. (different units with in ASE)

    during the sampling:
    orient_vector: the rotation vector for each rotation
    lab_frame_positions: the real position of the fragment in the sample
                        with units of Bohr
    lab_frame_COM: the center of mass of the fragment in the Lab frame
    mfo: the molecule frame orientation matrix during the sampling

    """

    def __init__(self, symbols=None,
                 positions=None, numbers=None,
                 tags=None, momenta=None, masses=None,
                 magmoms=None, charges=None,
                 scaled_positions=None,
                 cell=None, pbc=None, celldisp=None,
                 #cell=None, pbc=None, celldisp=np.array([[30,30,30]]),
                 constraint=None,
                 calculator=None,
                 info=None):

        Atoms.__init__(self, symbols, positions, numbers,
                       tags, momenta, masses, magmoms, charges,
                       scaled_positions, cell, pbc, celldisp,
                       constraint, calculator, info)
        self.frag_array = {}
        #print ("Hi I'm frag_array", self.frag_array)

        self.set_molframe_positions()
        # self.set_molframe_positions_with_slab()

        self.set_molecule_type()
        self.init_dynamic_variable()

    @property
    def molecule_type(self):
        """Get the molecule type"""
        #print ("The molecule type is:", self.frag_array['mol_type'])
        return self.frag_array['mol_type']

    @abstractmethod
    def set_molecule_type(self):
        """
        Input: MolType
        Set the type of the molecule and the molecule-type related parameters.
        """

    def get_total_mass(self):
        """Return the total masses of the molecule, in atomic units """
        #print ("BEEP! getting total_mass")
        return sum(self.get_masses()) * rotd_math.mp

    def get_stat_sum(self):
        """Return the parameters for calculating number of states. """
        #print ("Hi I'm get_stat_sum")
        return self.frag_array['stat_sum']

    def get_molframe_positions(self):
        """The molecule frame position of the molecule
        set at the initialization and never change during the run

        """
        #print ("get mf pos from frag.py")
        #print ("99999") 
        return self.frag_array['mol_frame_positions'].copy()

    def get_relative_positions(self):
        """Return the positions that relative to center of mass. """
        #print ("22222")
        # Called out same time as below function, set_molframe_position
        #print ("getting relative position")
        com = self.get_center_of_mass()
        # print ('com', com) # Initial input value COM
        #print ('slab COM', com)
        #print ("getting COM from def (get_rel_pos) in frag.py")
        # get_positions 할때, 내가 정해준 포지션으로 들어옴
        positions = self.get_positions() 
        #print ('positions', positions)
        #print ('get_relative_position, get_positions: ', positions)
        positions -= com
        #print ('minus com ', positions)
        return positions

    def get_original_positions(self):
        """Return the positions that relative to center of mass. """
        positions = self.get_positions() 
        return positions

    def set_molframe_positions(self): # In molframe, inertia tensor is diagonal and the kinetic energy does not depend on the Coordinate Q
        #This is called out only in the beginning.
        """Method used to set up the matrix convert the input Cartesian
        coordinates to molecular frame coordinates and the molecule frame
        positions.

        
        """
        rel_pos = self.get_relative_positions() / rotd_math.Bohr
        masses = self.get_masses() * rotd_math.mp
        I11 = I22 = I33 = I12 = I13 = I23 = 0.0
        for i in range(len(rel_pos)):
            x, y, z = rel_pos[i]
            m = masses[i]

            I11 += -m * (x ** 2)
            I22 += -m * (y ** 2)
            I33 += -m * (z ** 2)
            I12 += -m * x * y
            I13 += -m * x * z
            I23 += -m * y * z

            #print('i,m,x,y,z',i,m,x,y,z)

        I = np.array([[I11, I12, I13, ],
                      [I12, I22, I23],
                      [I13, I23, I33]])

        trace = I.trace()
        I = I - np.array([[trace, 0, 0],
                          [0, trace, 0],
                          [0, 0, trace]])
        #print ('I: ', I) # Diagonal matrix
        evals, evecs = np.linalg.eigh(I)
        
        self.frag_array['inertia_moments'] = evals # used for get_inertia_moments, which is used for rc_vec calculation # Principal moments of inertia
        #print ('original inertia evals: ', evals) # [-3.59920219e-55  4.93132289e+04  4.93132289e+04] for CO on Pt position # Original =  [    0.         52360.26894516 52360.26894516]
        self.frag_array['orig_mfo'] = evecs # Used to set labframe_pivot for nonlinear # Principal axes for original, Identity matrix
        # test_value = np.array([[0, 1, 0], 
        #                        [1, 0, 0], 
        #                        [0, 0, 1]])

        # self.frag_array['orig_mfo'] = test_value
        # print ('evecs ', evecs) #[[-1.00000000e+00  2.70160221e-30  0.00000000e+00] [-6.55234780e-31 -2.42535625e-01 -9.70142500e-01] [-2.62093912e-30 -9.70142500e-01  2.42535625e-01]] for CO on Pt position
        #print ('evecs', self.frag_array['orig_mfo'])
        #print ('rel_pos ', rel_pos) # relative position, original position - center of mass
        # Used for get_molframe_positions, which is used for set_labframe_position
        # 분자의 configuration 에 의해 정해진 rotation matrix (evecs)
        #print ('rel_pos for all', rel_pos)
        
        self.frag_array['mol_frame_positions'] = np.dot(rel_pos, evecs) # matrix X matrix 여서 지금은 그냥 곱셈이나 마찬가지

    def init_dynamic_variable(self):
        """Initialize the dynamic variable related to rotating the molecule. """
        self.frag_array['lab_frame_positions'] = np.zeros((self.get_number_of_atoms(), 3)) # for ase 3.13.0
        # self.frag_array['lab_frame_positions'] = np.zeros((self.get_global_number_of_atoms(), 3)) # for ase 3.19.1

        self.frag_array['testing_lab_frame_positions'] = np.zeros((self.get_number_of_atoms(), 3)) # for ase 3.13.0
        self.frag_array['lab_frame_COM'] = np.zeros(3)
        self.frag_array['orient_vector'] = np.zeros(self.get_ang_size())
        self.frag_array['mfo'] = np.zeros((3, 3))
        #print ("init_dynamic_variable")

    def set_ang_pos(self, orient_vec):
        """Set up the rotation vector (random generated) for the molecule.

        """
        #print("I'm setting the angle_position")

        if len(orient_vec) != len(self.frag_array['orient_vector']):
            raise ValueError("Orientation vector dimension does not fit")
        for i in range(0, len(orient_vec)):
            self.frag_array['orient_vector'][i] = orient_vec[i]
        

    def set_labframe_com(self, new_com): #여기여기. 어쩌면 바꿔야할지도 아닐지도
        """Set up the center of mass positions of the fragment in the lab frame
        after getting the rotation vector between the two fragments.

        """
        #print ("Set the labframe_COM")
        if any(item is None for item in new_com) or len(new_com) != 3:
            raise ValueError('Wrong dimension of position')
        if self.molecule_type == MolType.SLAB:
            # test_com = self.get_center_of_mass() / rotd_math.Bohr
            # for i in range(0,len(test_com)):
            #     self.frag_array['lab_frame_COM'][i] = test_com[i]
            for i in range(0,len(new_com)):
                self.frag_array['lab_frame_COM'][i] = new_com[i]
        else: 
            for i in range(0, len(new_com)):
                self.frag_array['lab_frame_COM'][i] = new_com[i]

    def get_ang_pos(self):
        return self.frag_array['orient_vector'].copy()

    def get_ang_size(self):
        return self.frag_array['ang_size']

    def get_rotation_matrix(self):
        return self.frag_array['mfo'].copy()
    
    def get_labframe_com(self):
        return self.frag_array['lab_frame_COM'].copy()

    def get_labframe_positions(self):
        return self.frag_array['lab_frame_positions'].copy()
    

    # The following functions are related to the type of molecule, so they are
    # defined separately under each molecular type.
    @abstractmethod
    def set_rotation_matrix(self):
        """Set up the mfo (molecule frame rotation )
        matrix after generating the rotation vector for the fragment

        """
        pass

    @abstractmethod
    def set_labframe_positions(self):
        """Update the lab frame positions of fragment after rotation.
            labframe_pos = lab_com + np.dot(molframe_pos, rotation_matrix)

        """
        pass

    @abstractmethod
    def lf2mf(self, lf_vector):
        """Convert the input vector to molecule frame.

        """

        pass

    @abstractmethod
    def mf2lf(self, mf_vector):
        """Convert the input vector to laboratory frame.

        """
        pass

    @abstractmethod
    def get_labframe_imm(self, i, j):
        """return the [i][j] value in the inertia moment matrix in lab frame
        """
        pass

    @abstractmethod
    def get_inertia_moments(self):
        """Return the moments of inertia of molecule, same to the
        atoms.get_moments_of_inertia() function, with different units.
        """

    @abstractmethod
    def get_inertia_moments_for_surface(self):
        """Return the moments of inertia of molecule, same to the
        atoms.get_moments_of_inertia() function, with different units.
        """