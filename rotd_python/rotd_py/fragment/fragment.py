from ase.atoms import Atoms
import rotd_py.rotd_math as rotd_math
import numpy as np
from abc import ABCMeta, abstractmethod
from rotd_py.system import MolType


class Fragment(Atoms):
    """Fragment Class is used for describing the fragment
    Fragment could be atom, molecule or even a slab.
    This is a class which inherits from the atoms object in Atomistic Simulation Environement (ASE).
    It can be initialized by the element and position of the molecule,
    and all other information related to the molecule can be either initialized
    or calculated. Users are referred to see the "Atoms" in ASE for more details.

    uniqe parameters for rotd_py:
    mol_type: the geometric information of the fragment, it could be NONATOMIC,
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
                 constraint=None,
                 calculator=None,
                 info=None):

        Atoms.__init__(self, symbols, positions, numbers,
                       tags, momenta, masses, magmoms, charges,
                       scaled_positions, cell, pbc, celldisp,
                       constraint, calculator, info)
        self.frag_array = {}

        self.set_molframe_positions()
        self.set_molecule_type()
        self.init_dynamic_variable()

    @property
    def molecule_type(self):
        """Get the molecule type"""
        return self.frag_array['mol_type']

    @abstractmethod
    def set_molecule_type(self):
        """
        Input: MolType
        Set the type of the molecule and the molecule-type related parameters.
        """

    def get_total_mass(self):
        """Return the total mass of the molecule, in atomic units """
        return sum(self.get_masses()) * rotd_math.mp

    def get_stat_sum(self):
        """Return the parameters for calculating number of states. """
        return self.frag_array['stat_sum']

    def get_molframe_positions(self):
        """The molecule frame position of the molecule,
        set at the initialization and NEVER change during the run

        """
        return self.frag_array['mol_frame_positions'].copy()

    def get_relative_positions(self):
        """Return the positions that are relative to center of mass. """
        com = self.get_center_of_mass()
        positions = self.get_positions()
        positions -= com

        return positions

    def set_molframe_positions(self):
        """Method used to set up the matrix. Convert the input Cartesian
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

        I = np.array([[I11, I12, I13, ],
                      [I12, I22, I23],
                      [I13, I23, I33]])

        trace = I.trace()
        I = I - np.array([[trace, 0, 0],
                          [0, trace, 0],
                          [0, 0, trace]])

        evals, evecs = np.linalg.eigh(I)

        self.frag_array['inertia_moments'] = evals
        self.frag_array['orig_mfo'] = evecs
        self.frag_array['mol_frame_positions'] = np.dot(rel_pos, evecs)

    # all the following are dynamic variable related to on-the-fly sampling
    def init_dynamic_variable(self):
        """Initialize the dynamic variable related to rotating the molecule. """
        self.frag_array['lab_frame_positions'] = np.zeros((self.get_number_of_atoms(), 3))
        self.frag_array['lab_frame_COM'] = np.zeros(3)
        self.frag_array['orient_vector'] = np.zeros(self.get_ang_size())
        self.frag_array['mfo'] = np.zeros((3, 3))

    def set_ang_pos(self, orient_vec):
        """Set up the rotation vector (random generated) for the molecule.

        """
        if len(orient_vec) != len(self.frag_array['orient_vector']):
            raise ValueError("Orientation vector dimension does not fit")
        for i in range(0, len(orient_vec)):
            self.frag_array['orient_vector'][i] = orient_vec[i]

    def set_labframe_com(self, new_com):
        """Set up the center of mass positions of the fragment in the lab frame
        after getting the rotation vector between the two fragments.

        """
        if any(item is None for item in new_com) or len(new_com) != 3:
            raise ValueError('Wrong dimension of position')
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

    # The following functions are related to the type of molecule, 
    # so they are defined separately under each molecular type.
    @abstractmethod
    def set_rotation_matrix(self):
        """Set up the mfo (molecule frame rotation)
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
        """Convert the input vector(lab frame vector) to molecule frame.

        """

        pass

    @abstractmethod
    def mf2lf(self, mf_vector):
        """Convert the input vector(molecule frame vector) to laboratory frame.

        """
        pass

    @abstractmethod
    def get_labframe_imm(self, i, j):
        """return the [i][j] value in the moments of inertia matrix in lab frame value
        """
        pass

    @abstractmethod
    def get_inertia_moments(self):
        """Return the moments of inertia of molecule, same to the
        atoms.get_moments_of_inertia() function, with different units.
        """
