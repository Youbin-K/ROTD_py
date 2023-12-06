from abc import ABCMeta, abstractmethod

import numpy as np
from mpi4py import MPI
from ase.atoms import Atoms
#from amp import Amp

import rotd_py
from rotd_py.system import MolType
import rotd_py.rotd_math as rotd_math
from rotd_py.molpro.molpro import Molpro
from scipy.interpolate import make_interp_spline


class Sample(object):
    """Class used for generating random configuration of given fragments and
    dividing surface.

    Parameters
    ----------
    fragments : List of Fragment
        The fragments in the system
    dividing_surface : Surface
        Current calculated surface considering all facets
    calculator: Calculator (ASE)
        Ase object calculator for calculating the potential
    min_fragments_distance : float
        Too close atoms distance threshold.
    Attributes
    ----------
    dividing_surface : Surface
    close_dist : float
    configuration : Atoms objects which will be used to store the combination of
                   two fragments (the sampled configuration) in Angstrom.
    dof_num : degree of freedom of the reaction system.
    weight : the geometry weight for the generated configuration.
    inf_energy: the energy of the configuration at infinite separation in Hartree
    scan_sample: 1D energies (kcal) along the dissociation computed at the sampling level
    scan_trust: 1D energies (kcal) along the dissociation computed at the highest affordable level
    """

    def __init__(self, name=None, fragments=None, dividing_surface=None,
                 min_fragments_distance=1.5, inf_energy=0.0, energy_size=1,
                 r_sample=None, e_sample=None, r_trust=None, e_trust=None,
                 scan_ref=None):
        __metaclass__ = ABCMeta
        self.name=name
        # list of fragments

        self.fragments = fragments
        if fragments is None or len(fragments) < 2:
            raise ValueError("For sample class, \
                            at least two None fragments are needed")
        self.div_surface = dividing_surface
        self.close_dist = min_fragments_distance
        self.weight = 0.0
        self.inf_energy = inf_energy*rotd_math.Hartree #Convert energy from Hartree to eV
        # correction potential parameters
        self.set_scan_ref(scan_ref)  # atom number of pivots, list of lists
        self._1d_correction = self.get_1d_correction(r_sample, e_sample, r_trust, e_trust)
        self.energy_size = energy_size
        self.energies = np.array([0.] * self.energy_size)
        self.ini_configuration()
        self.set_dof()

    def get_1d_correction(self, r_sample, e_sample, r_trust, e_trust):
        """Function that takes scan relative energies in Kcal
        and returns a spline corresponding to the 1d correction in eV."""
        if r_sample == None or not isinstance(r_sample, list):
            r_sample = [0.0,1.,2.,3.,4.]
        if r_trust == None or not isinstance(r_trust, list):
            r_trust = [0.0,1.,2.,3.,4.]
        if e_sample == None or not isinstance(e_sample, list):
            e_sample = [0., 0., 0., 0., 0.]
        if e_trust == None or not isinstance(e_trust, list):
            e_trust = [0., 0., 0., 0., 0.]
        
        x_spln_1d_correction = np.arange(min(r_sample + r_trust), max(r_sample + r_trust), 0.01)
        
        spln_sample = make_interp_spline(r_sample, np.asarray(e_sample)*rotd_math.Kcal/rotd_math.Hartree)
        spln_trust = make_interp_spline(r_trust, np.asarray(e_trust)*rotd_math.Kcal/rotd_math.Hartree)

        y_spln_sample = spln_sample(x_spln_1d_correction)
        y_spln_trust = spln_trust(x_spln_1d_correction)

        y_1d_correction = np.subtract(np.asarray(y_spln_sample), np.asarray(y_spln_trust))

        _1d_correction = make_interp_spline(x_spln_1d_correction, y_1d_correction)

        return _1d_correction
    
    def energy_correction(self):
        distance = np.inf
        for scr in self.scan_ref:
            distance = min(distance, np.absolute(np.linalg.norm(self.configuration.positions[scr[0]] -\
                                                  self.configuration.positions[scr[1]])))
        e = self._1d_correction(distance)
        return e

    def set_scan_ref(self, scan_ref):
        self.scan_ref = []
        for scr in scan_ref:
            self.scan_ref.append([scr[0], scr[1]+len(self.fragments[0].positions)])

    def get_dividing_surface(self):
        return self.div_surface

    def set_dividing_surface(self, surface):
        self.div_surface = surface

    def set_dof(self):
        """Set up the degree of freedom of the system for calculation

        """
        self.dof_num = 3
        for frag in self.fragments:
            if frag.molecule_type == MolType.MONOATOMIC:
                continue
            elif frag.molecule_type == MolType.LINEAR:
                self.dof_num += 2
            elif frag.molecule_type == MolType.NONLINEAR:
                self.dof_num += 3

    def ini_configuration(self):
        """Initialize the total system with ASE class Atoms

        """
        new_atoms = []
        new_positions = []
        for frag in self.fragments:
            new_atoms += frag.get_chemical_symbols()
            for pos in frag.get_positions():
                new_positions.append(pos)
        self.configuration = Atoms(new_atoms, new_positions)

    def get_dof(self):
        """return degree of freedom

        """
        return self.dof_num

    def get_canonical_factor(self):
        """return canonical factor used in the flux calculation

        """
        return 2.0 * np.sqrt(2.0 * np.pi)

    def get_microcanonical_factor(self):
        """return microcanonical factor used in the flux calculation

        """
        return rotd_math.M_2_SQRTPI * rotd_math.M_SQRT1_2 / 2.0 / \
            rotd_math.gamma_2(self.get_dof() + 1)

    def get_ej_factor(self):
        """return e-j resolved ensemble factor used in the flux calculation

        """
        return rotd_math.M_1_PI / rotd_math.gamma_2(self.get_dof() - 2)

    def get_tot_inertia_moments(self):
        """Return the inertia moments for the sampled configuration

        """

        return self.configuration.get_moments_of_inertia() * \
            rotd_math.mp / rotd_math.Bohr**2

    def get_weight(self):
        return self.weight

    def get_calculator(self):
        return self.calculator

    def if_fragments_too_close(self):
        """Check the distance among atoms between the two fragments.

        Returns
        -------
        type
            Boolean

        """
        lb_pos_0 = self.fragments[0].get_labframe_positions()
        lb_pos_1 = self.fragments[1].get_labframe_positions()
        for i in range(0, len(lb_pos_0)):
            for j in range(0, len(lb_pos_1)):
                dist = np.linalg.norm(lb_pos_0[i] - lb_pos_1[j])
                if dist < self.close_dist:
                    return True
        return False

    @abstractmethod
    def generate_configuration(self):
        """This function is an abstract method for generating random
        configuration based relative to different sample schema.
        1. generate the new configuration.
        2. set the self.weight
        3. set the new positions for self.configuration.

        """
        pass

    def get_energies(self, calculator, face_id=0, flux_id=0):

        # return absolute energy in e
        # This is a temporary fix specific for using Amp calculator as we are not able to
        # either 1) deepcopy the Amp calculator object or 2) using MPI send/receive Amp calculator object
        # Note: this may cause some performance issue as we need to load Amp calculator for each calculation.
        if calculator['code'] == 'amp.amp':
            amp_calc = Amp.load(f'{rotd_py.__path__[0]}/amp.amp')
            self.configuration.set_calculator(amp_calc)
            energy = self.configuration.get_potential_energy()
            self.configuration.set_calculator(None)
        elif calculator['code'] == 'molpro':
            label = f'surf{self.div_surface.surf_id}_face{face_id}_samp{flux_id}'
            mp = Molpro(label, self.configuration, calculator['scratch'],
                        calculator['processors'])
            mp.create_input()
            mp.run()
            energy = mp.read_energy()

        #Energies must be in eV
        self.energies[0] = (energy + self.energy_correction() - self.inf_energy)/rotd_math.Hartree

        return self.energies


# The following code can be helpful if already has some sampled configuration
# and want to do some tests. Otherwise, no need to use the following code.
class Geometry(object):
    def __init__(self, atoms=None, energy=0.0, weight=0.0, pivot_rot=None,
                 frag1_rot=None, frag2_rot=None, tot_im=None):
        self.atoms = atoms
        self.energy = np.array(energy)
        self.weight = weight
        self.tot_im = np.array(tot_im)
        self.pivot_rot = np.array(pivot_rot)
        self.frag1_rot = np.array(frag1_rot)
        self.frag2_rot = np.array(frag2_rot)

    def get_atoms(self):
        return self.atoms

    def get_energy(self):
        return self.energy

    def get_weight(self):
        return self.weight

    def get_pivot_rot(self):
        return self.pivot_rot

    def get_frag1_rot(self):
        return self.frag1_rot

    def get_frag2_rot(self):
        return self.frag2_rot

    def get_tot_im(self):
        return self.tot_im


def preprocess(file_name):
    file_lines = open(file_name).readlines()
    geometries = []
    atom_num = 8
    for i, line in enumerate(file_lines):

        if line.startswith('Geometry:'):
            line_index = i
            positions = None
            if len(line.split()) == 1:
                positions = np.zeros((atom_num, 3))
                for j in range(0, atom_num):
                    positions[j] = map(float, file_lines[i+1+j].split()[1:])
                line_index += atom_num

            energy = float(file_lines[line_index + 1].split()[1])
            weight = float(file_lines[line_index + 2].split()[1])
            tot_im = np.array(map(float, file_lines[line_index + 4].split()[:]))
            pivot_rot = map(float, file_lines[line_index + 6].split()[:])
            frag1_rot = map(float, file_lines[line_index + 8].split())
            frag2_rot = map(float, file_lines[line_index + 10].split())
            atoms = None
            if positions is not None:
                atoms = Atoms('CH3CH3', positions=positions)

            geom = Geometry(atoms=atoms, energy=energy, weight=weight,
                            tot_im=tot_im,
                            frag1_rot=frag1_rot,
                            frag2_rot=frag2_rot, pivot_rot=pivot_rot)
            geometries.append(geom)

    return geometries
