import numpy as np

from rotd_py.system import SampTag, FluxTag
import rotd_py.rotd_math as rotd_math
from rotd_py.flux.fluxbase import FluxBase
import copy
import os

import shutil
min_flux = 1e-99
# CAUTION! all units are atomic unit

import logging


from ase.io.trajectory import Trajectory
min_flux = 1e-99
# CAUTION! all units are atomic unit


class MultiFlux:
    """Manage a flux array for each dividing surface,
    each item in array is the flux calculation for a specific facet

    Parameters
    ----------
    fluxbase: FluxBase, all the info needed for the flux calculation
    num_faces: the number of faces for each dividing surface. This number
    determines the dimension of MultiFlux

    """


    def __init__(self, fluxbase=None, num_faces=1, selected_faces=None, sample=None,
                 calculator=None):


        if fluxbase is None:
            raise ValueError("The flux base cannot be NONE, the information\
                             for flux calculation is needed")
        self.flux_array = [None] * num_faces
        self.num_faces = num_faces
        self.temp_grid = fluxbase.temp_grid
        self.tol = fluxbase.tol()
        self.pot_max = fluxbase.pot_max()
        self.tot_max = fluxbase.tot_max()

        self.selected_faces = selected_faces
        self.energy_size = sample.energy_size
        self.sample = copy.deepcopy(sample)

        self.converged = False


        # initialize each flux to the corespondent facet.
        # convert the energy unit to Hartree
        for i in range(num_faces):
            self.flux_array[i] = Flux(temp_grid=fluxbase.temp_grid * rotd_math.Kelv,
                                      energy_grid=fluxbase.energy_grid * rotd_math.Kelv,

                                      angular_grid=fluxbase.angular_grid,
                                      flux_type=fluxbase.flux_type,
                                      flux_parameter=fluxbase._flux_parameter,
                                      sample=copy.deepcopy(sample),
                                      calculator=calculator
                                      )
            self.flux_array[i].sample.div_surface.set_face(i)

        self.logger = logging.getLogger('rotdpy')

    def check_state(self):
        """check whether the calculation for this dividing surface has converged
        or not. Return the flux type and the face index that need to be feed to
        slave (or should be added to the work queue)

        Returns
        -------
        type
            [FluxTag, int]

        """

        #for i in range(0, self.num_faces):
        for i in self.selected_faces:  # HACKED !!!

            if self.flux_array[i].acct_smp() >= self.flux_array[i].pot_min():
                continue
            else:
                return FluxTag.FLUX_TAG, i

        # estimate the error

        min_val = np.inf
        min_index = 0

        # find the minimum thermal flux among all faces and for each corrected flux
        for t_ind in range(0, len(self.temp_grid)):
            val = np.zeros(self.energy_size)
            # val = 0.
            #for i in range(0, self.num_faces):
            for i in self.selected_faces:  # HACKED !!!
                val += self.flux_array[i].average(t_ind, self.energy_size-1)
            if any(val) < min_val:
                min_val = np.min(val)
                min_index = t_ind
                energy_index = np.where(val==min(val))[0][0]

        # total potential variance
        tot_pot_var = 0.0
        for i in self.selected_faces:  # HACKED !!!
            tot_pot_var += self.flux_array[i].pot_fluctuation(min_index, energy_index)

        # projected number of samplings
        proj_smp_num = 0.0

        if min_val > 1.0e-99:
            proj_smp_num = 10000. * tot_pot_var**2 / min_val**2 / self.tol**2
            if proj_smp_num > self.pot_max:
                proj_smp_num = self.pot_max

        if proj_smp_num > 1.:
            max_smp = -1
            face = 0

            for i in self.selected_faces:  # HACKED !!!
                smp = proj_smp_num * \
                    self.flux_array[i].pot_fluctuation(min_index, energy_index) / tot_pot_var

                if smp <= 1.:
                    smp = -1
                else:
                    smp = 1.0 - self.flux_array[i].acct_smp() / smp
                if smp > max_smp:
                    max_smp = smp
                    face = i
            if max_smp > 0.:
                return [FluxTag.FLUX_TAG, face]

        # estimate the surface area sampling error
        proj_smp_num = 0.
        tot_vol_var = 0.

        #for i in range(0, self.num_faces):
        for i in self.selected_faces:  # HACKED !!!
            tot_vol_var += self.flux_array[i].vol_fluctuation(min_index, energy_index)
        # self.sample.div_surface.vol_var = tot_vol_var


        # projected number of sampling
        if min_val > 1.0e-99:
            proj_smp_num = 10000. * tot_vol_var**2 / min_val**2/self.tol**2
            if proj_smp_num > self.tot_max:
                proj_smp_num = self.tot_max

        if proj_smp_num <= 1.:
            return [FluxTag.STOP_TAG, -1]

        # check whether surface sampling is needed or not
        is_surf_smp = False

        smp_num = np.zeros(self.num_faces)#, dtype=int)
        #for i in range(0, self.num_faces):
        for i in self.selected_faces:  # HACKED !!!
            smp = proj_smp_num * self.flux_array[i].vol_fluctuation(min_index, energy_index)\

                / tot_vol_var - self.flux_array[i].tot_smp()
            if smp > 0:
                is_surf_smp = True
                smp_num[i] = int(smp)
            else:
                smp_num[i] = 0
        if is_surf_smp:
            return [FluxTag.SURF_TAG, smp_num]

        return [FluxTag.STOP_TAG, -1]

    def save_file(self, sid):
        """
        save the calculation results after the current dividing surface get
        converged
        sid is the index of current dividing surface
        """

        current_path = os.getcwd()
        # commented out, because it deleted all other surface_n files
        # try:
        #     shutil.rmtree(current_path + "/" + "output")
        # except FileNotFoundError:
        #     pass
        try:
            os.mkdir(current_path + "/" + "output")
        except FileExistsError:
            pass
        filename = "surface_"+str(sid)+".dat"
        if os.path.exists(filename):
            self.logger.info("The file  %s is already exists, REWRITE it!" % (filename))
        f = open("output" + "/" + filename, 'w')

        symbols = self.flux_array[0].sample.configuration.get_chemical_symbols()
        #for face in range(self.num_faces):
        for face in self.selected_faces:  # HACKED !!!

            f.write("Face: %d\n" % (face))
            curr_flux = self.flux_array[face]
            f.write("Successful sampling: %d \n" % (curr_flux.acct_smp()))
            f.write("Failed sampling: %d \n" % (curr_flux.fail_smp()))
            f.write("Close-atoms sampling: %d \n" % (curr_flux.close_smp()))
            f.write("Out of face sampling: %d \n" % (curr_flux.face_smp()))
            f.write("Dummy sampling: %d \n" % (curr_flux.fake_smp()))
            # here out put unit with kcal/mol

            f.write("Minimum energy: %.5f" % (curr_flux.min_energy[0]/rotd_math.Kcal))
            for corrected_e in range(1, len(curr_flux.min_energy)):
                f.write("   %.5f" % (curr_flux.min_energy[corrected_e]/rotd_math.Kcal))
            f.write("\n")
            f.write("Minimum energy geometry:\n")
            for line in range(0, len(curr_flux.min_geometry[0])):
                f.write("%5s" % (symbols[line]))
                for correction_index in range(len(curr_flux.min_geometry)):
                    item = curr_flux.min_geometry[correction_index][line]
                    f.write(" %16.8f %16.8f %16.8f |" % (item[0], item[1], item[2]))
                f.write("\n")

            # now normalize the calculated results
            normalized_face_flux = curr_flux.normalize()
            # the Canonical:
            f.write("Canonical: \n")
            f.write("%-16s %14s" % ("Temperature (K)", "Uncorrected"))
            for correction_name in normalized_face_flux.sample.corrections.keys():
                f.write(" %14s" % (correction_name))
            f.write("\n")
            for this_temp in range(0, len(normalized_face_flux.temp_grid)):
                line = "%-16.3f  " % (normalized_face_flux.temp_grid[this_temp]/rotd_math.Kelv)
                for x in normalized_face_flux.temp_sum[this_temp]:
                    line += " %14.3e" % (x)
                f.write(line + '\n')
            # the Micro-canonical:
            f.write("Microcanonical: \n")
            f.write("%-16s %14s" % ("Energy (K)", "Uncorrected"))
            for correction_name in normalized_face_flux.sample.corrections.keys():
                f.write(" %14s" % (correction_name))
            f.write("\n")
            for this_energy in range(0, len(normalized_face_flux.energy_grid)):
                line = "%-16.3f" % (normalized_face_flux.energy_grid[this_energy]/rotd_math.Kelv)
                for x in normalized_face_flux.e_sum[this_energy]:
                    line += " %14.3e" % (x)
                f.write(line + '\n')

            f.write("E-J resolved: \n")
            f.write("%-16s %14s" % ("[J, E]", "Uncorrected"))
            for correction_name in normalized_face_flux.sample.corrections.keys():
                f.write(" %14s" % (correction_name))
            f.write("\n")
            for this_energy in range(0, len(normalized_face_flux.energy_grid)):
                for this_J in range(0, len(normalized_face_flux.angular_grid)):
                    line = "%-7.3e %7.3e" % (normalized_face_flux.angular_grid[this_J], normalized_face_flux.energy_grid[this_energy]/rotd_math.Kelv)
                    for x in normalized_face_flux.ej_sum[this_energy, this_J]:
                        line += " %14.3e" % (x)
                    f.write(line + '\n')

        f.close()


class Flux(FluxBase):
    """Class used for flux calculation

    Parameters
    ----------
    energy_size : int
        The size of energy correction.
    sample : Sample
        The sample class which will in charge of generating configurations.


    Attributes
    ----------
    _acct_num : int
        The number of valid sampling.
    _fail_num : int
        The number of failed sampling.
    _fake_num : int
        The number of dummy sampling (sample without executing potentail calculation.)
    _close_num : int
        The number of sampling whose distance of atoms are too close.
    _face_num : int
        The number of face out smapling.

    """

    def __init__(self, temp_grid=None, energy_grid=None,
                 angular_grid=None, sample=None, calculator=None,
                 flux_type='CANONICAL', flux_parameter=None):

        super(Flux, self).__init__(temp_grid=temp_grid,
                                   energy_grid=energy_grid,
                                   angular_grid=angular_grid,
                                   flux_type=flux_type,
                                   flux_parameter=flux_parameter)
        self.sample = sample
        self.calculator = calculator

        self.job_id = None
        self.energy_size = sample.energy_size

        self.set_calculation()
        self.set_vol_num_max()
        self.set_fail_num_max()
        self.min_energy = [float('inf')] * self.energy_size
        self.min_geometry = [None] * self.energy_size

        self.logger = logging.getLogger('rotdpy')

    def set_calculation(self):

        # The input grid used for flux calculation
        # initialize the variable used for canonical flux
        self.temp_sum = np.zeros((len(self.temp_grid), self.energy_size))
        self.temp_var = np.zeros((len(self.temp_grid), self.energy_size))
        self.min_energy = np.zeros(self.energy_size)
        self.min_var = np.zeros(self.energy_size)

        # initialize the variable used for microcanonical flux
        self.e_sum = np.zeros((len(self.energy_grid), self.energy_size))

        self.e_var = np.zeros((len(self.energy_grid), self.energy_size))

        # initialize the variable used for EJ-resolved flux
        self.ej_sum = np.zeros((len(self.energy_grid), len(self.angular_grid),
                                self.energy_size))
        self.ej_var = np.zeros((len(self.energy_grid), len(self.angular_grid),
                                self.energy_size))

    def set_vol_num_max(self, vol_max=1000):
        self._vol_max_num = vol_max

    def set_fail_num_max(self, fail_max=10):
        self._fail_max_num = fail_max

    def get_vol_num_max(self):
        return self._vol_max_num

    def get_fail_num_max(self):
        return self._fail_max_num

    def is_pot(self):
        # test function to see whether can start flux normalization

        if not self.tot_smp():
            raise ValueError("NO Sampling Found!")
        if not self.pot_smp():
            return False
        if not self.acct_smp():
            raise ValueError("NO Accepted Sampling!")
        return True

    def normalize(self):
        """This function is used to normalize everything
        """

        if not self.acct_smp():
            return
        # TODO: figure out why
        normalized_flux = copy.deepcopy(self)

        samp_factor = float(self.pot_smp())/self.tot_smp()/self.acct_smp()

        fluct_factor = 1.0/self.acct_smp() + 1.0/self.tot_smp() - 1.0/self.pot_smp()

        # dealing with the thermal flux in total

        normalized_flux.temp_sum *= samp_factor
        normalized_flux.temp_var[np.where(normalized_flux.temp_sum < min_flux)] = 0.0
        t = normalized_flux.temp_var * samp_factor**2 - normalized_flux.temp_sum**2 * fluct_factor
        normalized_flux.temp_var[np.where(t <= 0.)] = 0.0
        index = np.where(t > 0.)
        normalized_flux.temp_var[index] = np.sqrt(t[index])/normalized_flux.temp_sum[index] * 100.0

        # microcanonical
        normalized_flux.e_sum *= samp_factor
        normalized_flux.e_var[np.where(normalized_flux.e_sum < min_flux)] = 0.0
        t = normalized_flux.e_var * samp_factor**2 - normalized_flux.e_sum**2 * fluct_factor
        normalized_flux.e_var[np.where(t <= 0.)] = 0.0
        index = np.where(t > 0.)
        normalized_flux.e_var[index] = np.sqrt(t[index])/normalized_flux.e_sum[index] * 100.0

        # e-j resolved
        normalized_flux.ej_sum *= samp_factor
        normalized_flux.ej_var[np.where(normalized_flux.ej_sum < min_flux)] = 0.0
        t = normalized_flux.ej_var * samp_factor**2 - normalized_flux.ej_sum**2 * fluct_factor
        normalized_flux.ej_var[np.where(t <= 0.)] = 0.0
        index = np.where(t > 0.)
        normalized_flux.ej_var[index] = np.sqrt(t[index])/normalized_flux.ej_sum[index] * 100.0

        return normalized_flux

    def check_index(self, i, j):
        """Check the validation of temperature index i and energy index j
        i: temperature index
        j: energy (correction) index

        """

        if i < 0 or i > len(self.temp_grid):
            raise ValueError("INVALID Temperature")

        if j < 0 or j >= self.energy_size:

            raise ValueError("INVALID Energy")

        if not self.is_pot():
            return False
        else:
            return True

    def average(self, i, j):

        """Calculate the flux average at temperature index i and energy index j
        i: temperature index
        j: energy (correction) index

        """
        if not self.check_index(i, j):
            return 0


        # ave = self.temp_sum[i, j]/float(self.acct_smp()) *\
        ave = self.temp_sum[i]/float(self.acct_smp()) *\
            self.pot_smp()/self.tot_smp()

        # This factor: pot_smp/tot_smp is mentioned in this paper
        # Georgievskii, Yuri, and Stephen J. Klippenstein. "
        # The Journal of Physical Chemistry A 107.46 (2003): 9776-9781.

        return ave

    def pot_fluctuation(self, temp_index, j):
        """ return the fluctuation based on potential sampling"""
        if not self.check_index(temp_index, j):
            return 0

        temp = self.temp_sum[temp_index, j]/float(self.acct_smp())


        fluc = np.sqrt(float(self.temp_var[temp_index, j])/float(self.acct_smp()) -
                       float(np.power(temp, 2))) * float(self.pot_smp())/float(self.tot_smp())

        return fluc

    def vol_fluctuation(self, i, j):
        """ return the fluctuation based on volume sampling"""
        if not self.check_index(i, j):
            return 0

        fluc = self.temp_sum[i, j]/float(self.acct_smp()) * \

            np.sqrt(float(self.space_smp())*float(self.pot_smp())) / self.tot_smp()


        return fluc

    def run_surf(self, samp_len):


        #Round up and transform as integer samp_len
        for i in range(0, np.ceil(samp_len).astype(int)):


            tag = self.sample.generate_configuration()

            if tag == 0:
                self._fake_num += 1
                continue
            elif tag == SampTag.SAMP_ATOMS_CLOSE:
                self._close_num += 1
                continue
            elif tag == SampTag.SAMP_FACE_OUT:
                self._face_num += 1
                continue

            elif tag == SampTag.SAMP_SUCCESS:  # new addition
                pass
            else:
                self.logger.error("rand_pos status unknow, EXITING\n")
                exit()

    def run(self, samp_len, face_id, flux_id=0):

        """This function is used for flux calculation

        Parameters
        ----------
        samp_len : int
            The number of sampling points for this time the "run" funcion is called.

        face_id : int
            Id of the face being sampled
        """
        # some constant

        as_num = 0  # account Sampling
        fs_num = 0  # failed samplings
        cs_num = 0  # face samping
        ds_num = 0  # distance too close sampling

        cn_fac = self.sample.get_canonical_factor()
        mc_fac = self.sample.get_microcanonical_factor()
        ej_fac = self.sample.get_ej_factor()
        dof = self.sample.get_dof()

        # each time, number of samp_len points are sampled.
        while (as_num < samp_len):

            # self.logger.info "the %dth point" % (as_num)

            pot_num = as_num + fs_num
            vol_num = cs_num + ds_num

            if fs_num > self.get_fail_num_max() and not as_num:

                self.logger.error("Opps, all potential failed")

                self.add_acct_smp(as_num)
                self.add_fail_smp(fs_num)
                self.add_face_smp(cs_num)
                self.add_dist_smp(ds_num)
                raise RuntimeError("ALL potentail failed")

            if vol_num > self.get_vol_num_max() and not self.is_pot():
                break

            """

            # tag is an integer represent the sampling info,

              weight, is a float which is a statistic weight that will be used
              in the flux calculation
              energy, is an array of energy, which includes all
              the corrected energy
              tim is the [x,y,z] which is the inertia moments of the configuration.
            """
            tag = self.sample.generate_configuration()

            if tag == 0:
                break
            elif tag == SampTag.SAMP_ATOMS_CLOSE:
                ds_num += 1
                continue
            elif tag == SampTag.SAMP_FACE_OUT:
                cs_num += 1
                continue
            elif tag == SampTag.SAMP_SUCCESS:
                pass
            else:

                self.logger.error("rand_pos status unknow, EXITING\n")
                exit()

            # now check energy:
            energy = self.sample.get_energies(self.calculator, face_id=face_id, flux_id=flux_id)

            if energy is None:
                fs_num += 1
                continue

            tim = self.sample.get_tot_inertia_moments()
            weight = self.sample.get_weight()

            if tim.any() is None or tim.any() < 0:
                fs_num += 1
                continue

            # now update the minimum energy
            for i in range(0, self.energy_size):

                traj = Trajectory('all_successful_config.traj', 'a', self.sample.configuration)
                traj.write()
                if energy[i] < self.min_energy[i]:
                    self.min_energy[i] = energy[i]
                    self.min_geometry[i] = self.sample.configuration.get_positions()
                    
            traj.close()


            as_num += 1
            # calculate the flux.
            for pes in range(0, self.energy_size):  # PES cycle

                # Cannonical flux:
                for t, temp in enumerate(self.temp_grid):  # temperature cycle
                    ratio = -energy[pes]/temp
                    if ratio > 200.0:
                        ratio = 300.0
                    if ratio > - 200.0:

                        ratio = cn_fac * weight * np.exp(ratio) * np.sqrt(temp)
                        self.temp_sum[t][pes] += ratio
                        self.temp_var[t][pes] += ratio ** 2

                # microcanonical flux:
                if self.flux_type == 'CANONICAL':
                    continue
                for en_ind in range(0, len(self.energy_grid)):  # energy grid cycle
                    ken = self.energy_grid[en_ind] - energy[pes]
                    if (ken) < 0:
                        continue
                    dtemp = np.power(ken, (dof-1)/2.0)
                    if (dof-1) % 2:
                        dtemp *= np.sqrt(ken)
                    dtemp *= (mc_fac) * weight
                    self.e_sum[en_ind][pes] += dtemp
                    self.e_var[en_ind][pes] += dtemp ** 2

                # E-J resoved flux
                if self.flux_type == 'MICROCANONICAL':
                    continue

                if self.flux_type != 'EJ-RESOLVED':
                    raise ValueError("INVALID flux type %s" % (self.flux_type))
                for en_ind in range(0, len(self.energy_grid)):  # enegy cycle
                    for am_ind in range(0, len(self.angular_grid)):  # angular momentum cycle
                        ken = self.energy_grid[en_ind] - energy[pes]
                        # rint ken
                        dtemp = rotd_math.mc_stat_weight(
                            ken, self.angular_grid[am_ind], tim, dof)

                        dtemp *= (ej_fac * weight / np.sqrt(tim[0] * tim[1] * tim[2]))
                        self.ej_sum[en_ind][am_ind][pes] += dtemp
                        self.ej_var[en_ind][am_ind][pes] += dtemp ** 2

        # update the sampled points
        self.add_acct_smp(as_num)
        self.add_fail_smp(fs_num)
        self.add_face_smp(cs_num)
        self.add_close_smp(ds_num)
