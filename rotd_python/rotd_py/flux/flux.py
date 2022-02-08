import numpy as np

from rotd_py.system import SampTag, FluxTag
import rotd_py.rotd_math as rotd_math
from rotd_py.flux.fluxbase import FluxBase
import copy
import os
min_flux = 1e-99
# CAUTION! all units are in atomic unit (au)


class MultiFlux:
    """Manage a flux array for each dividing surface,
    each item in array is the flux calculation for a specific facet

    Parameters
    ----------
    fluxbase: FluxBase, all the information needed for the flux calculation
    num_faces: the number of faces for each dividing surface. 
               This number determines the dimension of the MultiFlux
    """

    def __init__(self, fluxbase=None, num_faces=1, sample=None, calculator=None):

        if fluxbase is None:
            raise ValueError("The flux base cannot be NONE, the information\
                             for flux calculation is needed")
        self.flux_array = [None] * num_faces
        self.num_faces = num_faces
        self.temp_grid = fluxbase.temp_grid
        self.tol = fluxbase.tol()
        self.pot_max = fluxbase.pot_max()
        self.tot_max = fluxbase.tot_max()

        # Initialize each flux to the correspondent facet.
        # Convert the energy unit to Hartree
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

    def check_state(self):
        """Check whether the calculation for this dividing surface has converged
        or not. Return the flux type and the face index that need to be feed to
        slave (or should be added to the work queue)

        Returns
        -------
        type
            [FluxTag, int]

        """
        for i in range(0, self.num_faces):
            if self.flux_array[i].acct_smp() >= self.flux_array[i].pot_min():
                continue
            else:
                return FluxTag.FLUX_TAG, i

        # estimate the error
        min_val = float('inf')
        min_index = 0

        # find the minimum thermal flux
        for t_ind in range(0, len(self.temp_grid)):
            val = 0.
            for i in range(0, self.num_faces):
                val += self.flux_array[i].average(t_ind, 0)
            if val < min_val:
                min_val = val
                min_index = t_ind

        # total potential variance
        tot_pot_var = 0.0
        for i in range(0, self.num_faces):
            tot_pot_var += self.flux_array[i].pot_fluctuation(min_index, 0)

        # projected number of samplings
        proj_smp_num = 0.
        if min_val > 1.0e-99:
            proj_smp_num = 10000. * tot_pot_var**2 / min_val**2 / self.tol**2
            if proj_smp_num > self.pot_max:
                proj_smp_num = self.pot_max

        if proj_smp_num > 1.:
            max_smp = -1
            face = 0
            for i in range(0, self.num_faces):
                smp = proj_smp_num * \
                    self.flux_array[i].pot_fluctuation(min_index, 0) / tot_pot_var
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
        for i in range(0, self.num_faces):
            tot_vol_var += self.flux_array[i].vol_fluctuation(min_index, 0)

        # projected number of sampling
        if min_val > 1.0e-99:
            proj_smp_num = 10000. * tot_vol_var**2 / min_val**2/self.tol**2
            if proj_smp_num > self.tot_max:
                proj_smp_num = self.tot_max

        if proj_smp_num <= 1.:
            return [FluxTag.STOP_TAG, -1]

        # check whether surface sampling is needed or not
        is_surf_smp = False
        smp_num = np.zeros(self.num_faces)
        for i in range(0, self.num_faces):
            smp = proj_smp_num * self.flux_array[i].vol_fluctuation(min_index, 0)\
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
        Save the calculation results after the current dividing surface is 
        converged
        sid is the index of current dividing surface
        """
        filename = "surface_"+str(sid)+".dat"
        if os.path.exists(filename):
            print("The file  %s is already exists, REWRITE it!" % (filename))
        f = open(filename, 'w')

        symbols = self.flux_array[0].sample.configuration.get_chemical_symbols()
        for face in range(self.num_faces):
            f.write("Face: %d\n" % (face))
            curr_flux = self.flux_array[face]
            f.write("Successful sampling: %d \n" % (curr_flux.acct_smp()))
            f.write("Failed sampling: %d \n" % (curr_flux.fail_smp()))
            f.write("Close-atoms sampling: %d \n" % (curr_flux.close_smp()))
            f.write("Out of face sampling: %d \n" % (curr_flux.face_smp()))
            f.write("Dummy sampling: %d \n" % (curr_flux.fake_smp()))
            # here out put unit with kcal/mol
            f.write("Minimum energy: %.5f\n" % (curr_flux.min_energy[0]/rotd_math.Kcal))
            f.write("Minimum energy geometry:\n")
            for i in range(0, len(curr_flux.min_geometry[0])):
                item = curr_flux.min_geometry[0][i]
                f.write("%5s %16.8f %16.8f %16.8f\n" % (symbols[i], item[0], item[1], item[2]))

            # Now normalize the calculated results
            curr_flux.normalize()
            
            # the Canonical:
            f.write("Canonical: \n")
            for i in range(0, len(curr_flux.temp_grid)):
                line = "%-16.3f  " % (curr_flux.temp_grid[i]/rotd_math.Kelv)
                line += " ".join(format(x, ".3e") for x in curr_flux.temp_sum[i, :])
                f.write(line + '\n')
            
            # the Micro-canonical:
            f.write("Microcanonical: \n")
            for i in range(0, len(curr_flux.energy_grid)):
                line = "%-16.3f" % (curr_flux.energy_grid[i]/rotd_math.Kelv)
                line += " ".join(format(x, ".3e") for x in curr_flux.e_sum[i, :])
                f.write(line + '\n')

            f.write("E-J resolved: \n")
            for j in range(0, len(curr_flux.angular_grid)):
                for i in range(0, len(curr_flux.energy_grid)):
                    line = " ".join(format(x, ".3e") for x in curr_flux.ej_sum[i, j, :])
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
        The number of dummy sampling (sample without executing potential calculation.)
    _close_num : int
        The number of sampling whose distance of atoms are too close.
    _face_num : int
        The number of face out sampling.

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
        self.energy_size = sample.energy_size

        self.set_calculation()
        self.set_vol_num_max()
        self.set_fail_num_max()
        self.min_energy = [float('inf')] * self.energy_size
        self.min_geometry = [None] * self.energy_size

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
        # test function to see whether flux normalization can be started
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
        samp_factor = float(self.pot_smp())/self.tot_smp()/self.acct_smp()

        fluct_factor = 1.0/self.acct_smp() + 1.0/self.tot_smp() - 1.0/self.pot_smp()

        # dealing with the thermal flux in total
        self.temp_sum *= samp_factor
        self.temp_var[np.where(self.temp_sum < min_flux)] = 0.0
        t = self.temp_var * samp_factor**2 - self.temp_sum**2 * fluct_factor
        self.temp_var[np.where(t <= 0.)] = 0.0
        index = np.where(t > 0.)
        self.temp_var[index] = np.sqrt(t[index])/self.temp_sum[index] * 100.0

        # microcanonical
        self.e_sum *= samp_factor
        self.e_var[np.where(self.e_sum < min_flux)] = 0.0
        t = self.e_var * samp_factor**2 - self.e_sum**2 * fluct_factor
        self.e_var[np.where(t <= 0.)] = 0.0
        index = np.where(t > 0.)
        self.e_var[index] = np.sqrt(t[index])/self.e_sum[index] * 100.0

        # e-j resolved
        self.ej_sum *= samp_factor
        self.ej_var[np.where(self.ej_sum < min_flux)] = 0.0
        t = self.ej_var * samp_factor**2 - self.ej_sum**2 * fluct_factor
        self.ej_var[np.where(t <= 0.)] = 0.0
        index = np.where(t > 0.)
        self.ej_var[index] = np.sqrt(t[index])/self.ej_sum[index] * 100.0

    def check_index(self, i, j):
        """Check the validation of temperature index i and energy index j
        i: temperature index
        j: energy index
        """

        if i < 0 or i > len(self.temp_grid):
            raise ValueError("INVALID Temperature")
        if j < 0 or j >= len(self.energy_grid):
            raise ValueError("INVALID Energy")

        if not self.is_pot():
            return False
        else:
            return True

    def average(self, i, j):
        """Calculate the flux average at energy index i and temperature index j
        i: temperature index
        j: energy index
        """
        if not self.check_index(i, j):
            return 0

        ave = self.temp_sum[i, j]/float(self.acct_smp()) *\
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

        fluc = np.sqrt(self.temp_var[temp_index, j]/float(self.acct_smp()) -
                       np.power(temp, 2)) * self.pot_smp()/self.tot_smp()
        return fluc

    def vol_fluctuation(self, i, j):
        """ return the fluctuation based on volume sampling"""
        if not self.check_index(i, j):
            return 0

        fluc = self.temp_sum[i, j]/float(self.acct_smp()) * \
            np.sqrt(self.space_smp()*self.pot_smp()) / self.tot_smp()

        return fluc

    def run_surf(self, samp_len):

        for i in range(0, samp_len):

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
            else:
                print("rand_pos status unknow, EXITING\n")
                exit()

    def run(self, samp_len):
        """This function is used for flux calculation

        Parameters
        ----------
        samp_len : int
            The number of sampling points for this time the "run" funcion is called.
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
            # print "the %dth point" % (as_num)
            pot_num = as_num + fs_num
            vol_num = cs_num + ds_num

            if fs_num > self.get_fail_num_max() and not as_num:
                print("Opps, all potential failed")
                self.add_acct_smp(as_num)
                self.add_fail_smp(fs_num)
                self.add_face_smp(cs_num)
                self.add_dist_smp(ds_num)
                raise RuntimeError("ALL potentail failed")

            if vol_num > self.get_vol_num_max() and not self.is_pot():
                break

            """
              tag is an integer which represents the sampling info.
              weight, is a float which is a statistic weight that will be used in the flux calculation.
              energy is an array of energy, which includes all the corrected energy
              tim is the [x,y,z] which is the moments of inertia for the configuration.
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
                print("rand_pos status unknow, EXITING\n")
                exit()

            # now check energy:
            energy = self.sample.get_energies(self.calculator)
            print("flux: %f" % (energy[0]))
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
                if energy[i] < self.min_energy[i]:
                    self.min_energy[i] = energy[i]
                    self.min_geometry[i] = self.sample.configuration.get_positions()

            as_num += 1
            # calculate the flux.
            for pes in range(0, self.energy_size):  # PES cycle
                # print energy[pes]
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
                if self.flux_type == 'CANNONICAL':
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
                if self.flux_type == 'MICROCANNONICAL':
                    continue

                if self.flux_type != 'EJ-RESOLVED':
                    raise ValueError("INVALID flux type%s" % (self.flux_type))
                # you need to figure out what is mc stat weight
                for en_ind in range(0, len(self.energy_grid)):  # enegy cycle
                    for am_ind in range(0, len(self.angular_grid)):  # angular momentum cycle
                        ken = self.energy_grid[en_ind] - energy[pes]
                        # rint ken
                        dtemp = rotd_math.mc_stat_weight(
                            ken, self.angular_grid[am_ind], tim, dof)
                        # print "mc_weight:%f\n" % (dtemp)
                        dtemp *= (ej_fac * weight / np.sqrt(tim[0] * tim[1] * tim[2]))
                        self.ej_sum[en_ind][am_ind][pes] += dtemp
                        self.ej_var[en_ind][am_ind][pes] += dtemp ** 2

        # update the sampled points
        self.add_acct_smp(as_num)
        self.add_fail_smp(fs_num)
        self.add_face_smp(cs_num)
        self.add_close_smp(ds_num)