import os
import time
import copy
import glob
from collections import OrderedDict
from sqlite3 import connect
import pickle
import shutil
from subprocess import Popen, PIPE
import re
from itertools import compress
import numpy as np
import ase
import sys

from rotd_py.flux.flux import MultiFlux, Flux
import rotd_py.rotd_math as rotd_math
from rotd_py.system import FluxTag
from rotd_py.job_tpl import py_tpl_str
from rotd_py.analysis import integrate_micro
from rotd_py.analysis import create_matplotlib_graph

from rotd_py.config_log import config_log

class Multi(object):
    """
    This is the application Multi that has a lot of work to do
    so it gives work to do to its slaves until all the work is done
    """

    def __init__(self, fluxbase=None, dividing_surfaces=None, sample=None,
                 calculator=None, selected_faces=None):
        """Initialize the multi flux calculation.

        Parameters
        ----------
        fluxbase : FluxBase
            The essential information for a flux calculation.
        dividing_surfaces :
            A 1-D array, each item in array is a Surface object.
        sample : Sample
            The way of sampling that is going to be used in the flux calculation
        selected_faces : 
            A 1-D array determining which faces to use. If set to None, all faces are active.

        """
        self.dividing_surfaces = dividing_surfaces
        self.sample = sample
        self.fluxbase = fluxbase
        self.total_flux = OrderedDict()
        self.calculator = calculator
        self.work_queue, self.running_jobs, self.finished_jobs = [], [], []
        self.newly_finished_jobs = []
        self.show_grid_warn = True
        num_faces = 0
        self.workdir = os.getcwd()
        if not os.path.isdir(self.sample.name):
            os.mkdir(self.sample.name)
        shutil.copy(sys.argv[0], self.sample.name)
        os.chdir(self.sample.name)
        self.logger = config_log('rotdpy')
        os.chdir(self.workdir)
        if self.dividing_surfaces is not None:
            num_faces = [surface.get_num_faces() for surface in self.dividing_surfaces]
        if selected_faces == None:
            self.selected_faces = [range(surface.get_num_faces()) for surface in self.dividing_surfaces]
        else: #TODO: add a check on the shape of the user input selected faces
            self.selected_faces = selected_faces 
        self.flux_indexes = []
        self.converged = []
        for index, surf in enumerate(self.dividing_surfaces):
            surf.surf_id = str(index)
            indivi_sample = copy.deepcopy(sample)
            indivi_sample.set_dividing_surface(surf)
            #Restart db exists
            if os.path.isfile(f"{self.sample.name}/rotdPy_restart.db"):
                with connect(f'{self.sample.name}/rotdPy_restart.db', timeout=60) as cursor:
                    sql_cmd = 'SELECT * FROM rotdpy_saved_runs WHERE surf_id=?'
                    rows = cursor.execute(sql_cmd, (surf.surf_id,)).fetchall()
                if rows:
                    self.logger.info(f"RESTART: Entry for surface {surf.surf_id} found in {self.sample.name}/rotdPy_restart.db.")
                    surf_id, pkl_surf_flux, sample_list, pkl_old_flux_base = rows[-1]
                    self.flux_indexes.append(pickle.loads(sample_list))
                    self.total_flux[surf.surf_id] = pickle.loads(pkl_surf_flux)
                    old_flux_base = pickle.loads(pkl_old_flux_base)
                    for face_index, face_flux in enumerate(self.total_flux[surf.surf_id].flux_array):
                        self.flux_indexes[-1][face_index] = face_flux.acct_smp() + 1
                    #Surface is converged if it has an output, and the same or lower convergence threshold
                    if os.path.isfile(f'{self.sample.name}/output/surface_{surf.surf_id}.dat') and \
                    (old_flux_base._flux_parameter['flux_rel_err'] <= self.fluxbase._flux_parameter['flux_rel_err'] and \
                     old_flux_base._flux_parameter['pot_smp_max'] >= self.fluxbase._flux_parameter['pot_smp_max'] and \
                     old_flux_base._flux_parameter['pot_smp_min'] >= self.fluxbase._flux_parameter['pot_smp_min']):
                        self.converged.append(True)
                        self.logger.info(f"RESTART: Surface {surf.surf_id} successfully converged. No calculations needed.")
                    else:
                        self.converged.append(False)
                        self.logger.info(f"RESTART: Surface {surf.surf_id} not converged. More calculations needed.")
                        for fidx, face_num_samples in enumerate(self.flux_indexes[-1]):
                            #If a face has no sample, reinitialize the surface
                            if face_num_samples == 1:
                                self.logger.info(f"RESTART: Face {fidx} saved with no sample. Reinitializing the face's flux.")
                                self.total_flux[surf.surf_id].flux_array[fidx] = Flux(temp_grid=self.fluxbase.temp_grid * rotd_math.Kelv,
                                                                                    energy_grid=self.fluxbase.energy_grid * rotd_math.Kelv,
                                                                                    angular_grid=self.fluxbase.angular_grid,
                                                                                    flux_type=self.fluxbase.flux_type,
                                                                                    flux_parameter=self.fluxbase._flux_parameter,
                                                                                    sample=copy.deepcopy(indivi_sample),
                                                                                    calculator=self.calculator,
                                                                                    )
                                self.total_flux[surf.surf_id].flux_array[fidx].sample.div_surface.set_face(fidx)
                                self.flux_indexes[int(surf.surf_id)][fidx] = 1
                                for file in glob.glob(f"{self.sample.name}/Surface_{surf.surf_id}/jobs/surf{surf.surf_id}_face{fidx}*"):
                                    os.remove(file)
                        self.logger.info(f"RESTART: Jobs will restart form indexes {self.flux_indexes[-1]}.")
                else:
                    self.logger.info(f"RESTART: No entry for surface {surf.surf_id} found in {self.sample.name}/rotdPy_restart.db.")
                    self.logger.info(f"RESTART: Surface {surf.surf_id} initialized from scratch.")
                    self.flux_indexes.append([])
                    self.converged.append(False)
                    self.total_flux[surf.surf_id] = MultiFlux(fluxbase=fluxbase,
                                                        num_faces=num_faces[index],
                                                        selected_faces=self.selected_faces[index],
                                                        sample=indivi_sample,
                                                        calculator=self.calculator)
                    for face_index in range(0, self.total_flux[surf.surf_id].num_faces):
                        self.flux_indexes[int(surf.surf_id)].append(1)
            else:
                self.logger.info(f"RESTART: Database {self.sample.name}/rotdPy_restart.db was not found.")
                self.logger.info(f"RESTART: Surface {surf.surf_id} initialized from scratch.")
                self.flux_indexes.append([])
                self.converged.append(False)
                self.total_flux[surf.surf_id] = MultiFlux(fluxbase=fluxbase,
                                                    num_faces=num_faces[index],
                                                    selected_faces=self.selected_faces[index],
                                                    sample=indivi_sample,
                                                    calculator=self.calculator)
                for face_index in range(0, self.total_flux[surf.surf_id].num_faces):
                    self.flux_indexes[int(surf.surf_id)].append(1)
        self.ref_flux = copy.deepcopy(self.total_flux)
        #Submission script template
        self.py_tpl_str = py_tpl_str
        if self.calculator['queue'].casefold() == 'slurm':
            self.sub_cmd = 'sbatch'
            self.chk_cmd = 'scontrol show jobid -d {job_id}'
            self.cancel_cmd = 'scancel {job_id}'
            try:
                shutil.copy('qu.tpl', self.sample.name)
            except:
                print("Could not find qu.tpl: slurm submission template")
                exit()
        elif self.calculator['queue'].casefold() == 'mpi':
            pass
        if self.calculator['code'].casefold() == 'molpro':
            try:
                shutil.copy('molpro.tpl', self.sample.name)
            except:
                print("Could not find qu.tpl: slurm submission template")
                exit()
        elif self.calculator['code'][-3:].casefold() == 'amp':
            pass
        os.chdir(f"{self.workdir}/{self.sample.name}")
        for surface_id in range(0, len(self.dividing_surfaces)):
            if not os.path.isdir(f'Surface_{surface_id}'):
                os.mkdir(f'Surface_{surface_id}')
            os.chdir(f'Surface_{surface_id}')
            if not os.path.isdir('jobs'):
                os.mkdir('jobs')
            os.chdir(f"{self.workdir}/{self.sample.name}")
        with open('qu.tpl') as qu_tpl_fh:
            self.qu_tpl_str = qu_tpl_fh.read()   
        os.chdir(self.workdir)
       

    def run(self):

        os.chdir(self.sample.name)
        self.logger.info('Starting rotdpy run')
        first_job = True
        jobs_submitted = 0
        initial_submission = 0

        """Keep submitting jobs as long as there is work to do"""
        num_surfaces = len(self.total_flux)
        for surf in list(compress(self.dividing_surfaces, [not conv for conv in self.converged])):
            curr_flux = self.total_flux[surf.surf_id]  # multiflux
            self.logger.info(f'Information about runs')
            self.logger.info(f'Number of surfaces: {num_surfaces}')
            self.logger.info(f'Current surface: {surf.surf_id}')
            self.logger.info(f'Current number of facets: {curr_flux.num_faces}')


            # initialize the calculation for all facets of the surface
            for face_index in range(0, curr_flux.num_faces):
                flux = copy.deepcopy(curr_flux.flux_array[face_index])
                if first_job:
                    self.logger.info('The grids are:\n'
                                f'temperature (K) {flux.temp_grid}\n'
                                f'energy (cm-1) {flux.energy_grid}\n'
                                f'angular momentum (cm-1) {flux.angular_grid}')
                    first_job = False
                # create the minimum number of flux jobs, i.e., flux.pot_min()
                if face_index not in self.selected_faces[int(surf.surf_id)]:  # HACKED !!!
                    self.logger.info(f'Skipping face {face_index}')
                    continue
                for j in range(max(flux.pot_min()-self.flux_indexes[int(surf.surf_id)][face_index]+1, 1)):
                    #self.logger.info(f'Creating job {j} for surface {surf.surf_id} face {face_index} with id {self.flux_indexes[int(surf.surf_id)][face_index]}.')
                    self.work_queue.append((FluxTag.FLUX_TAG, flux,
                                            surf.surf_id, face_index, flux.samp_len(), 
                                            self.flux_indexes[int(surf.surf_id)][face_index], 'TO DO'))
                    self.flux_indexes[int(surf.surf_id)][face_index] += 1
                    initial_submission += 1

        while not all(self.converged):
            #Stop submitting when all work_queue has been submitted
            if len(self.work_queue) > jobs_submitted:
                self.submit_work(self.work_queue[jobs_submitted], procs=self.calculator["processors"])
                jobs_submitted += 1
                if jobs_submitted == 10000:
                    self.work_queue = self.work_queue[9999:]
                    jobs_submitted = 0
                    self.save_run_in_db()
            if initial_submission:
                initial_submission -= 1
            if not initial_submission and len(self.work_queue[jobs_submitted:]) < self.calculator['max_jobs']/2:
                self.check_running_jobs()
            for job in reversed(self.newly_finished_jobs):
                # update flux
                flux_tag, job_flux, surf_id, face_id, samp_len, samp_id, status = job
                #ßself.logger.debug(f'{flux_tag}, surf {surf_id}, face {face_id}, sample {samp_id} {status}')
                if self.converged[int(surf_id)]:
                    self.newly_finished_jobs.remove(job)
                    continue
                face_index = job_flux.sample.div_surface.get_curr_face()
                if face_index not in self.selected_faces[int(surf_id)]:  # HACKED !!!!!
                    self.newly_finished_jobs.remove(job)
                    continue
                curr_multi_flux = self.total_flux[surf_id]
                curr_flux = curr_multi_flux.flux_array[face_index]  # multiflux

                curr_flux.add_acct_smp(job_flux.acct_smp())
                curr_flux.add_close_smp(job_flux.close_smp())
                curr_flux.add_face_smp(job_flux.face_smp())
                curr_flux.add_fail_smp(job_flux.fail_smp())
                curr_flux.temp_sum += job_flux.temp_sum
                curr_flux.temp_var += job_flux.temp_var
                curr_flux.e_sum += job_flux.e_sum
                curr_flux.e_var += job_flux.e_var
                curr_flux.ej_sum += job_flux.ej_sum
                curr_flux.ej_var += job_flux.ej_var

                #self.logger.info(f'Successful samplings so far for face {face_index}: {curr_flux._acct_num}') # {job_flux.close_smp()} {job_flux.face_smp()} {job_flux.fail_smp()} \
                            #{job_flux.temp_sum} {job_flux.temp_var} {job_flux.e_sum} {job_flux.e_var}')

                # update the minimum energy and configuration
                for i in range(0, curr_flux.energy_size):
                    if job_flux.min_energy[i] < curr_flux.min_energy[i]:
                        curr_flux.min_energy[i] = job_flux.min_energy[i]
                        curr_flux.min_geometry[i] = job_flux.min_geometry[i].copy()
                # check the current flux converged or not
                flux_tag, smp_info = curr_multi_flux.check_state()
                self.logger.debug(f'tag {flux_tag} info {smp_info}')
                # if continue sample for current flux and for the face index:
                if flux_tag == FluxTag.FLUX_TAG:
                    face_index = smp_info
                    #self.logger.info(f'Creating job for face {face_index} with id {self.flux_indexes[int(surf_id)][face_index]}.')
                    flux = copy.deepcopy(self.ref_flux[surf_id].flux_array[face_index])
                    self.work_queue.append((flux_tag, flux, surf_id, face_index, flux.samp_len(), 
                                            self.flux_indexes[int(surf_id)][face_index], 'TO DO'))
                    self.flux_indexes[int(surf_id)][face_index] += 1

                elif flux_tag == FluxTag.SURF_TAG:
                    smp_num = smp_info
                    for face_index in range(0, len(smp_num)):
                        if smp_num[face_index] != 0:
                            #self.logger.info(f'Creating SURFACE job for face {face_index} with id {self.flux_indexes[int(surf_id)][face_index]}.')
                            flux = copy.deepcopy(self.ref_flux[surf_id].flux_array[face_index])
                            self.work_queue.append((flux_tag, flux, surf_id, face_index,
                                                    smp_num[face_index], self.flux_indexes[int(surf_id)][face_index], 'TO DO'))
                            self.flux_indexes[int(surf_id)][face_index] += 1

                            #self.logger.info(f'{FluxTag.SURF_TAG} flux_idx {self.flux_indexes[int(surf_id)][face_index]} face {face_index} smp_num {smp_num} surface {surf_id}')

                elif flux_tag == FluxTag.STOP_TAG:
                    self.logger.info(f'{FluxTag.STOP_TAG} was assigned to surface {surf_id}')
                    self.total_flux[surf_id].save_file(surf_id)
                    self.converged[int(surf_id)] = True #TODO: add full converged check
                    self.save_run_in_db()
                    self.logger.info(f'Calculations are done for surface {surf_id}')
                else:
                    self.logger.warning('The flux tag is INVALID')
                    raise ValueError("The flux tag is INVALID")
                self.finished_jobs.append(job)
                self.newly_finished_jobs.remove(job)
        self.save_run_in_db()

    def print_results(self, ignore_surf_id=None, dynamical_correction=1):
        os.chdir(f"{self.workdir}/{self.sample.name}")
                
        if ignore_surf_id == None or not isinstance(ignore_surf_id, list):
            ignore_surf_id = []

        mc_rate = []
        min_energies = []
        min_energies_dist = []
        sorted_r = []
        sorted_e = []
        splines = []
        temp_list = []
        data_legends_e = []
        data_legends_r = []
        comments = []
        save_min_energy_dist = False

        corrections = []
        for correction in self.sample.corrections.values():
            corrections.append(correction)
            if correction.type == "1d":
                save_min_energy_dist = True
                scan_ref = correction.scan_ref
                correction.plot()

        for output_energy_index in range(self.sample.energy_size):
            ftype = None

            mc_rate.append([])
            min_energies.append([])
            min_energies_dist.append([])
            multi_flux = {'Canonical': {},
                      'Microcanonical': {}}
                    #   'E-J resolved': {}}
            min_flux = {'Canonical': [np.inf for i in range(len(self.fluxbase.temp_grid))],
                        'Microcanonical': [np.inf for i in range(len(self.fluxbase.energy_grid))]}
                        #   'E-J resolved': []}
            flux_origin = {'Canonical': [],
                       'Microcanonical': []}
                    #   'E-J resolved': []}
            for surf in self.dividing_surfaces:
                # if not self.converged[int(surf.surf_id)]:
                #     continue
                
                if surf.surf_id not in ignore_surf_id:
                    for key in multi_flux :
                        multi_flux[key][surf.surf_id] = []
                else:
                    continue
                min_energies[output_energy_index].append(np.inf)
                min_geometry = []
                symbols = []
                # Collect data from the output
                with open(f'output/surface_{surf.surf_id}.dat', 'r') as f:
                    recording = False
                    for line in f.readlines():
                        if line.startswith('Face'):
                            face = int(line.split()[1])
                            continue
                        elif line.startswith('Minimum energy:'):
                            if float(line.split()[2+output_energy_index]) < min_energies[output_energy_index][-1]:
                                min_energies[output_energy_index][-1] = float(line.split()[2+output_energy_index])
                                recording = True
                            continue
                        elif line.startswith('Minimum energy geometry:'):
                            ftype = 'geometry'
                            continue
                        elif line.startswith('Canonical:'):
                            ftype = 'Canonical'
                            #recording = True
                            continue
                        elif line.startswith('Microcanonical:'):
                            #recording = True
                            ftype = 'Microcanonical'
                            continue
                        elif "Uncorrected" in line:
                            recording = True
                            continue
                        elif ftype == 'geometry':
                            if recording:
                                symbols.append(line.split()[0])
                                min_geometry.append([float(line.split()[1+4*output_energy_index]), float(line.split()[2+4*output_energy_index]), float(line.split()[3+4*output_energy_index])])
                            continue
                        elif line.startswith('E-J resolved:'):
                            ftype = None
                            recording = False
                            break
                        if not recording:
                            continue
                        if face == 0:
                            multi_flux[ftype][surf.surf_id].append(0.)

                        #Sum-up the flux of all faces
                        multi_flux[ftype][surf.surf_id][-1] += float(line.split()[output_energy_index+1])*dynamical_correction

                if save_min_energy_dist:
                    atoms_min = ase.Atoms(symbols, positions=min_geometry)
                    min_energies_dist[output_energy_index].append(atoms_min.get_distance(scan_ref[0][0], scan_ref[0][1]))
                #Save surface flux if minimum for a given temperature
                for temp_index in range(len(self.fluxbase.temp_grid)):
                    #Initialize flux_origin, which saves from which surface the flux is coming from
                    if len(flux_origin['Canonical']) < len(self.fluxbase.temp_grid):
                        flux_origin['Canonical'].append(surf.surf_id,)
                    if multi_flux['Canonical'][surf.surf_id][temp_index] < min_flux['Canonical'][temp_index]:
                        min_flux['Canonical'][temp_index] = multi_flux['Canonical'][surf.surf_id][temp_index]
                        flux_origin['Canonical'][temp_index] = surf.surf_id
                #Save surface flux if minimum for a given energy
                for energy_index in range(len(self.fluxbase.energy_grid)):
                    #Initialize flux_origin, which saves from which surface the flux is coming from
                    if len(flux_origin['Microcanonical']) < len(self.fluxbase.energy_grid):
                        flux_origin['Microcanonical'].append(surf.surf_id,)
                    if multi_flux['Microcanonical'][surf.surf_id][energy_index] < min_flux['Microcanonical'][energy_index]:
                        min_flux['Microcanonical'][energy_index] = multi_flux['Microcanonical'][surf.surf_id][energy_index]
                        flux_origin['Microcanonical'][energy_index] = surf.surf_id

            #Prepare micro-canonical plot
            mc_rate[output_energy_index] = integrate_micro(np.array(min_flux['Microcanonical']), self.fluxbase.energy_grid, self.fluxbase.temp_grid, self.sample.get_dof()) / 6.0221e12
            temp_list.append(self.fluxbase.temp_grid.tolist())
            comments.append(f"Sources: {flux_origin['Microcanonical']}")
            data_legends_r.append(f"rate_{self.sample.name}")

            #Prepare min energy plot
            sorted_r.append([])
            sorted_e.append([])
            for r, e in sorted(zip(min_energies_dist[output_energy_index], min_energies[output_energy_index])):
                sorted_r[output_energy_index].append(r)
                sorted_e[output_energy_index].append(e)
            if output_energy_index == 0:
                data_legends_e.append(f"Uncorrected")
            else:
                data_legends_e.append(f"{corrections[output_energy_index-1].name}({corrections[output_energy_index-1].type}) corrected")
            splines.append(False)

            if output_energy_index > 0.:
                if corrections[output_energy_index-1].type == "1d":
                    sorted_e.append(corrections[output_energy_index-1].e_sample)
                    sorted_r.append(corrections[output_energy_index-1].r_sample)
                    splines.append(True)
                    data_legends_e.append(f"sample_{self.sample.name}")
                    sorted_e.append(corrections[output_energy_index-1].e_trust)
                    sorted_r.append(corrections[output_energy_index-1].r_trust)
                    splines.append(True)
                    data_legends_e.append(f"trust_{self.sample.name}")


        create_matplotlib_graph(x_lists=sorted_r, data=sorted_e, name=f"{self.sample.name}_min_energy",\
                                x_label=f"{symbols[scan_ref[0][0]]}{scan_ref[0][0]} to {symbols[scan_ref[0][1]]}{scan_ref[0][1]} distance ($\AA$)",
                                y_label="Energy (Kcal/mol)", data_legends=data_legends_e,\
                                exponential=False, splines=splines, title="Sampled minimum energy")#, comments=comments)
        
        create_matplotlib_graph(x_lists=temp_list, data = mc_rate, name=f"{self.sample.name}_micro_rate",\
                                x_label="Temperature (K)", y_label="Rate constant (cm$^{3}$molecule$^{-1}$s$^{-1}$)", data_legends=data_legends_r,\
                                exponential=True, comments=comments, title="Micro-canonical rate")

    def save_run_in_db(self):
        if not os.path.isfile(f'rotdPy_restart.db'):
            with connect(f'rotdPy_restart.db', timeout=60) as cursor:
                cursor.execute("CREATE TABLE IF NOT EXISTS rotdpy_saved_runs "
                            "(surf_id int, multi_flux blob, sample_list blob, flux_base blob)")
            sql_command = "INSERT INTO rotdpy_saved_runs VALUES "\
                          "(:surf_id, :multi_flux, :sample_list, :flux_base)"
            for surf in self.dividing_surfaces:
                with connect(f'rotdPy_restart.db', timeout=60) as cursor:
                    cursor.execute(sql_command, {'surf_id': surf.surf_id,
                                                'multi_flux': pickle.dumps(self.total_flux[surf.surf_id]),
                                                'sample_list': pickle.dumps(self.flux_indexes[int(surf.surf_id)]),
                                                'flux_base': pickle.dumps(self.fluxbase)
                                                })
        else:
            for surf in self.dividing_surfaces:
                with connect(f'rotdPy_restart.db', timeout=60) as cursor:
                    cursor.execute('UPDATE rotdpy_saved_runs SET multi_flux=:multi_flux, sample_list=:sample_list, flux_base=:flux_base '
                        'WHERE surf_id = :surf_id', \
                        {'surf_id': surf.surf_id,
                        'multi_flux': pickle.dumps(self.total_flux[surf.surf_id]),
                        'sample_list': pickle.dumps(self.flux_indexes[int(surf.surf_id)]),
                        'flux_base': pickle.dumps(self.fluxbase)
                                                    })

    def submit_work(self, job, procs=1):
        """

        Parameters
        ----------
        job

        Returns
        -------

        """
        # Initial checks of job db/queue status and maximum jobs limit.
        job = self.check_job_status(job)
        flux_tag, flux, surf_id, face_id, samp_len, samp_id, status = job
        #Avoid submitting jobs for surfaces already converged if jobs are in the work queue
        if not self.converged[int(surf_id)]:
            if status == 'NEWLY FINISHED' and job not in self.newly_finished_jobs:
                self.newly_finished_jobs.append(job)
                # self.del_db_job(job)
                return
            elif status == 'RUNNING':
                self.running_jobs.append(job)
                return
            # elif status == 'FAILED':
            #     self.del_db_job(job)
            while len(self.running_jobs) >= self.calculator['max_jobs']:
                time.sleep(1)
                self.check_running_jobs()

            #Jobs with status FAILED or TO DO will come here

            # Serialize the job to be picked-up by the sample
            with open(f'Surface_{surf_id}/jobs/surf{surf_id}_face{face_id}_samp{samp_id}.pkl', 'wb') as pkl_file:
                pickle.dump([flux_tag.value, flux, surf_id, face_id, samp_len, samp_id, 'RUNNING'], pkl_file)

            # Launch the job
            os.chdir(f'Surface_{surf_id}/jobs')
            with open(f'surf{surf_id}_face{face_id}_samp{samp_id}.py', 'w') as py_job_fh:
                py_job_fh.write(self.py_tpl_str.format(surf_id=surf_id,
                                                    face_id=face_id,
                                                    samp_id=samp_id))
            with open(f'surf{surf_id}_face{face_id}_samp{samp_id}.sh', 'w') as qu_job_fh:
                qu_job_fh.write(self.qu_tpl_str.format(surf_id=surf_id,
                                                    face_id=face_id,
                                                    samp_id=samp_id,
                                                    procs=procs,))
            stdout, stderr = Popen(f'{self.sub_cmd} surf{surf_id}_face{face_id}_samp{samp_id}.sh',
                                shell=True, stdout=PIPE,
                                stderr=PIPE).communicate()
            stdout, stderr = (std.decode() for std in (stdout, stderr))
            try:
                flux.job_id = stdout.split()[3]
                err = False
            except IndexError:
                err = True
            if err:
                raise OSError('SLURM does not seem to be installed. Error '
                            f'message:\n\n{stderr}')
            os.chdir(f"{self.workdir}/{self.sample.name}")

            # Relocate the job into the running list
            job = (flux_tag, flux, surf_id, face_id, samp_len, samp_id, 'RUNNING')
            self.running_jobs.append(job)


    def check_running_jobs(self):
        for job in reversed(self.running_jobs):
            updated_job = self.check_job_status(job)
            try:
                status = updated_job[6]
            except TypeError:
                while updated_job == None:
                    updated_job = self.check_job_status(job)
                    try:
                        status = updated_job[6]
                    except TypeError:
                        continue
            if status == 'RUNNING':
                continue
            elif status == 'FAILED':
                #Delete files when job is finished
                self.del_db_job(job)
                self.work_queue.append(updated_job)
                self.running_jobs.remove(job)
            elif status == 'NEWLY FINISHED' and updated_job not in self.newly_finished_jobs:
                #Delete files when job is finished
                self.del_db_job(job)
                self.newly_finished_jobs.append(updated_job)
                self.running_jobs.remove(job)

    def check_job_status(self, job):
        flux_tag, flux, surf_id, face_id, samp_len, samp_id, status = job
        while os.path.isfile(f'Surface_{surf_id}/jobs/surf{surf_id}_face{face_id}_samp{samp_id}.pkl'):
            try:
                with open(f'Surface_{surf_id}/jobs/surf{surf_id}_face{face_id}_samp{samp_id}.pkl', 'rb') as pkl_file:
                    pickle_job = pickle.load(pkl_file)
                break
            except:
                time.sleep(0.1)
                for i in range(3):
                    self.logger.debug(f'UnpicklingError: Unsuccesful opening of surf{surf_id}_face{face_id}_samp{samp_id}.pkl, retrying...')
                    try:
                        with open(f'Surface_{surf_id}/jobs/surf{surf_id}_face{face_id}_samp{samp_id}.pkl', 'rb') as pkl_file:
                            pickle_job = pickle.load(pkl_file)
                        break
                    except:
                        time.sleep(0.1)
                        if i == 2:
                            return flux_tag, flux, surf_id, face_id, samp_len, samp_id, "FAILED"
        else:
            return flux_tag, flux, surf_id, face_id, samp_len, samp_id, "TO DO"
        db_status = pickle_job[6]
        db_flux = pickle_job[1]
        db_flux.job_id = flux.job_id
        # Check if the data from the db is usable.
        if len(db_flux.energy_grid) != len(flux.energy_grid) \
            or len(db_flux.temp_grid) != len(flux.temp_grid) \
            or len(db_flux.angular_grid) != len(flux.angular_grid) \
            or not (db_flux.energy_grid == flux.energy_grid).all() \
            or not (db_flux.temp_grid == flux.temp_grid).all() \
            or not (db_flux.angular_grid == flux.angular_grid).all():
            self.logger.warning('The database entries have points calculated with '
                            'different grids. Unable to use them with the current '
                            'calculation.')
            return job
        # Check if the job is actually running.
        if db_status.upper() == 'RUNNING':
            out, _ = Popen(self.chk_cmd.format(job_id=flux.job_id),
                            shell=True, stdout=PIPE, stderr=PIPE).communicate()
            if out:
                for line in out.decode().split('\n'):
                    if 'JobState' in line:
                        queue_status = re.split('\W+', line)[2].upper()
                        if queue_status == 'RUNNING' or queue_status == 'PENDING':
                            return flux_tag, flux, surf_id, face_id, samp_len, \
                                    samp_id, "RUNNING"
                        elif queue_status == 'COMPLETED':
                            return flux_tag, db_flux, surf_id, face_id,\
                                    samp_len, samp_id, "NEWLY FINISHED"
                        elif queue_status == 'FAILED':
                            return flux_tag, flux, surf_id, face_id, samp_len, \
                                    samp_id, "FAILED"
                        break
            else:
                return flux_tag, flux, surf_id, face_id, samp_len, samp_id, "FAILED"
        else:
            return flux_tag, db_flux, surf_id, face_id, samp_len, samp_id, "NEWLY FINISHED"

    def del_db_job(self, job):
        _0, _1, surf_id, face_id, _4, samp_id, _6 = job
        if os.path.isfile(f'Surface_{surf_id}/jobs/surf{surf_id}_face{face_id}_samp{samp_id}.pkl'):
            try:
                os.remove(f'Surface_{surf_id}/jobs/surf{surf_id}_face{face_id}_samp{samp_id}.pkl')
            except FileNotFoundError:
                self.logger.debug('Could not delete surf{surf_id}_face{face_id}_samp{samp_id}.pkl')
        if os.path.isfile(f'Surface_{surf_id}/jobs/surf{surf_id}_face{face_id}_samp{samp_id}.xml'):
            try:
                os.remove(f'Surface_{surf_id}/jobs/surf{surf_id}_face{face_id}_samp{samp_id}.xml')
            except FileNotFoundError:
                self.logger.debug('Could not delete surf{surf_id}_face{face_id}_samp{samp_id}.xml')
        if os.path.isfile(f'Surface_{surf_id}/jobs/surf{surf_id}_face{face_id}_samp{samp_id}.inp'):
            try:
                os.remove(f'Surface_{surf_id}/jobs/surf{surf_id}_face{face_id}_samp{samp_id}.inp')
            except FileNotFoundError:
                self.logger.debug('Could not delete surf{surf_id}_face{face_id}_samp{samp_id}.inp')
        if os.path.isfile(f'Surface_{surf_id}/jobs/surf{surf_id}_face{face_id}_samp{samp_id}.sh'):
            try:
                os.remove(f'Surface_{surf_id}/jobs/surf{surf_id}_face{face_id}_samp{samp_id}.sh')
            except FileNotFoundError:
                self.logger.debug('Could not delete surf{surf_id}_face{face_id}_samp{samp_id}.sh')
        # if os.path.isfile(f'Surface_{surf_id}/jobs/surf{surf_id}_face{face_id}_samp{samp_id}.py'):
        #     try:
        #         os.remove(f'Surface_{surf_id}/jobs/surf{surf_id}_face{face_id}_samp{samp_id}.py')
        #     except FileNotFoundError:
        #         self.logger.debug('Could not delete surf{surf_id}_face{face_id}_samp{samp_id}.py')
