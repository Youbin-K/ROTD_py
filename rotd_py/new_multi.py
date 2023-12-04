import os
import time
import copy
from collections import OrderedDict
from sqlite3 import connect
import pickle
import shutil
from subprocess import Popen, PIPE
import re

from rotd_py.flux.flux import MultiFlux
from rotd_py.system import FluxTag
from rotd_py.job_tpl import py_tpl_str

from rotd_py.config_log import config_log
#global self.logger

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
        if self.dividing_surfaces is not None:
            num_faces = [surface.get_num_faces() for surface in self.dividing_surfaces]
        if selected_faces == None:
            self.selected_faces = [range(surface.get_num_faces()) for surface in self.dividing_surfaces]
        else: #TODO: add a check on the shape of the user input selected faces
            self.selected_faces = selected_faces 
        for index, surf in enumerate(self.dividing_surfaces):
            surf.surf_id = str(index)
            indivi_sample = copy.deepcopy(sample)
            indivi_sample.set_dividing_surface(surf)
            self.total_flux[surf.surf_id] = MultiFlux(fluxbase=fluxbase,
                                                 num_faces=num_faces[index],
                                                 selected_faces=self.selected_faces[index],
                                                 sample=indivi_sample,
                                                 calculator=self.calculator)
        self.surf_id_gen = (str(surf.surf_id) for surf in self.dividing_surfaces)
        self.ref_flux = copy.deepcopy(self.total_flux) 
        self.py_tpl_str = py_tpl_str
        self.workdir = os.getcwd()
        if not os.path.isdir(self.sample.name):
            os.mkdir(self.sample.name)
        if self.calculator['queue'].casefold() == 'slurm':
            self.sub_cmd = 'sbatch'
            self.chk_cmd = 'scontrol show jobid -d {job_id}'  
            try:
                shutil.copy('qu.tpl', self.sample.name)
            except:
                print("Could not find qu.tpl: slurm submission template")
                exit()
        if self.calculator['code'].casefold() == 'molpro':
            try:
                shutil.copy('molpro.tpl', self.sample.name)
            except:
                print("Could not find qu.tpl: slurm submission template")
                exit()
        os.chdir(f"{self.workdir}/{self.sample.name}")
        self.logger = config_log('rotdpy')
        for surface_id in range(0, len(self.dividing_surfaces)):
            if not os.path.isdir(f'Surface_{surface_id}'):
                os.mkdir(f'Surface_{surface_id}')
            os.chdir(f'Surface_{surface_id}')
            # with connect(f'rotd.db', timeout=60) as cursor:
            #     cursor.execute("CREATE TABLE IF NOT EXISTS fluxes "
            #                 "(flux_tag int, flux blob, surf_id text, face_id int, "
            #                 "samp_len int, samp_id int, status text)")
            if not os.path.isdir('jobs'):
                os.mkdir('jobs')
            os.chdir(f"{self.workdir}/{self.sample.name}")
        with open('qu.tpl') as qu_tpl_fh:
            self.qu_tpl_str = qu_tpl_fh.read()   
        os.chdir(self.workdir)
       

    def run(self):

        os.chdir(self.sample.name)
        self.logger.info('Starting rotdpy')
        first_job = True
        jobs_submitted = 0
        initial_submission = 0


        """Keep submitting jobs as long as there is work to do"""
        num_surfaces = len(self.total_flux)
        self.flux_indexes = []
        self.converged = []
        for surf in self.dividing_surfaces:
            #test if surface converged
            #Restart check
            if os.path.isfile(f"output/Surface_{surf.surf_id}.dat"):
                self.converged.append(True)
                continue
            else:
                self.converged.append(False)
            self.flux_indexes.append([])
            curr_flux = self.total_flux[surf.surf_id]  # multiflux
            self.logger.info(f'Information about runs')
            self.logger.info(f'Number of surfaces: {num_surfaces}')
            self.logger.info(f'Current surface: {surf.surf_id}')
            self.logger.info(f'Current number of facets: {curr_flux.num_faces}')


            # initialize the calculation for all facets of the surface
            for face_index in range(0, curr_flux.num_faces):
                
                self.flux_indexes[int(surf.surf_id)].append(1)
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
                for j in range(flux.pot_min()):
                    self.logger.info(f'Creating job {j} for face {face_index} with id {self.flux_indexes[int(surf.surf_id)][face_index]}.')
                    self.work_queue.append((FluxTag.FLUX_TAG, flux,
                                            surf.surf_id, face_index, flux.samp_len(), 
                                            self.flux_indexes[int(surf.surf_id)][face_index], 'TO DO'))
                    self.flux_indexes[int(surf.surf_id)][face_index] += 1
                    initial_submission += 1


        while not all(self.converged):
            if len(self.work_queue) >= jobs_submitted:
                self.submit_work(self.work_queue[jobs_submitted], procs=self.calculator["processors"])
                jobs_submitted += 1
                if jobs_submitted == 500:
                    self.work_queue = self.work_queue[499:]
                    jobs_submitted = 0
                if initial_submission:
                    initial_submission -= 1
            if not initial_submission and len(self.work_queue) < self.calculator['max_jobs']/2:
                self.check_running_jobs()
            for job in reversed(self.newly_finished_jobs):
                # update flux
                flux_tag, job_flux, surf_id, face_id, samp_len, samp_id, status = job
                self.logger.debug(f'{flux_tag}, surf {surf_id}, face {face_id}, sample {samp_id} {status}')
                if self.converged[int(surf_id)]:
                    continue
                face_index = job_flux.sample.div_surface.get_curr_face()
                if face_index not in self.selected_faces[int(surf_id)]:  # HACKED !!!!!
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

                self.logger.info(f'Successful samplings so far for face {face_index}: {curr_flux._acct_num}') # {job_flux.close_smp()} {job_flux.face_smp()} {job_flux.fail_smp()} \
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
                    self.logger.info(f'Creating job for face {face_index} with id {self.flux_indexes[int(surf_id)][face_index]}.')
                    flux = copy.deepcopy(self.ref_flux[surf_id].flux_array[face_index])
                    self.work_queue.append((flux_tag, flux, surf_id, face_index, flux.samp_len(), 
                                            self.flux_indexes[int(surf_id)][face_index], 'TO DO'))
                    self.flux_indexes[int(surf_id)][face_index] += 1

                elif flux_tag == FluxTag.SURF_TAG:
                    smp_num = smp_info
                    for face_index in range(0, len(smp_num)):
                        if smp_num[face_index] != 0:
                            self.logger.info(f'Creating SURFACE job for face {face_index} with id {self.flux_indexes[int(surf_id)][face_index]}.')
                            flux = copy.deepcopy(self.ref_flux[surf_id].flux_array[face_index])
                            self.work_queue.append((flux_tag, flux, surf_id, face_index,
                                                    smp_num[face_index], self.flux_indexes[int(surf_id)][face_index], 'TO DO'))
                            self.flux_indexes[int(surf_id)][face_index] += 1

                            self.logger.info(f'{FluxTag.SURF_TAG} flux_idx {self.flux_indexes[int(surf_id)][face_index]} face {face_index} smp_num {smp_num} surface {surf_id}')

                elif flux_tag == FluxTag.STOP_TAG:
                    self.logger.info(f'{FluxTag.STOP_TAG} was assigned to surface {surf_id}')
                    self.total_flux[surf_id].save_file(surf_id)
                    self.converged[int(surf_id)] = True #TODO: add full converged check
                    self.logger.info(f'Calculations are done for surface {surf_id}')
                else:
                    self.logger.warning('The flux tag is INVALID')
                    raise ValueError("The flux tag is INVALID")
                self.finished_jobs.append(job)
                self.newly_finished_jobs.remove(job)


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
        if status == 'NEWLY FINISHED' and job not in self.newly_finished_jobs:
            self.newly_finished_jobs.append(job)
            return
        elif status == 'RUNNING':
            self.running_jobs.append(job)
            return
        elif status == 'FAILED':
            self.del_db_job(job)
        while len(self.running_jobs) >= self.calculator['max_jobs']:
            time.sleep(5)
            self.check_running_jobs()

        # Create a db entry
        with open(f'Surface_{surf_id}/jobs/surf{surf_id}_face{face_id}_samp{samp_id}.pkl', 'wb') as pkl_file:
            pickle.dump([flux_tag.value, flux, surf_id, face_id, samp_len, samp_id, 'RUNNING'], pkl_file)
        # srl_flux = pickle.dumps(flux)  # Serialize the flux object
        # sql_cmd = "INSERT INTO fluxes VALUES (:flux_tag, :flux, :surf_id, :face_id, " \
        #           ":samp_len, :samp_id, :status)"
        # with connect(f'Surface_{surf_id}/rotd.db', timeout=60) as cursor:
        #     cursor.execute(sql_cmd, {'flux_tag': int_flux_tag,
        #                              'flux': srl_flux,
        #                              'surf_id': surf_id,
        #                              'face_id': face_id,
        #                              'samp_len': samp_len,
        #                              'samp_id': samp_id,
        #                              'status': 'RUNNING'
        #                              })

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
                self.del_db_job(job)
                self.work_queue.append(updated_job)
                self.running_jobs.remove(job)
            elif status == 'NEWLY FINISHED' and updated_job not in self.newly_finished_jobs:
                self.newly_finished_jobs.append(updated_job)
                self.running_jobs.remove(job)

    def check_job_status(self, job):
        flux_tag, flux, surf_id, face_id, samp_len, samp_id, status = job
        # Check if the job is in the database:
        # with connect(f'Surface_{surf_id}/rotd.db', timeout=60) as cursor:
        #     sql_cmd = 'SELECT * FROM fluxes WHERE surf_id=? AND face_id=? AND samp_id=?'
        #     rows = cursor.execute(sql_cmd, (surf_id, face_id, samp_id)).fetchall()
        if os.path.isfile(f'Surface_{surf_id}/jobs/surf{surf_id}_face{face_id}_samp{samp_id}.pkl'):
            with open(f'Surface_{surf_id}/jobs/surf{surf_id}_face{face_id}_samp{samp_id}.pkl', 'rb') as pkl_file:
                pickle_job = pickle.load(pkl_file)
            db_status = pickle_job[6]
            db_flux = pickle_job[1]
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
        else:
            return flux_tag, flux, surf_id, face_id, samp_len, samp_id, "TO DO"

    def update_db_job_status(self, job, status):
        flux_tag, flux, surf_id, face_id, samp_len, samp_id, old_status = job
        with open(f'Surface_{surf_id}/jobs/surf{surf_id}_face{face_id}_samp{samp_id}.pkl', 'wb') as pkl_file:
            pickle.dump([flux_tag, flux, surf_id, face_id, samp_len, samp_id, status], pkl_file)
        # with connect(f'Surface_{surf_id}/rotd.db', timeout=60) as cursor:
        #     try:
        #         cursor.execute('UPDATE fluxes SET status = :status WHERE '
        #                     'surf_id=:surf_id? AND face_id=:face_id? AND samp_id=:samp_id',
        #                     {'status': status, 'surf_id': surf_id,
        #                         'samp_id': samp_id})
        #     except OperationalError:
        #         for i in range(3):
        #             try:
        #                 time.sleep(0.2)
        #                 cursor.execute('UPDATE fluxes SET status = :status WHERE '
        #                             'surf_id=:surf_id? AND face_id=:face_id? AND samp_id=:samp_id',
        #                             {'status': status, 'surf_id': surf_id,
        #                                 'samp_id': samp_id})
        #                 break
        #             except OperationalError:
        #                 pass

    def del_db_job(self, job):
        _0, _1, surf_id, face_id, _4, samp_id, _6 = job
        os.remove(f'Surface_{surf_id}/jobs/surf{surf_id}_face{face_id}_samp{samp_id}.pkl')
        # with connect(f'Surface_{surf_id}/rotd.db', timeout=60) as cursor:
        #     sql_cmd = 'DELETE FROM fluxes WHERE surf_id=? AND face_id=? AND samp_id=?'
        #     try:
        #         cursor.execute(sql_cmd, (surf_id, face_id, samp_id))
        #     except OperationalError:
        #         for i in range(3):
        #             try:
        #                 time.sleep(0.2)
        #                 cursor.execute(sql_cmd, (surf_id, face_id, samp_id))
        #                 break
        #             except OperationalError:
        #                 pass 