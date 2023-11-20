import os
import time
import copy
from collections import OrderedDict
from sqlite3 import connect
import pickle
from subprocess import Popen, PIPE
import re

from rotd_py.flux.flux import MultiFlux
from rotd_py.system import FluxTag
from rotd_py.job_tpl import py_tpl_str

from rotd_py.config_log import config_log

class Multi(object):
    """
    This is the application Multi that has a lot of work to do
    so it gives work to do to its slaves until all the work is done
    """

    def __init__(self, fluxbase=None, dividing_surfaces=None, sample=None,
                 calculator=None):
        """Initialize the multi flux calculation.

        Parameters
        ----------
        fluxbase : FluxBase
            The essential information for a flux calculation.
        dividing_surfaces :
            A 1-D array, each item in array is a Surface object.
        sample : Sample
            The way of sampling that is going to be used in the flux calculation

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
            num_faces = self.dividing_surfaces[0].get_num_faces()
        for index, surf in enumerate(self.dividing_surfaces):
            surf.surf_id = str(index)
            indivi_sample = copy.deepcopy(sample)
            indivi_sample.set_dividing_surface(surf)
            self.total_flux[surf.surf_id] = MultiFlux(fluxbase=fluxbase,
                                                 num_faces=num_faces,
                                                 sample=indivi_sample,
                                                 calculator=self.calculator)
        self.surf_id_gen = (str(surf.surf_id) for surf in self.dividing_surfaces)
        self.ref_flux = copy.deepcopy(self.total_flux)
        with connect('rotd.db', timeout=60) as cursor:
            cursor.execute("CREATE TABLE IF NOT EXISTS fluxes "
                           "(flux_tag int, flux blob, surf_id float, "
                           "samp_len int, samp_id int, status text)")
        self.py_tpl_str = py_tpl_str
        if not os.path.isdir('jobs'):
            os.mkdir('jobs')
        with open('qu.tpl') as qu_tpl_fh:
            self.qu_tpl_str = qu_tpl_fh.read()
        if self.calculator['queue'] == 'slurm':
            self.sub_cmd = 'sbatch'
            self.chk_cmd = 'scontrol show jobid -d {job_id}'        
        
        global logger
        logger = config_log('rotdpy')

    def run(self):
        logger.info('Starting rotdpy')
        first_job = True

        """Keep submitting jobs as long as there is work to do"""
        num_surfaces = len(self.total_flux)
        curr_surf = next(self.surf_id_gen)
        curr_flux = self.total_flux[curr_surf]  # multiflux
        logger.info(f'Number of surfaces: {num_surfaces}')
        logger.info(f'Current surface: {curr_surf}')

        # initialize the calculation for the first dividing surface
        flux_idx = 1  # 1 indexed (why?)
        for i in range(0, curr_flux.num_faces):
            flux = copy.deepcopy(curr_flux.flux_array[i])
            if first_job:
                logger.info('The grids are:'
                            f'temperature (K) {flux.temp_grid}'
                            f'energy (cm-1) {flux.energy_grid}'
                            f'angular momentum (cm-1) {flux.angular_grid}')
                first_job = False
            # create the minimum number of flux jobs, i.e., flux.pot_min()
            for j in range(flux.pot_min()):
                logger.info(f'Creating {flux.pot_min()} jobs to run for facet [flux_idx] {flux_idx}.')
                self.work_queue.append((FluxTag.FLUX_TAG, flux,
                                        curr_surf, flux.samp_len(), 
                                        flux_idx, 'TO DO'))
                flux_idx += 1

        while self.work_queue or self.running_jobs:
            if self.work_queue:
                self.submit_work(self.work_queue.pop(0))
            self.check_running_jobs()
            for job in reversed(self.newly_finished_jobs):
                # update flux
                flux_tag, job_flux, surf_id, samp_len, samp_id, status = job
                logger.info(f'{flux_tag}, surf {surf_id}, samp len {samp_len}, sample {samp_id} {status}')
                if int(surf_id) < int(curr_surf):
                    continue
                face_index = job_flux.sample.div_surface.get_curr_face()
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

                logger.info(f'successful sampling {job_flux.acct_smp()}') # {job_flux.close_smp()} {job_flux.face_smp()} {job_flux.fail_smp()} \
                            #{job_flux.temp_sum} {job_flux.temp_var} {job_flux.e_sum} {job_flux.e_var}')

                # update the minimum energy and configuration
                for i in range(0, curr_flux.energy_size):
                    if job_flux.min_energy[i] < curr_flux.min_energy[i]:
                        #print("multi:%f" % (job_flux.min_energy[i]))
                        curr_flux.min_energy[i] = job_flux.min_energy[i]
                        curr_flux.min_geometry[i] = job_flux.min_geometry[i].copy()
                # check the current flux converged or not
                flux_tag, smp_info = curr_multi_flux.check_state()
                logger.info(f'tag {flux_tag} info {smp_info}')
                # if continue sample for current flux and for the face index:
                if flux_tag == FluxTag.FLUX_TAG:
                    #print(f'curr_surf {curr_surf}')
                    face_index = smp_info
                    flux = copy.deepcopy(self.ref_flux[surf_id].flux_array[face_index])
                    self.work_queue.append((flux_tag, flux, surf_id, flux.samp_len(), 
                                            flux_idx, 'TO DO'))
                    flux_idx += 1

                elif flux_tag == FluxTag.SURF_TAG:
                    smp_num = smp_info
                    for face_index in range(0, len(smp_num)):
                        if smp_num[face_index] != 0:
                            flux = copy.deepcopy(self.ref_flux[surf_id].flux_array[face_index])
                            self.work_queue.append((flux_tag, flux, surf_id,
                                                    smp_num[face_index], flux_idx, 'TO DO'))
                            flux_idx += 1

                            logger.info(f'{FluxTag.SURF_TAG} flux_idx {flux_idx} face {face_index} smp_num {smp_num} surface {surf_id}')

                elif flux_tag == FluxTag.STOP_TAG:
                    logger.info(f'{FluxTag.STOP_TAG} was assigned to surface {surf_id}')
                    self.total_flux[surf_id].save_file(surf_id)
                    self.work_queue = []
                    # initialize the calculation for the next dividing surfaces
                    if int(curr_surf) == num_surfaces:
                        logger.info('There is no more work to be done')
                        continue
                    else:
                        try:
                            curr_surf = next(self.surf_id_gen)
                            #surf_id = curr_surf
                        except StopIteration:
                            return curr_surf
                        curr_flux = self.total_flux[curr_surf]
                        for i in range(curr_flux.num_faces):
                            flux = copy.deepcopy(curr_flux.flux_array[i])
                            for j in range(flux.pot_min()):
                                self.work_queue.append((FluxTag.FLUX_TAG,
                                                        flux, curr_surf,
                                                        flux.samp_len(),
                                                        flux_idx, 'TO DO'))
                                flux_idx += 1
                else:
                    logger.warning('The flux tag is INVALID')
                    raise ValueError("The flux tag is INVALID")
                self.finished_jobs.append(job)
                self.newly_finished_jobs.remove(job)

            #time.sleep(0.5)
        return curr_surf

    def submit_work(self, job, procs=1):
        """

        Parameters
        ----------
        job

        Returns
        -------

        """

        #TODO: Find a better fix
        #if not isinstance(job[2], str):
        #    tmp_job = (copy.deepcopy(job[0]), copy.deepcopy(job[2]), copy.deepcopy(job[1]), copy.deepcopy(job[3]), copy.deepcopy(job[4]), copy.deepcopy(job[5]))
        #    job = copy.deepcopy(tmp_job)
        # Initial checks of job db/queue status and maximum jobs limit.
        job = self.check_job_status(job)
        flux_tag, flux, surf_id, samp_len, samp_id, status = job
        if status == 'NEWLY FINISHED':
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
        srl_flux = pickle.dumps(flux)  # Serialize the flux object
        int_flux_tag = flux_tag.value
        sql_cmd = "INSERT INTO fluxes VALUES (:flux_tag, :flux, :surf_id, " \
                  ":samp_len, :samp_id, :status)"
        with connect('rotd.db', timeout=60) as cursor:
            cursor.execute(sql_cmd, {'flux_tag': int_flux_tag,
                                     'flux': srl_flux,
                                     'surf_id': surf_id,
                                     'samp_len': samp_len,
                                     'samp_id': samp_id,
                                     'status': 'RUNNING'
                                     })

        # Launch the job
        init_dir = os.getcwd()
        os.chdir('jobs')
        with open(f'surf{surf_id}_samp{samp_id}.py', 'w') as py_job_fh:
            py_job_fh.write(self.py_tpl_str.format(surf_id=surf_id,
                                                   samp_id=samp_id))
        with open(f'surf{surf_id}_samp{samp_id}.sh', 'w') as qu_job_fh:
            qu_job_fh.write(self.qu_tpl_str.format(surf_id=surf_id,
                                                   samp_id=samp_id,
                                                   procs=procs,))
        stdout, stderr = Popen(f'{self.sub_cmd} surf{surf_id}_samp{samp_id}.sh',
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
        os.chdir(init_dir)

        # Relocate the job into the running list
        job = (flux_tag, flux, surf_id, samp_len, samp_id, 'RUNNING')
        self.running_jobs.append(job)



    def check_running_jobs(self):
        for job in reversed(self.running_jobs):
            updated_job = self.check_job_status(job)
            try:
                status = updated_job[5]
            except TypeError:
                while updated_job == None:
                    updated_job = self.check_job_status(job)
                    try:
                        status = updated_job[5]
                    except TypeError:
                        continue
            if status == 'RUNNING':
                continue
            elif status == 'FAILED':
                self.del_db_job(job)
                self.work_queue.append(updated_job)
                self.running_jobs.remove(job)
            elif status == 'NEWLY FINISHED':
                self.newly_finished_jobs.append(updated_job)
                self.running_jobs.remove(job)

    def check_job_status(self, job):
        flux_tag, flux, surf_id, samp_len, samp_id, status = job
        # Check if the job is in the database:
        with connect('rotd.db', timeout=60) as cursor:
            sql_cmd = 'SELECT * FROM fluxes WHERE surf_id=? AND samp_id=?'
            rows = cursor.execute(sql_cmd, (surf_id, samp_id)).fetchall()
        if rows:
            for db_job in rows:
                db_status = db_job[5]
                db_flux = pickle.loads(db_job[1])
                # Check if the data from the db is usable.
                if len(db_flux.energy_grid) != len(flux.energy_grid) \
                        or len(db_flux.temp_grid) != len(flux.temp_grid) \
                        or len(db_flux.angular_grid) != len(flux.angular_grid):
                    continue
                elif not (db_flux.energy_grid == flux.energy_grid).all() \
                        or not (db_flux.temp_grid == flux.temp_grid).all() \
                        or not (db_flux.angular_grid == flux.angular_grid).all():
                    continue
                else:
                    break
            else:
                if self.show_grid_warn:
                    logger.warning('The database entries have points calculated with '
                                   'different grids. Unable to use them with the current '
                                   'calculation.')
                    self.show_grid_warn = False
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
                                return flux_tag, flux, surf_id, samp_len, \
                                       samp_id, "RUNNING"
                            elif queue_status == 'COMPLETED':
                                return flux_tag, db_flux, surf_id, \
                                       samp_len, samp_id, "NEWLY FINISHED"
                            elif queue_status == 'FAILED':
                                return flux_tag, flux, surf_id, samp_len, \
                                       samp_id, "FAILED"
                            break
                else:
                    return flux_tag, flux, surf_id, samp_len, samp_id, "FAILED"
            else:
                return flux_tag, db_flux, surf_id, samp_len, samp_id, "NEWLY FINISHED"
        else:
            return flux_tag, flux, surf_id, samp_len, samp_id, "TO DO"

    def update_db_job_status(self, job, status):
        flux_tag, flux, surf_id, samp_len, samp_id, old_status = job
        with connect('rotd.db', timeout=60) as cursor:
            cursor.execute('UPDATE fluxes SET status = :status WHERE '
                           'surf_id=:surf_id? AND samp_id=:samp_id',
                           {'status': status, 'surf_id': surf_id,
                            'samp_id': samp_id})

    def del_db_job(self, job):
        _1, _2, surf_id, _3, samp_id, _4 = job
        with connect('rotd.db', timeout=60) as cursor:
            sql_cmd = 'DELETE FROM fluxes WHERE surf_id=? AND samp_id=?'
            cursor.execute(sql_cmd, (surf_id, samp_id))
