
import os

from mpi4py import MPI
from mpi_master_slave import Master, Slave
from mpi_master_slave import WorkQueue
import time
import copy
from collections import OrderedDict
from rotd_py.flux.flux import MultiFlux
from rotd_py.system import FluxTag


class multi_master():
    """This is the class for describing the work of the master,
    including, accumulate calculation results from slaves and adding new
    task into the Work queue.

    Parameters
    ----------
    slaves : Slaves in mpi_master_slave
    total_flux : dict
        A dictionary of MultiFlux, each index represents each dividing surface.

    Attributes
    ----------
    master : Master in mpi_master_slave
    work_queue : WordQueue in master_slave
    ref_flux : MultiFlux, for initializing calculation
    total_flux

    """

    def __init__(self, slaves, total_flux):
        self.master = Master(slaves)
        self.work_queue = WorkQueue(self.master)
        self.total_flux = total_flux
        self.ref_flux = copy.deepcopy(total_flux)

    def terminate_slaves(self):
        """
        Call this to make all slaves exit their run loop
        """
        self.master.terminate_slaves()

    def run(self):
        """
        This is the core of  application, keep starting slaves
        as long as there is work to do
        """
        # Here the number of tasks is the initial number of tasks saved in the
        # working queue.

        #print ("asdf", self.total_flux)
        num_surfaces = len(self.total_flux)
        curr_surf = 0
        curr_flux = self.total_flux[str(curr_surf)]  # multiflux
        #print ("self.total_flux, in multi_master", self.total_flux)

        #print ("curr_flux:", curr_flux)

        # initialize the calculation for the first dividing surface
        
        # 진짜 딱 1번만 불림 Multi master는 처음에만 불리는 function   
        for i in range(0, curr_flux.num_faces):
            flux = copy.deepcopy(curr_flux.flux_array[i])
            #print ("fluxxxx", flux)

            for j in range(flux.pot_min()):
                self.work_queue.add_work(data=(FluxTag.FLUX_TAG, flux,
                                               curr_surf, flux.samp_len()))

        #
        # while we have work to do and not all slaves completed
        #
        while not self.work_queue.done():

            #
            # give work to do to each idle slave
            #
            self.work_queue.do_work()
            for slave_flux, sid in self.work_queue.get_completed_work():
                # update flux

                #print ("self.work_queue", self.work_queue.get_completed_work) # Not helpful
                #print ("This is slave_flux", slave_flux) # Dict?
                #print ("sid, j ", sid) # Always 0

                if sid < curr_surf:
                    continue
                face_index = slave_flux.sample.div_surface.get_curr_face()
                curr_multi_flux = self.total_flux[str(sid)]
                curr_flux = curr_multi_flux.flux_array[face_index]  # multiflux

                print("surface_index%d" % (sid)) # Should keep this turned on(original)
                #print ("test", slave_flux.acct_smp())

                curr_flux.add_acct_smp(slave_flux.acct_smp())
                curr_flux.add_close_smp(slave_flux.close_smp())
                curr_flux.add_face_smp(slave_flux.face_smp())
                curr_flux.add_fail_smp(slave_flux.fail_smp())
                curr_flux.temp_sum += slave_flux.temp_sum
                curr_flux.temp_var += slave_flux.temp_var
                curr_flux.e_sum += slave_flux.e_sum

                #print ("multi.py curr_flux.e_sum: ", curr_flux.e_sum) # all added value of self.e_sum[en_ind][pes] is printed with this

                curr_flux.e_var += slave_flux.e_var
                curr_flux.ej_sum += slave_flux.ej_sum
                curr_flux.ej_var += slave_flux.ej_var

                # update the minimum energy and configuration
                for i in range(0, curr_flux.energy_size):
                    if slave_flux.min_energy[i] < curr_flux.min_energy[i]:

                        # This updates the minimum energy obtained from flux.py
                        print("multi:%f" % (slave_flux.min_energy[i]), "unit in Hartree")

                        curr_flux.min_energy[i] = slave_flux.min_energy[i]
                        curr_flux.min_geometry[i] = slave_flux.min_geometry[i].copy()
                # check the current flux converged or not
                flux_tag, smp_info = curr_multi_flux.check_state()
                # if continue sample for current flux and for the face index:
                if flux_tag == FluxTag.FLUX_TAG:
                    face_index = smp_info
                    flux = copy.deepcopy(self.ref_flux[str(sid)].flux_array[face_index])

                    #print ("FLUX_TAG", flux)
                    self.work_queue.add_work(data=(flux_tag, flux, sid,
                                                   flux.samp_len()))

                elif flux_tag == FluxTag.SURF_TAG:
                    smp_num = smp_info
                    print("SURFACE")
                    for face_index in range(0, len(smp_num)):
                        if smp_num[i] != 0:
                            flux = copy.deepcopy(self.ref_flux[str(sid)].flux_array[face_index])

                            print ("SURF_TAG", flux)
                            self.work_queue.add_work(data=(flux_tag, sid, flux,
                                                           smp_num[i]))

                elif flux_tag == FluxTag.STOP_TAG:
                    self.total_flux[str(sid)].save_file(sid)
                    self.work_queue.empty_work_queue()
                    # initialize the calculation for the next dividing surfaces
                    if curr_surf == num_surfaces:
                        print("There is no work to be done")
                        continue
                    else:
                        curr_surf += 1
                        curr_flux = self.total_flux[str(curr_surf)]
                        for i in range(curr_flux.num_faces):
                            flux = copy.deepcopy(curr_flux.flux_array[i])

                            print ("STOP_TAG", flux)

                            for j in range(flux.pot_min()):
                                self.work_queue.add_work(data=(FluxTag.FLUX_TAG,
                                                               flux, curr_surf,
                                                               flux.samp_len()))

                else:
                    raise ValueError("The flux tag is INVALID")

            #time.sleep(0.3)
        return curr_surf


class MySlave(Slave):

    """
    A slave process extends Slave class, overrides the 'do_work' method
    and calls 'Slave.run'. The Master will do the rest
    """

    def __init__(self):
        super(MySlave, self).__init__()

    def do_work(self, data):
        rank = MPI.COMM_WORLD.Get_rank()
        name = MPI.Get_processor_name()
        # calculate the matrix value you want it to calculate
        # here data is input value x, should be a float number in (0,1)
        # the length of the matrix
        flux_tag, flux, sid, samp_len = data
        if flux_tag == FluxTag.FLUX_TAG:
            flux.run(samp_len)
        elif flux_tag == FluxTag.SURF_TAG:
            flux.run_surf(samp_len)
        elif flux_tag == FluxTag.STOP_TAG:
            pass
        else:
            raise ValueError("The communication tas is INVALID")

        print('  Slave %s rank %d executing ' % (name, rank))
        return (flux, sid)


class Multi(object):
    """
    This is the application Multi that has a lot of work to do
    so it gives work to do to its slaves until all the work is done
    """


    def __init__(self, fluxbase=None, dividing_surfaces=None, sample=None,
                 calculator=None, max_jobs=float('inf')):

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


        num_faces = 0
        if self.dividing_surfaces is not None:
            num_faces = self.dividing_surfaces[0].get_num_faces()
        for i in range(0, len(self.dividing_surfaces)):
            surf_index = str(i)
            indivi_sample = copy.deepcopy(sample)
            indivi_sample.set_dividing_surface(self.dividing_surfaces[i])
            self.total_flux[surf_index] = MultiFlux(fluxbase=fluxbase,
                                                    num_faces=num_faces,
                                                    sample=indivi_sample,
                                                    calculator=calculator)
        self.slave = MySlave()

        if calculator == 'molpro':
            try:
                os.mkdir('scratch')
            except FileExistsError:
                pass


    def run(self):
        # start the mpi run
        name = MPI.Get_processor_name()
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()

        print('I am  %s rank %d (total %d)' % (name, rank, size))

        #print (size) # total %d prints the size
        try:
            if rank == 0:  # Master

                app = multi_master(slaves=range(1, size), total_flux=self.total_flux)
                app.run()
                # save the last flux
                app.terminate_slaves()
            elif rank ==1:
                self.slave.run() 
            else:
                print ("rank is not 1")
                #self.slave.run()

        except KeyError:
            MPI.COMM_WORLD.Abort(1)
        print('Task complete! (rank %d' % rank)
