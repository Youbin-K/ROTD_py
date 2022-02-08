from rotd_py.fragment.nonlinear import Nonlinear
from rotd_py.system import Surface
from rotd_py.multi import Multi
from rotd_py.sample.multi_sample import MultiSample
from rotd_py.flux.fluxbase import FluxBase

import numpy as np
from amp import Amp

def generate_grid(start, interval, factor, num_point):
    #return the grid needed for simulation
    #interval += interval * factor
    #start += interval
    #return an numpy array with length of num_point
    i = 1
    grid = [start]
    for i in range(1, num_point):
        start += interval
        grid.append(start)
        interval = interval*factor
    return np.array(grid)

# temperature, energy grid and angular momentum grid
temperature = generate_grid(10, 10, 1.05, 51)
energy = generate_grid(0, 10, 1.05, 169)
angular_mom = generate_grid(0, 1, 1.1, 40)

print(temperature)
#fragment info
ch3_1 = Nonlinear('CH3', positions=[[-0.0, .0, 0.0],
                                 [1.0788619988,0.0000000000,0.0000000000],
                                [-0.5394309994,-0.9343218982,0.0000000000],
                                [-0.5394309994,0.9343218982,0.0000000000]])

ch3_2 = Nonlinear('CH3', positions=[[-0.0, .0, 0.0],
                                 [1.0788619988,0.0000000000,0.0000000000],
                                [-0.5394309994,-0.9343218982,0.0000000000],
                                [-0.5394309994,0.9343218982,0.0000000000]])

# Setting the dividing surfaces
# This creates an output file, as much as the given dividing surface.
# The output file contains temperature and the flux based on the temperature.
# 
divid_surf = [Surface({'0':np.array([[0.0,0.0,1.0],
                                      [0.0,0.0,-1.0]]),
                       '1':np.array([[0.0,0.0,1.0],
                                    [0.0,0.0,-1.0]])},
                        distances=np.array([[6.5,6.5],
                                            [6.5,6.5]])),
            Surface({'0':np.array([[0.0,0.0,1.0],
                                      [0.0,0.0,-1.0]]),
                       '1':np.array([[0.0,0.0,1.0],
                                    [0.0,0.0,-1.0]])},
                        distances=np.array([[6.0,6.0],
                                            [6.0,6.0]])),
            Surface({'0':np.array([[0.0,0.0,1.0],
                                      [0.0,0.0,-1.0]]),
                       '1':np.array([[0.0,0.0,1.0],
                                    [0.0,0.0,-1.0]])},
                        distances=np.array([[5.5,5.5],
                                            [5.5,5.5]])),
            Surface({'0':np.array([[0.0,0.0,1.0],
                                      [0.0,0.0,-1.0]]),
                       '1':np.array([[0.0,0.0,1.0],
                                    [0.0,0.0,-1.0]])},
                        distances=np.array([[5.0,5.0],
                                            [5.0,5.0]])),
            Surface({'0':np.array([[0.0,0.0,1.0],
                                      [0.0,0.0,-1.0]]),
                       '1':np.array([[0.0,0.0,1.0],
                                    [0.0,0.0,-1.0]])},
                        distances=np.array([[4.5,4.5],
                                            [4.5,4.5]])),
            Surface({'0':np.array([[0.0,0.0,1.0],
                                      [0.0,0.0,-1.0]]),
                       '1':np.array([[0.0,0.0,1.0],
                                    [0.0,0.0,-1.0]])},
                        distances=np.array([[4.0,4.0],
                                            [4.0,4.0]])),
            Surface({'0':np.array([[0.0,0.0,1.0],
                                      [0.0,0.0,-1.0]]),
                       '1':np.array([[0.0,0.0,1.0],
                                    [0.0,0.0,-1.0]])},
                        distances=np.array([[3.5,3.5],
                                            [3.5,3.5]])),
            Surface({'0':np.array([[0.0,0.0,1.0],
                                      [0.0,0.0,-1.0]]),
                       '1':np.array([[0.0,0.0,1.0],
                                    [0.0,0.0,-1.0]])},
                        distances=np.array([[3.0,3.0],
                                            [3.0,3.0]]))]

#divid_surf = [Surface({'0':np.array([[0.0,0.0,1.0],
#                                    [0.0, 0.0, -1.0]]),
#                       '1':np.array([[0.0,0.0,0.5]])},
#                        distances=np.array([[13.0],[ 12.5]]))]

#how to sample the two fragments
calc = 'amp.amp'

r_inf= -79.47971696 #RS2/cc-pvtz
ch3_sample = MultiSample(fragments=[ch3_1, ch3_2],
                inf_energy=r_inf, energy_size=1, min_fragments_distance=2.1)

#the flux info
flux_parameter={'pot_smp_max':2000, 'pot_smp_min': 100,
            'tot_smp_max' : 10000, 'tot_smp_min' :50,
            'flux_rel_err' : 10.0, 'smp_len' : 1}

flux_base = FluxBase(temp_grid = temperature,
                 energy_grid = energy,
                 angular_grid = angular_mom,
                flux_type = 'MICROCANNONICAL',
                flux_parameter=flux_parameter)

#start the final run
multi = Multi(sample=ch3_sample, dividing_surfaces=divid_surf, fluxbase=flux_base, calculator=calc)
multi.run()
#multi.total_flux['0'].flux_array[0].run(50)
#multi.total_flux['0'].save_file(0)
print(multi.total_flux['0'].flux_array[0].acct_smp())
print(multi.total_flux['0'].flux_array[0].fail_smp())
print(multi.total_flux['0'].flux_array[0].face_smp())
print(multi.total_flux['0'].flux_array[0].close_smp())

print(multi.total_flux['1'].flux_array[0].acct_smp())
print(multi.total_flux['1'].flux_array[0].fail_smp())
print(multi.total_flux['1'].flux_array[0].face_smp())
print(multi.total_flux['1'].flux_array[0].close_smp())

