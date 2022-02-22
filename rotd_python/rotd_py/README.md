# rotd_py

This is a project aim to implement the variable coordinate transition state theory in Python.

Fragment is the class used for defining the fragment in reactants. This is a class inherits from the class Atoms in ASE,
which makes the convenience of building and manipulating the molecule by directly calling already built-in features in
`Atom` such as calculating properties of the molecule and rotating or translating it. One different feature
we're holding here for VRC-TST is a reference frame of the fragment which makes the random rotating the fragment
consistently. In order to capture different parameters for calculation for different types of molecule, there are three
different subclasses--monoatomic,linear and nonlinear--in `Fragment` (slab could also be added if one interests).
One fragment can be initialized as follows:

```
from rotd_py.fragment.nonlinear import Nonlinear

ch3_1 = Nonlinear('CH3', positions=[[-0.0, .0, 0.0],
		[1.0788619988,0.0000000000,0.0000000000],
		[-0.5394309994,-0.9343218982,0.0000000000],
		[-0.5394309994,0.9343218982,0.0000000000]])

ch3_2 = Nonlinear('CH3', positions=[[-0.0, .0, 0.0],
		[1.0788619988,0.0000000000,0.0000000000],
		[-0.5394309994,-0.9343218982,0.0000000000],
		[-0.5394309994,0.9343218982,0.0000000000]])
```

which set up the fragments used for two \ce{CH3} recombination reaction. \\

Surface is the class used for setting up the diving surface and supporting multifaceted surfaces. The numbers and
positions of pivot points are represented using a 2-D array with a key of '0' or '1' to connect to the index of
fragment. An example of setting up one dividing surface is shown as:

```
from rotd_py.system import Surface

surface_1 = Surface({'0':np.array([[0.0, 0.0, 1.0]]),
		    '1':np.array([[0.0,0.0,0.5]])},
		    distances=np.array([[13.0]]))
```

One thing should be mentioned here is the coordinate of pivot point is set relative to the center of mass of the
molecule. Thus, if the reference atom for pivot point does not sit in the center of mass, the coordinate of that atom in
the reference frame should be considered while setting up the dividing surface. \\

Sample takes care of the generating random configuration based on reactants and dividing surface. By being initialized
with a calculator (can be referred to calculators supported by ASE), the `Sample` class also does the energy
calculation for the generated configuration. The energy then will be used in the flux calculation for evaluating the E-J
resolved number of states. An illustration of using this class is given as:

```
from rotd_py.sample.multisample import MultiSample
from ase.calculators.abinit import Abinit

calc = Abinit(...)
r_inf = -79.47370538 # This is the infinite seperation energy for the reactants

ch3_sample = MultiSample(fragments=[ch3_1, ch3_2],
    dividing_surface=surface_1, calculator=calc, inf_energy=r_inf, energy_size=1)

```

The `MultiSample` is the sampling class used for multifaceted dividing surface. One can also add a sampling
schema for surface reaction with constraint that only sample to one direction of the surface is valid. \\

`Flux` calculate the E-J resolved one-way flux based on customized convergence criteria. The user should also
set up the temperature, energy, and angular momentum range for the simulation in `FluxBase` as follows:

```
from rotd_py.flux.fluxbase import FluxBase
 flux_parameter={'pot_smp_max':100, 'pot_smp_min': 50,
             'tot_smp_max' : 10000, 'tot_smp_min' :50,
             'flux_rel_err' : 10.0, 'smp_len' : 1}

 flux_base = FluxBase(temp_grid = temperature,
                  energy_grid = energy,
                  angular_grid = angular_mom,
                 flux_type = 'MICROCANNONICAL',
                 flux_parameter=flux_parameter)

Then the calculation as a whole will be integrated by \textbf{Multi}. In \textit{Multi}, the communication among the master and slaves are achieved using \textbf{Mpi4py} package. The master is designed to assign work to slaves based on the termination criteria set by the user. An array of \textit{Flux} is maintained such that a simulation with multiple multifaceted dividing surfaces can be executed automatically. After setting up the sample schema and flux parameters, a VRC-TST calculation can be invoked by the following codes:

from rotd_py.multi import Multi

multi = Multi(sample=ch3_sample, dividing_surfaces=[surface_1], fluxbase=flux_base)
multi.run()
```
