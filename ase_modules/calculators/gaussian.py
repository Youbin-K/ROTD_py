import os
import copy
from collections.abc import Iterable
from shutil import which
from typing import Dict, Optional

from ase.io import read, write
from ase.calculators.calculator import FileIOCalculator, EnvironmentError


class GaussianDynamics:
    calctype = 'optimizer'
    delete = ['force']
    keyword: Optional[str] = None
    special_keywords: Dict[str, str] = dict()

    def __init__(self, atoms, calc=None):
        self.atoms = atoms
        if calc is not None:
            self.calc = calc
        else:
            if self.atoms.calc is None:
                raise ValueError("{} requires a valid Gaussian calculator "
                                 "object!".format(self.__class__.__name__))

            self.calc = self.atoms.calc

    def todict(self):
        return {'type': self.calctype,
                'optimizer': self.__class__.__name__}

    def delete_keywords(self, kwargs):
        """removes list of keywords (delete) from kwargs"""
        for d in self.delete:
            kwargs.pop(d, None)

    def set_keywords(self, kwargs):
        args = kwargs.pop(self.keyword, [])
        if isinstance(args, str):
            args = [args]
        elif isinstance(args, Iterable):
            args = list(args)

        for key, template in self.special_keywords.items():
            if key in kwargs:
                val = kwargs.pop(key)
                args.append(template.format(val))

        kwargs[self.keyword] = args

    def run(self, **kwargs):
        calc_old = self.atoms.calc
        params_old = copy.deepcopy(self.calc.parameters)

        self.delete_keywords(kwargs)
        self.delete_keywords(self.calc.parameters)
        self.set_keywords(kwargs)

        self.calc.set(**kwargs)
        self.atoms.calc = self.calc

        try:
            self.atoms.get_potential_energy()
        except OSError:
            converged = False
        else:
            converged = True

        atoms = read(self.calc.label + '.log')
        self.atoms.cell = atoms.cell
        self.atoms.positions = atoms.positions

        self.calc.parameters = params_old
        self.calc.reset()
        if calc_old is not None:
            self.atoms.calc = calc_old

        return converged


class GaussianOptimizer(GaussianDynamics):
    keyword = 'opt'
    special_keywords = {
        'fmax': '{}',
        'steps': 'maxcycle={}',
    }


class GaussianIRC(GaussianDynamics):
    keyword = 'irc'
    special_keywords = {
        'direction': '{}',
        'steps': 'maxpoints={}',
    }


class Gaussian(FileIOCalculator):
    implemented_properties = ['energy', 'forces', 'dipole']
    command = 'GAUSSIAN < PREFIX.com > PREFIX.log'
    discard_results_on_any_change = True

    def __init__(self, *args, label='Gaussian', **kwargs):
        FileIOCalculator.__init__(self, *args, label=label, **kwargs)

    def calculate(self, *args, **kwargs):
        gaussians = ('g16', 'g09', 'g03')
        if 'GAUSSIAN' in self.command:
            for gau in gaussians:
                if which(gau):
                    self.command = self.command.replace('GAUSSIAN', gau)
                    break
            else:
                raise EnvironmentError('Missing Gaussian executable {}'
                                       .format(gaussians))

        FileIOCalculator.calculate(self, *args, **kwargs)

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        write(self.label + '.com', atoms, properties=properties,
              format='gaussian-in', parallel=False, **self.parameters)
        if 'fragment1' in self.parameters:
            f1_indexes = [int(idx) for idx in self.parameters['fragment1'].split(';')[0][1:-1].split(',')]
            f1_charge = int(self.parameters['fragment1'].split(';')[1])
            f1_mult = int(self.parameters['fragment1'].split(';')[2])
            f2_indexes = [int(idx) for idx in self.parameters['fragment2'].split(';')[0][1:-1].split(',')]
            f2_charge = int(self.parameters['fragment2'].split(';')[1])
            f2_mult = int(self.parameters['fragment2'].split(';')[2])
            start = 100000
            with open(f'{self.label}.com', 'r') as f:
                lines = f.readlines()
            for index, line in enumerate(lines):
                if 'fragment1' in line:
                    lines.pop(index)
                    lines[index] = "Counterpoise=2\n"
                if 'Gaussian input prepared by ASE' in line:
                    start = index + 2
                if index == start:
                    lines[index] = f"{line[:-1].split()[0]},{line[:-1].split()[1]}" + f" {f1_charge},{f1_mult} {f2_charge},{f2_mult}" + "\n"
                if index > start:
                    if index-(start+1) in f1_indexes:
                        lines[index] = f"{line.split()[0]}" + "(fragment=1)" + f"{line.split(line.split()[0])[1]}"
                    elif index-(start+1) in f2_indexes:
                        lines[index] = f"{line.split()[0]}" + "(fragment=2)" + f"{line.split(line.split()[0])[1]}"
            
            with open(f'{self.label}.com', 'w') as f:
                for line in lines:
                    f.write(line)

    def read_results(self):
        output = read(self.label + '.log', format='gaussian-out')
        self.calc = output.calc
        self.results = output.calc.results

    # Method(s) defined in the old calculator, added here for
    # backwards compatibility
    def clean(self):
        for suffix in ['.com', '.chk', '.log']:
            try:
                os.remove(os.path.join(self.directory, self.label + suffix))
            except OSError:
                pass

    def get_version(self):
        raise NotImplementedError  # not sure how to do this yet
