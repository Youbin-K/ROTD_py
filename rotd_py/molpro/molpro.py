import os
import subprocess
import time
import rotd_py.rotd_math as rotd_math


class Molpro:
    """
    Class to write and read molpro file and to run molpro
    """
    def __init__(self, label, mol, scratch, procs):
        """
        The user-supplied molpro template file, molpro.tpl has 
        the following fields:
        {name}: the unique name of the calculation, same as the
                file name
        {natom}: number of atoms
        {geom}: geometry
        It is the user's responsibility to hardwire properly everything else, 
        for instance: number of electrons, symmetry, spin, charge, method, 
        basis set, active space, etc.
        Furthermore, it is compulsory to have a line that reads:
        myenergy = energy
        The value to which myenergy is set to will be read from the file
        and sent back to sample.py

        """
        with open('../../molpro.tpl', 'r') as f:
            self.tpl = f.read()
        self.name = label
        self.mol = mol
        self.scratch = scratch
        self.procs = procs

    def create_input(self):
        """
        Create the input for molpro based on the template called 
        molpro.tpl
        located in the working directory
        mol: ASE Atoms object of the structure
        label: the unique part of the file name
        """

        geom = self.mol.get_positions()
        atoms = self.mol.get_chemical_symbols()
        xyz = ''
        for ati, pos in enumerate(geom):
            x, y, z = pos
            xyz += '{} {:.8f} {:.8f} {:.8f}\n'.format(atoms[ati], x, y, z)

        with open(f'{self.name}.inp', 'w') as f:
            f.write(self.tpl.format(name=self.name,
                                    natom=len(geom),
                                    geom=xyz,
                                ))
        return 

    def read_energy(self):
        """
        Verify if the molpro calculation is done and read energy
        using the line:
        SETTING MYENERGY = ...
        Returns the energy in eV if successful.
        Returns +1 Hartree if failed.
        """
        
        while True:
            status = os.path.exists(f'{self.name}.out')
            if status:
                with open(f'{self.name}.out') as f:
                    lines = f.readlines()
                for index, line in enumerate(reversed(lines)):
                    if ('SETTING MYENERGY') in line:
                        return float(line.split()[3])*rotd_math.Hartree
                    elif ('ERROR') in line:
                        return 1.*rotd_math.Hartree
            #time.sleep(0.5)
            
    def run(self):
        """
        Submit the molpro job to a slurm queue. 
        It requires a slurm template to be provided, called
        slurm.tpl
        in the working directory.
        The file should be hard-wired, with a single core to be used
        on a single node.
        Only field to fill is the name of the file with {name}
        """
        command = f'molpro -d {self.scratch} -n {self.procs} {self.name}.inp'
        process = subprocess.Popen(command, shell=True,
                                   stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, stderr = (std.decode() for std in process.communicate())
