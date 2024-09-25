import os
import subprocess
import time
import re
from dataclasses import dataclass
import rotd_py.rotd_math as rotd_math
import math
from rotd_py.molpro.rotdpy_molpro_template import default_molpro_template


class Molpro:
    """
    Class to write and read molpro file and to run molpro
    """
    def __init__(self, label, mol, calculator):
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
        if os.path.isfile('../../molpro.tpl'):
            with open('../../molpro.tpl', 'r') as f:
                self.tpl = f.read()
        else:
            self.tpl = default_molpro_template
        self.name = label
        self.mol = mol
        self.calc = calculator
        self.scratch = calculator['scratch']
        self.procs = calculator['processors']

        self.nelectron = 0
        for n in self.mol.get_atomic_numbers():
            self.nelectron += n

        self.charge = 0
        for n in self.mol.get_initial_charges():
            self.charge += n

        self.nelectron -= self.charge

        if 'spin' in self.calc:
            self.spin = self.calc['spin']
        else:
            self.spin = self.nelectron % 2

        if 'mem' in self.calc:
            if isinstance(self.calc['mem'], int):
                #default unit is MW in calculator
                self.memory_in_MW = self.calc['mem']
            else:
                self.memory_in_MW = 500
        else:
            self.memory_in_MW = 500

        self.symm = 1
        

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
        
        if os.path.isfile('../../molpro.tpl'):
            with open(f'{self.name}.inp', 'w') as f:
                f.write(self.tpl.format(name=self.name,
                                        natom=len(geom),
                                        geom=xyz,
                                        mem=self.memory_in_MW
                                    ))
            return
        else:
            if 'method' not in self.calc or 'basis' not in self.calc:
                raise Exception("If the user does not have a molpro.tpl file in the input's folder, 'basis' and 'method' need to provided in the calculator.")
            options = "GPRINT,ORBITALS,ORBEN,CIVECTOR \nGTHRESH,energy=1.d-7 \nangstrom \n orient,noorient\n nosym"
            basis = f"basis = {self.calc['basis']}"
                
            method = ""

            mtd = regex_in(self.calc['method'])
            if mtd == r".*caspt2\([0-9]+,[0-9]+\)":
                if 'shift' in self.calc:
                    shift = self.calc['shift']
                else:
                    shift = 0.3
                active_electrons = int(self.calc['method'].split("caspt2(")[1].split(",")[0])
                active_orbitals = int(self.calc['method'].split("caspt2(")[1].split(",")[1][:-1])
                closed_orbitals = int(math.trunc(self.nelectron-active_electrons)/2)
                occ_obitals = closed_orbitals + active_orbitals
                method += " {multi,\n" + f" occ,{occ_obitals}\n closed,{closed_orbitals}\n" + " }\n\n"
                if self.spin == 0:
                    method += " {rs2c, shift=" + f"{shift}" + "}\n"
                else:
                    method += " {rs2, shift=" + f"{shift}" + "}\n"
            elif mtd == "ccsd\(t\)":
                method += " {ccsd(t)-f12}\n"
            elif mtd ==  "uwb97xd":
                method += " omega=0.2    !range-separation parameter\n"
                method += " srx=0.222036 !short-range exchange\n"
                method += " {grid,wcut=1d-30,min_nr=[175,250,250,250],max_nr=[175,250,250,250],min_L=[974,974,974,974],max_L=[974,974,974,974]}\n"
                method += " {int; ERFLERFC,mu=$omega,srfac=$srx}\n"
                method += " uks,HYB_GGA_XC_WB97X_D\n"
            elif mtd ==  "wb97xd":
                method += " omega=0.2    !range-separation parameter\n"
                method += " srx=0.222036 !short-range exchange\n"
                method += " {grid,wcut=1d-30,min_nr=[175,250,250,250],max_nr=[175,250,250,250],min_L=[974,974,974,974],max_L=[974,974,974,974]}\n"
                method += " {int; ERFLERFC,mu=$omega,srfac=$srx}\n"
                method += " ks,HYB_GGA_XC_WB97X_D\n"
            elif mtd ==  "ub3lyp":
                method += " {grid,wcut=1d-30,min_nr=[175,250,250,250],max_nr=[175,250,250,250],min_L=[974,974,974,974],max_L=[974,974,974,974]}\n"
                method += " uks,HYB_GGA_XC_B3LYP\n"
            elif mtd ==  "b3lyp":
                method += " {grid,wcut=1d-30,min_nr=[175,250,250,250],max_nr=[175,250,250,250],min_L=[974,974,974,974],max_L=[974,974,974,974]}\n"
                method += " ks,HYB_GGA_XC_B3LYP\n"
            else:
                raise NotImplementedError

            with open(f'{self.name}.inp', 'w') as f:
                f.write(self.tpl.format(name=self.name,
                                        natom=len(geom),
                                        geom=xyz,
                                        basis=basis,
                                        method=method,
                                        options=options,
                                        mem=self.memory_in_MW
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
                        return None
                        #return 1.*rotd_math.Hartree
            # time.sleep(0.5)
            
    def run(self) -> None:
        """
        Submit the molpro job to a slurm queue. 
        It requires a slurm template to be provided, called
        slurm.tpl
        in the working directory.
        The file should be hard-wired, with a single core to be used
        on a single node.
        Only field to fill is the name of the file with {name}
        """
        # command = f'molpro -d {self.scratch} -M{self.memory_in_MW} -n {self.procs} {self.name}.inp'

        command = f'export OMP_NUM_THREAD={self.procs}\n molpro -d {self.scratch} -t {self.procs} -n {self.procs} {self.name}.inp'
        process = subprocess.Popen(command, shell=True,
                                   stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, stderr = (std.decode() for std in process.communicate())

@dataclass
class regex_in:
    string: str

    def __eq__(self, other):#Python 3.10 only: str | re.Pattern):
        if isinstance(other, str):
            other = re.compile(other)
        assert isinstance(other, re.Pattern)
        # TODO extend for search and match variants
        return other.fullmatch(self.string) is not None
