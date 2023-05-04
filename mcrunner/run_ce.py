import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ase.io import read
from ase.visualize import view

from copy import deepcopy as cp

from icet import ClusterSpace, StructureContainer, ClusterExpansion
from icet.tools import ConvexHull, enumerate_structures

from trainstation import CrossValidationEstimator


class ClusterExpansionCreator():

    def __init__(self, **kwargs):

        self._plot = kwargs.get('plot', True)
        self._verbose = kwargs.get("verbose", True)


    def get_CE(self, filename, cutoffs, **kwargs):
        """
        Creates the ClusterExpansion instance

        Parameters
        ==========
        filename:   str
            ase readable file containing the 2d-adsorption patterns, with
            empty sites being filled with the filler atom (e.g. 'X')
            The 0th structure must be the clean slab.
        cutoffs:    list of floats
            The list of cutoff distances in angstrom, respectively:
            [2b, 3b, 4b...]
        **kwargs
        ads_species:    list of strings
        energy_intro:   str
            name of the entry in the atoms.info dictionary containing the
            relevant information for the ICET code
        """
        self._struc_file    = filename
        self.ads_species   = kwargs.pop("ads_species")
        self.strucs         = read(self._struc_file, ":")

        # Define the clean slab to get the site array
        self.clean_slab     = cp(self.strucs[0])
        self.clean_slab.pbc = [True]*3

        # Get the cutoffs, and parse information of the type of ClusterExpansion
        self.cutoffs = cutoffs
        if len(cutoffs) == 1:
            self.interaction = '2b'
        elif len(cutoffs) == 2:
            self.interaction = '3b'
        elif len(cutoffs) == 3:
            self.interaction = '4b'
        else:
            raise ValueError("Do not understand more than 3 interaction types")

        # Define the cluster_space for the ClusterExpansion
        chemical_symbols = [self.ads_species for i in self.clean_slab]
        self.cluster_space = ClusterSpace(
            structure=self.clean_slab, cutoffs=self.cutoffs,
            chemical_symbols=chemical_symbols,
            symprec = kwargs.pop("symprec", 0.05)
        )
        if self._verbose:
            print(self.cluster_space)

        # Create the StructureContainer, it is necessary to weight the clean
        # slab quite immensly, otherwise the fit will disregard low coverage
        # energies
        self.sc = StructureContainer(cluster_space=self.cluster_space)

        struc_list = [self.clean_slab] * kwargs.pop("clean_weight", 1000)
        struc_list += self.strucs

        e_kwarg = kwargs.get("energy_intro", "ICET_E")
        self.e_kwarg = e_kwarg

        for struc in struc_list:
            self.sc.add_structure(
                structure=struc,
                properties = {'per_site_energy': struc.info[e_kwarg]}
            )

        # Optimize the CE parameters via CrossValidation
        opt = CrossValidationEstimator(
            fit_data = self.sc.get_fit_data(key='per_site_energy'),
            **kwargs.get("CV_kwargs", {})
        )
        opt.train()
        if self._verbose:
            print("OPTIMIZED CROSSVALIDATION:\n")
            print(opt)

        # Create the ClusterExpansion
        self.ce = ClusterExpansion(
            cluster_space=self.cluster_space,
            parameters=opt.parameters, metadata=opt.summary
        )
        if self._verbose:
            print("CREATED CLUSTER EXPANSION: \n")
            print(self.ce)
        return None


    def write_CE(self, filename):
        if self._verbose:
            print("WRITE CLUSTER EXPANSION TO: ", filename)
        self.ce.write(filename)


    def plot_energetics(self, figname = False, ads = 'default'):
        if ads == 'default':
            ads = self.ads_species[0]

        ## Check quality of CE, plot prediction vs reference
        data = {'cov': [], 'e_ref': [], 'e_pred': []}

        for at in self.strucs:
            cov = at.get_chemical_symbols().count(ads) / len(at)
            if cov != 0:
                # to deal with divided by 0
                data['cov'].append(cov)
                # the factor of 1e3 serves to convert from eV/atom to meV/atom
                data['e_ref'].append(at.info[self.e_kwarg]/cov)
                data['e_pred'].append(self.ce.predict(at)/cov)

        # step 2: Plot results
        fig, ax = plt.subplots()
        ax.set_xlabel(f'{ads} coverage (ML)')
        ax.set_ylabel(r'$\rm E_{per~site}~(meV)$')
        ax.set_xlim(min(data['cov'])-0.05, max(data['cov'])+0.05)
        ax.scatter(data['cov'], data['e_ref'], marker='o', label='ref.')
        ax.scatter(data['cov'], data['e_pred'], marker='x', label='CE pred.')

        plt.legend()
        fig.tight_layout()
        if figname:
            fig.savefig(figname, transparent=True)
        #plt.show()
        return None


    def view_hull_in_size_range(self, sizerange=[8], figname=False, ads='default'):

        if ads == 'default':
            ads = self.ads_species[0]

        cluster_space           = self.ce.get_cluster_space_copy()
        chemical_symbols        = cluster_space.chemical_symbols
        primitive_structure     = cluster_space.primitive_structure

        ## .pbc =[True, True, False] should keep the sampled supercells in the
        ## in plane directions. To be checked
        primitive_structure.pbc = [True, True, False]

        # step 1: Predict energies for enumerated structures
        data = {'cov': [], 'e_pred': []}
        atoms = []

        # Here we build different different unit cells
        for at in enumerate_structures(
                structure=primitive_structure, sizes=sizerange,
                chemical_symbols=chemical_symbols):
            cov = at.get_chemical_symbols().count(ads) / len(at)
            data['cov'].append(cov)
            data['e_pred'].append(self.ce.predict(at))
            atoms.append(at)
        if self._verbose:
            print('Predicted energies for {} structures'.format(len(atoms)))

        # step 2: Construct convex hull
        hull = ConvexHull(data['cov'], data['e_pred'])

        # step 3: Plot the results
        fig, ax = plt.subplots()
        ax.set_xlabel(f'{ads} coverage (ML)')
        ax.set_ylabel(r'$\rm E_{per~site}~(meV)$')
        ax.set_xlim(min(data['cov'])-0.05, max(data['cov'])+0.05)
        ax.scatter(data['cov'], data['e_pred'], marker='x')
        ax.plot(hull.concentrations, hull.energies, '-o')
        fig.tight_layout()
        if figname:
            fig.savefig(figname, transparent=True)

        return hull, atoms, data


def view_hull(CE, cluster_style, sizerange = [4*4]):
    hull, structures, data = CE.view_hull_in_size_range(
            sizerange = sizerange, cluster_style=cluster_style
    )

    tol = 0.0000001
    low_energy_structures = hull.extract_low_energy_structures(
        data['coverage'], data['per_site_energy'], tol)
    print('Found {} structures within {} meV/atom of the convex hull'.
          format(len(low_energy_structures),  tol))

    cl = list(set(data['coverage']))
    done = []
    for i in low_energy_structures:
        conc = data['coverage'][i]
        if conc not in done:
            structure = structures[i]
            if np.abs(np.linalg.norm(structure.cell[0]) - np.linalg.norm(structure.cell[1])) < 0.01:
                view(structure*(3,3,1))
                done.append(conc)
        if sorted(done) == sorted(cl):
            break