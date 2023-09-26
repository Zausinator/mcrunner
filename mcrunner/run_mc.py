import json, os, sys

import numpy as np
import pandas as pd

from ase.io import read, write
from ase.build import  make_supercell
from ase.visualize import view

from copy import deepcopy as cp

from icet import ClusterExpansion

from typing import Union

from .ensembles import SemiGrandCanonicalEnsemble_Diffsteps

from mchammer.calculators import ClusterExpansionCalculator

from multiprocessing import Pool



class MCRunner():

    def __init__(self):
        """
        """
        self._batch_kwargs = {}
        self._mc_args = {}
        return None

    def _update_mc_args(self):
        mc_arg_keys = ['init_type', 'ads_species', 'size', 'temperature', 
                       'steps', 'prob_threshold', 'from_read', 'strucname'
                       ]
        for key in self._batch_kwargs:
            if key in mc_arg_keys:
                self._mc_args[key] = self._batch_kwargs.pop(key)
        return None

    @property
    def mc_args(self):
        return self._mc_args
    
    @mc_args.setter
    def mc_args(self, kwargs : dict):
        self._mc_args.update(kwargs)
    
    @property
    def batch_kwargs(self):
        return self._batch_kwargs

    @batch_kwargs.setter
    def batch_kwargs(self, kwargs : dict):
        self._batch_kwargs.update(kwargs)
        self._update_mc_args()

    @staticmethod
    def build_initial_struc(CE,
                            size : int,
                            ads_species : list,
                            init_type : str
                            ):
        """
        Builds the initial ase structure from prompts given by the user

        Inputs 
        ======
        size: int
            Size of the square lattice in x and y dimension
        ads_species: list
            List of chemical symbols that can adsorb on the structure
            The order of the strings matter, last one should always be 'X'
        init_type: str
            Keyword that indicates how the structure should look
            Allowed keywords are: 'c2x2', 'bare', 'full', 'random'
        """
        # Get the primitive structure from the Cluster Expansion
        primstruct = CE.get_cluster_space_copy().primitive_structure

        structure = make_supercell(primstruct,
                        1 * np.array([[size, 0, 0],
                                        [0, size, 0],
                                        [0, 0, 1]]))
        n_species = len(ads_species)
        # Determines how the sites get occupied
        occ = []
        for x in range(size):
            for y in range(size):
                if init_type == 'c2x2':
                    tmp = (x+y) % 2
                    if tmp ==0:
                        occ.append(0)
                    else:
                        occ.append(1)
                elif init_type == 'bare':
                    occ.append(n_species-1)
                elif init_type == 'full':
                        occ.append(0)
                else:
                    occ.append(np.random.randint(n_species))

        structure.set_chemical_symbols([ads_species[o] for o in occ ])
        return structure
    
    @staticmethod
    def read_initial_struc(filename : str):
        """
        Gets the initial structure for the Monte Carlo simulation from an
        ase readable file
        """
        structure = read(filename)
        return structure


    @staticmethod
    def run_single_mc(mc_args):
        """
        Runs a Monte Carlo Simulation on a single Core
        
        Inputs
        ======
        ce_file, str
            Path to a mchammer readable file to a Cluster Expansion
        out_file, str
            Path to the mchammer output file
        delta_mu, list of floats,
            Reference chemical potentials of the adsorbed species,
            the length of the list must be as long as in the same order as the
            ads_species
        from_read, bool (default = False)
            Should the initial structure be read from an ase readable file
            if True requires additional kwarg strucname
        """
        ads_species = mc_args.get('ads_species')
        delta_mu = mc_args.get('delta_mu')
        chemical_potentials = {
            ads: delta_mu[i] for i, ads in enumerate(ads_species)
        }

        ce = ClusterExpansion.read(mc_args.get('ce_file'))

        from_read = mc_args.get('from_read', False)
        if from_read:
            strucname = mc_args.get('strucname', "")
            structure   = MCRunner.read_initial_struc(strucname)
        else:
            size        = mc_args.get('size')
            init_type   = mc_args.get('init_type')
            structure   = MCRunner.build_initial_struc(
                    ce, size, ads_species, init_type
            )
        
        calculator = ClusterExpansionCalculator(structure, ce)
        
        mc = SemiGrandCanonicalEnsemble_Diffsteps(
            structure=structure,
            calculator=calculator,
            temperature=mc_args.get('temperature'),
            dc_filename=mc_args.get('out_file'),
            chemical_potentials=chemical_potentials,
            prob_threshold=mc_args.get('prob_threshold')
        )
        mc.run(
            number_of_trial_steps=mc_args.get('steps')
        )
        return None


    def run_batch_mcs(self,
                      ce_df : pd.DataFrame,
                      delta_mu: Union[list, None],
                      n_cores : int
        ):
        """
        Organizes the parallel submission of multiple Monte Carlo simulations

        Inputs 
        ======
        ce_df, pd.DataFrame (required columns: 'filename', 'ref', 'pot')
            DataFrame containing the paths to the cluster expansion files
            and the reference electrode and potential they were measured for
        delta_mu: list or None,
            List of floats as long as the ads_species in the cluster expansion
            We assume here that this value doesn't change throughout all 
            monte carlo runs, meaning that all necessary experimental 
            parameters have already been set during the cluster expansion fit
            if Nonetype, we search for the delta_mu within the dataframe as 
            column named 'delta_mu'
        n_cores, int
            Number of cores over which multiprocessing Pools over
        """
        out_dir = self._batch_kwargs.pop("out_dir")
        # Make sure output directory exists
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
            
        log_file = self._batch_kwargs.pop(
            'log_file', os.path.join(out_dir, 'log.json')
        )

        if isinstance(delta_mu, list):
            delta_mus = [delta_mu for i in range(ce_df.shape[0])]
        else:
            delta_mus = [ce_df.loc[i]['delta_mu'] for i in range(ce_df.shape[0])]
        
        if os.path.isfile(log_file):
            raise ValueError("log file already exists! Aborting now...")

        count           = 0
        n_repeats       = self._batch_kwargs.get('n_repeats')
        args = []
        for i in range(n_repeats):
            for j in ce_df.index:
                row = cp(ce_df.loc[j])
                mc_run_args = cp(self._mc_args)
                mc_run_args['ce_file']  = row['filename']
                mc_run_args['out_file'] = f'{out_dir}/run_{count}.dc'
                mc_run_args['delta_mu'] = delta_mus[j]
                mc_run_args['repeat']   = i
                mc_run_args['file_idx'] = count
                mc_run_args['ref']      = row['ref']
                mc_run_args['pot']      = row['pot']

                with open(log_file, 'a') as f:
                    json.dump(mc_run_args, f)
                    f.write('\n')
                args.append(mc_run_args)
                count += 1

        # step 4: Define a Pool object with the desired number of processes and run
        pool        = Pool(processes=n_cores)
        results     = pool.map_async(MCRunner.run_single_mc, args)
        results.get()
        return None
