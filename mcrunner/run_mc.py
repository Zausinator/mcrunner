import json, os, random, sys

import numpy as np
import pandas as pd

from ase.io import read, write
from ase.build import  make_supercell
from ase.visualize import view

from copy import deepcopy as cp

from icet import ClusterExpansion

from mchammer.calculators import ClusterExpansionCalculator
from mchammer.configuration_manager import SwapNotPossibleError
from mchammer.ensembles import SemiGrandCanonicalEnsemble

from multiprocessing import Pool


class SemiGrandCanonicalEnsemble_Diffsteps(SemiGrandCanonicalEnsemble):
    """
    Adds possible diffusion steps to the SGC Ensemble from mchammer
    """
    
    @property
    def prob_threshold(self):
        return self._prob_threshold
    

    @prob_threshold.setter
    def prob_threshold(self, value : float):
        self._prob_threshold = value


    def _do_trial_step(self):
        """ Carries out one Monte Carlo trial step. """
        sublattice_index = self.get_random_sublattice_index(
            probability_distribution=self._flip_sublattice_probabilities)

        if random.random() < self._prob_threshold:
            try:
                r = self.do_canonical_swap(
                    sublattice_index=sublattice_index
                )
            except SwapNotPossibleError:
                r = self.do_sgc_flip(
                    self.chemical_potentials, sublattice_index=sublattice_index
                )
            return r
        else:
            return self.do_sgc_flip(
                self.chemical_potentials, sublattice_index=sublattice_index
            )


def build_initial_struc(primstruct, size, ads_species, init_type):
    structure = make_supercell(primstruct,
                       1 * np.array([[size, 0, 0],
                                     [0, size, 0],
                                     [0, 0, 1]]))
    n_species = len(ads_species)
    occ = []
    for x in range(size):
        for y in range(size):
            if init_type == 'c2x2':
                tmp = (x+y) % 2
                if tmp ==0:
                    occ.append(0)
                else:
                    occ.append(1)
                #occ.append((x+y) % n_species)
            elif init_type == 'bare':
                occ.append(n_species-1)
            elif init_type == 'full':
                    occ.append(0)
            else:
                occ.append(np.random.randint(n_species))

    structure.set_chemical_symbols([ads_species[o] for o in occ ])
    return structure

def read_initial_struc(filename):
    structure = read(filename)
    return structure


def run_single_mc(args):
    ce_file     = args['ce_file']
    ce = ClusterExpansion.read(ce_file)

    init_type   = args['init_type']
    ads_species = args['ads_species']
    size        = args['size']

    primstruct  = ce.get_cluster_space_copy().primitive_structure
    structure   = build_initial_struc(primstruct, size, ads_species, init_type)
    #structure   = read_initial_struc(strucname)
    calculator = ClusterExpansionCalculator(structure, ce)

    out_file    = args['out_file']
    temperature = args['temperature']
    dmu         = args['dmu']
    steps       = args['steps']

    chemical_potentials = {
        ads_species[i]: dmu[i] for i in range(len(ads_species))}

    mc = SemiGrandCanonicalEnsemble_Diffsteps(
        structure=structure, calculator=calculator,
        temperature=temperature, dc_filename=out_file,
        chemical_potentials=chemical_potentials
    )
    mc.prob_threshold = args['prob_threshold']
    mc.run(number_of_trial_steps=steps)
    return None


def run_batch_mcs(**kwargs):

    out_dir = kwargs.get("out_dir")
    # Make sure output directory exists
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    mc_args = {
        'init_type':            kwargs.pop('init_type'),
        'ads_species':          kwargs.pop('ads_species'),
        'size':                 kwargs.pop('size'),
        'temperature':          kwargs.pop('temperature', 300.),
        'steps':                kwargs.pop('steps'),
        'prob_threshold':       kwargs.pop('prob_threshold')
    }
    log_file = kwargs.pop('log_file', 'log.json')

    ce_df = kwargs.pop("ce_df")

    count = 0
    n_repeats       = kwargs.pop('n_repeats', 1)
    pot             = kwargs.pop("potential")
    args = []
    for i in range(n_repeats):
        for j in ce_df.index:
            # Here we also want to go up to 0.001 V resolution on the x axis
            row = cp(ce_df.loc[j])
            mc_run_args = cp(mc_args)
            mc_run_args['ce_file']  = row['filename']
            mc_run_args['out_file'] = f'{out_dir}/run_{count}.dc'
            mc_run_args['dmu']      = [pot, pot, 0.]
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
    processes = kwargs.pop('processes', 20)
    pool = Pool(processes=processes)
    results = pool.map_async(run_single_mc, args)
    results.get()
    return None


def main():
    n_cores     = 30
    n_steps     = 10**6
    init_type   = "bare"

    df = pd.read_json(f'./CE_at_applied_potential/log.json', lines=True)

    out_dir     = f"./run"

    args = {
        'ce_df':                df,
        'init_type':            init_type,
        'ads_species':          ['Br', 'X'],
        'size':                 18,
        'temperature':          300.,
        'prob_threshold':       0.75,
        'steps':                n_steps,
        'processes':            n_cores,
        'potential':            np.round(0., 3),
        'out_dir':              out_dir,
        'n_repeats':            1,
        'log_file':             f'{out_dir}/log.json'
    }
    run_batch_mcs(**args)
    return None

if __name__ == '__main__':
    main()
