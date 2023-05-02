import os 
import numpy as np 
import pandas as pd

from copy import deepcopy as cp

from mchammer import DataContainer
from multiprocessing import Pool


class MCParser:

    def __init__(self):
        return None
    
    @staticmethod
    def parse_single_mc_run(args):
        """
        Parse a single Monte Carlo run on one core, for a detailed description
        of the arguments see `analyze_batch_mc_runs`
        """
        ensemble    = args.get('ensemble')
        dc_file     = args.get('dc_file')

        dc = DataContainer.read(dc_file)
        data_row = dc.ensemble_parameters
        data_row['filename'] = dc_file
        n_atoms = data_row['n_atoms']

        equis = args.get('equilibration')
        if not isinstance(equis, list):
            equis = [equis]

        equis = [int(i * args['steps']) for i in equis]
        
        ads_species = args.get('ads_species')
        if isinstance(ads_species, set):
            ads_species = list(ads_species)

        data_rows = []
        for equi in equis:
            dr = cp(data_row)
            dr['pot'] = args['pot']
            dr['ref'] = args['ref']
            dr['steps'] = args['steps']
            dr['size'] = args['size']
            dr['repeat'] = args['repeat']
            dr['cov'] = 0.
            for ads in ads_species:
                stats = dc.analyze_data(f'{ads}_count', start=equi)
                dr[f'{ads}_cov']        = stats['mean'] / n_atoms
                dr[f'{ads}_cov_std']    = stats['std'] / n_atoms
                dr[f'{ads}_cov_error']  = stats['error_estimate'] / n_atoms
                dr['cov'] += dr[f'{ads}_cov']

            stats = dc.analyze_data('potential', start=equi)
            dr['e_per_site']        = stats['mean'] / n_atoms
            dr['e_per_site_std']    = stats['std'] / n_atoms
            dr['e_per_site_error']  = stats['error_estimate'] / n_atoms

            dr['acceptance_ratio'] = dc.get_average('acceptance_ratio', start=equi)

            if ensemble == 'sgc':
                dr['free_energy_derivative'] = -1. * dr['mu_X']
                for ads in ads_species:
                    dr['free_energy_derivative'] += dr[f'mu_{ads}']
            elif ensemble == 'vcsgc':
                dr['free_energy_derivative'] = \
                    dc.get_average('free_energy_derivative_Pd', star=equi)
            dr['equilibration'] = equi
            data_rows.append(dr)
        return data_rows
    
    def parse_batch_mc_runs(self,
                            dc_log_df,
                            equis : list,
                            outfiles : list,
                            n_cores : int,
                            ensemble : str = 'sgc'
                            ):
        """
        Parse all the Monte Carlo runs described in a logfile created by the 
        MCRunner class
        """
        if not len(outfiles) == len(equis):
            raise ValueError("Length of outfiles does not match equi length")
        
        for outfile in outfiles:
            if not os.path.exists(outfile):
                raise ValueError("Path to outfile does not exist!")
        
        ads_species = set(dc_log_df['ads_species'].sum())
        ads_species.remove('X')

        args = []
        for i in dc_log_df.index:
            row = dc_log_df.loc[i]
            run_args = {
                'ads_species':  ads_species,
                'dc_file':      row['out_file'],
                'ensemble':     ensemble,
                'equilibration':equis,
                'pot':          row['pot'],
                'ref':          row['ref'],
                'steps':        row['steps'],
                'size':         row['size'],
                'repeat':       row['repeat']
            }
            args.append(run_args)

        pool = Pool(processes=n_cores)
        results = pool.map_async(MCParser.parse_single_mc_run, args)

        # Store the results in a pandas dataframe object
        data = np.array([data_row for data_row in results.get()], dtype='object')
        dfs = []
        for i, equi in enumerate(equis):
            df = (pd.DataFrame(data[:, i].tolist())
                  .fillna(0.)
                  .sort_values(['pot', 'repeat'], ascending=True)
                  .reset_index(drop=True)
                  )
            df['equi'] = equi
            if outfiles[i].endswith('.json'):
                df.to_json(outfiles[i])
            elif outfiles[i].endswith('.csv'):
                df.to_csv(outfiles[i], sep='\t')
            else:
                raise ValueError("Do not know how you want to save the outfile!")
            dfs.append(df)
        return dfs 
