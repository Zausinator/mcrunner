import os, sys
import pandas as pd

from mchammer import DataContainer
from multiprocessing import Pool


def analyse_single_mc_run(kwargs):
    #Set the default settings for output files and such
    ensemble    = kwargs.get("ensemble", "sgc")
    filename    = kwargs.get('filename')

    dc = DataContainer.read(filename)

    data_row = dc.ensemble_parameters
    data_row['filename'] = filename
    n_atoms = data_row['n_atoms']

    #Equilibration should be ca. 308 steps per site, worth a check
    equilibration = kwargs.get("equilibration", 100000)

    ads = kwargs.get("ads")

    for ad in ads:
        stats = dc.analyze_data(f'{ad}_count', start=equilibration)
        data_row[f'{ad}_cov']          = stats['mean'] / n_atoms
        data_row[f'{ad}_cov_std']      = stats['std'] / n_atoms
        data_row[f'{ad}_cov_error']    = stats['error_estimate'] / n_atoms

    stats = dc.analyze_data('potential', start=equilibration)
    data_row['e_per_site']          = stats['mean'] / n_atoms
    data_row['e_per_site_std']      = stats['std'] / n_atoms
    data_row['e_per_site_error']    = stats['error_estimate'] / n_atoms

    data_row['acceptance_ratio'] = \
        dc.get_average('acceptance_ratio', start=equilibration)
    if ensemble == 'sgc':
        data_row['free_energy_derivative'] = - data_row['mu_X']
        for ad in ads:
            data_row['free_energy_derivative'] += data_row[f'mu_{ad}']
    elif ensemble == 'vcsgc':
        data_row['free_energy_derivative'] = \
            dc.get_average('free_energy_derivative_Pd', start=equilibration)
    return data_row


def analyze_batch_mc_runs(**kwargs):

    # Define directory hierarchy and outputfile location-name
    data_dir = kwargs.pop("data_dir")
    if not os.path.exists(data_dir):
        raise ValueError("Data directory does not exist yet")
    outfile = kwargs.pop("outfile", os.path.join(data_dir, "results.json"))

    # Define the mpi multiprocessing parameters?
    processes = kwargs.pop("processes", 20)

    files = [f"{data_dir}/{i}" for i in os.listdir(data_dir) if i.endswith(".dc")]
    pool_args = [{"filename": i, **kwargs} for i in files]

    pool = Pool(processes=processes)
    results = pool.map_async(analyse_single_mc_run, pool_args)

    # Store the results in a pandas dataframe object
    data = [data_row for data_row in results.get()]
    df = pd.DataFrame(data)
    df['cov'] = df[[f'{ad}_cov' for ad in kwargs.get('ads')]].sum(axis=1)
    df['Br_cov_error']      = df['Br_cov_error'].fillna(0.)
    df['e_per_site_error']  = df['e_per_site_error'].fillna(0.)
    df = df.sort_values("mu_Br", ascending=True).reset_index(drop=True)
    if outfile.endswith('.json'):
        df.to_json(outfile)
    elif outfile.endswith('.csv'):
        df.to_csv(outfile, sep='\t')
    else:
        raise ValueError("Do not know how you want to save the outfile!")
    return df


def main(verbose=False):
    n_steps = 10**8
    n_cores = 30 

    ads = ['Br']
    swap_rate = 0.75
    equi = 0.99
    data_dir = f'./run'

    equilibration = int(equi * n_steps)
    kwargs = {
        "ads":              ads,
        "data_dir":         data_dir,
        "equilibration":    equilibration,
        "ensemble":         "sgc",
        "processes":        n_cores,
        "outfile":          os.path.join(data_dir, f'results_{equi}_equi.json')
    }
    df = analyze_batch_mc_runs(**kwargs)
    return None


if __name__ == '__main__':
    main()
