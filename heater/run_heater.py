import numpy as np
import os
import running_functions_heater as run
from itertools import product
import pandas as pd
import warnings
warnings.filterwarnings("ignore")  # we're dealing with convergence problems internally


if __name__ == '__main__':

    # defining simulation settings
    experiment_name = "din2_freerun_mc_100_300_600_neurons"  # name of the folder containing the results
    sim_settings = {'use_iteractive': False,  # choose iteractive or SVD implementation of ridge regression
                    'state_transform': 'quadratic',  # choose identity or quadratic state transformation
                    'burnout_simulate': 100,  # number of initial model samples to be discarded for training
                    'osa': False,  # choose to test models using one step aheada prediction or freerun
                    'invert_alpha': True}  # set the model with the complementary value of the alpha set
    current_directory = os.getcwd()
    results_folder = os.path.join(current_directory, experiment_name)
    n_of_cpu = 8  # setting the number of vCPUs to be used in parallel execution

    # loading dataset for training and testing
    y_train = np.load('data/y_train.npy')
    u_train = np.load('data/u_train.npy')
    y_test = np.load('data/y_test.npy')
    u_test = np.load('data/u_test.npy')

    # defining the grid search to be explored
    n_features_grid = [100, 300, 600]
    ridge_grid = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    radius_grid = [1e-3, 1e-2, 7e-2, 1e-1, 2e-1, 5e-1, 1e0, 3e0, 1e1]
    input_scale_grid = [1]
    leaky_factor_grid = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    density_grid = [0.2]
    parameters_list = list(product(n_features_grid, ridge_grid, radius_grid, input_scale_grid,
                                   leaky_factor_grid, density_grid))
    seed_list = list(range(30, 60))
    parameters_block = list(run.chunks(parameters_list, 100))  # chunking hyperparameter combinations

    # running the experiments on each chunk of parameters
    for current_block, current_list in enumerate(parameters_block):
        _ = run.run_experiment(path=results_folder,
                               n_of_cpu=n_of_cpu,
                               sim_settings=sim_settings,
                               u_train=u_train,
                               y_train=y_train,
                               u_test=u_test,
                               y_test=y_test,
                               parameters_list=current_list,
                               seed_list=seed_list,
                               block=current_block)
    print("\n\nThe experiment was a success!\n")
    print("Now I'm gathering all chunks of results...\n")

    # concatenating chunks of results and saving into a parquet file
    total_results = pd.concat([pd.read_parquet(os.path.join(results_folder,
                                                            f'results_dataframe_{block}.parquet.gzip'))
                               for block, _ in enumerate(parameters_block)], axis=0, ignore_index=True)
    total_results.to_parquet(os.path.join(results_folder, 'results_dataframe_complete.parquet.gzip'),
                             compression='gzip')
    print("\nDone!!!")
