import numpy as np
import os
import running_functions_duffing as run
from itertools import product
import pandas as pd
import warnings
warnings.filterwarnings("ignore")  # we're aware of convergence problems. However, some spectral radius aren't adjusted.


if __name__ == '__main__':

    # defining simulation settings
    experiment_name = "freerun_a63_seed5"  # name of the folder containing the results
    sim_settings = {'use_iteractive': False,  # choose iteractive or SVD implementation of ridge regression
                    'osa': False,  # choose to test models using one step aheada prediction or freerun
                    'invert_alpha': True,  # set the model with the complementary value of the alpha set
                    'state_transform': 'quadratic',  # choose identity or quadratic state transformation
                    'burnout_simulate': 200,  # number of initial model samples to be discarded for training
                    'collect_data': False,  # whether to simulate the system to gather new data or not
                    'train_time': 100,  # amount of time [seconds] used for system simulation to gahter training data
                    'test_time': 50000,  # amount of time [seconds] used for system simulation to gahter testing data
                    'trans_time': 300,  # amount of time [seconds] considered for transients
                    'dt': np.pi / 600,  # step used for numerical integration
                    'decimate': 10,  # ratio used for data decimation after measurement
                    'ss_amp': 11,  # steady state amplitude
                    'plot_train': False,  # whether to plot training data gathered or not
                    'plot_test': False,  # whether to plot testing data gathered or not
                    'model_color': 'red',  # color used to represent model timeseries
                    'train_name': None,  # name used for results concerning training simulation
                    'test_name': None,  # name used for results concerning testing simulation
                    'embedd_delay': 4}  # time delay used for embedding
    current_directory = os.getcwd()
    results_folder = os.path.join(current_directory, experiment_name)  # defining the folder to save the results
    n_of_cpu = 8  # setting the number of vCPUs to be used in parallel execution

    if sim_settings['collect_data']:  # collecting dataset for training and testing
        y_train, u_train, y_test, u_test = run.collect_dataset(sim_settings=sim_settings, path=results_folder)
        np.save('data/y_test_validation_5000.npy', y_test)
        np.save('data/u_test_validation_5000.npy', u_test)
    else:  # using saved dataset for training and testing
        y_train = np.load('data/y_train.npy')
        u_train = np.load('data/u_train.npy')
        y_test = np.load('data/y_test_63.npy')
        u_test = np.load('data/u_test_63.npy')

    # defining the grid search to be explored
    n_features_grid = [200]
    ridge_grid = [1e-9, 1e-6, 1e-3, 1e-1, 1e1]
    radius_grid = [0.1, 0.3, 0.6, 0.8, 0.9, 1.2, 2, 3]
    input_scale_grid = [1, 0.05, 0.1, 0.2, 0.5, 2]
    leaky_factor_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
    density_grid = [0.05, 0.1, 0.15, 0.5, 0.75, 0.9]
    parameters_list = list(product(n_features_grid, ridge_grid, radius_grid, input_scale_grid,
                                   leaky_factor_grid, density_grid))
    seed_list = [1, 5]
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
