import numpy as np
import os
from scipy import stats
import running_functions_heater as run
import pandas as pd
import matplotlib.pyplot as plt
from reservoir.model import ESN
import json

plt.rcParams.update({'font.size': 42})


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

    # loading dataset for training and testing
    y_train = np.load('data/y_train.npy')
    u_train = np.load('data/u_train.npy')
    y_test = np.load('data/y_test.npy')
    u_test = np.load('data/u_test.npy')

    # loading results dataframe
    raw_data = pd.read_parquet(os.path.join(results_folder, 'results_dataframe_complete.parquet.gzip'))
    if sim_settings['invert_alpha']:
        raw_data['leaky_factor'] = 1 - raw_data.loc[:, 'leaky_factor']
    raw_data.info()

    # getting best model and its HPs
    data_by_seed = run.results_by_seed(data=raw_data, n_of_seeds=30)
    k_smallest = np.argsort(data_by_seed['mape_testing'])
    best_pos = k_smallest.iloc[0]
    n_neurons = [100, 300, 600]
    dataset = 'din2'
    if os.path.exists(os.path.join(results_folder, f"best_results_{dataset}.txt")):
        os.remove(os.path.join(results_folder, f"best_results_{dataset}.txt"))
    for _, n_features in enumerate(n_neurons):
        data = raw_data[raw_data['n_features'] == n_features].copy()
        best_model = run.get_best_model(data=data, model_selected=0, show_k_best=True, k_best=3)
        best_rho = best_model.spectral_radius
        if sim_settings['invert_alpha']:
            best_alpha = 1 - best_model.leaky_factor
        else:
            best_alpha = best_model.leaky_factor
        best_ridge = best_model.ridge
        with open(os.path.join(results_folder, f'best_results_{dataset}.txt'), 'a') as f:
            f.write(f'\n n_features: {n_features}, best rho: {best_rho:.3e}, '
                    f'best alpha: {best_alpha:.2f}, '
                    f'best ridge: {best_ridge:.3e}\n')

    # plotting time series of a chosen model
    data = raw_data[(raw_data['n_features'] == 300) &
                    (round(raw_data['leaky_factor'], 2) == 0.2) &
                    (raw_data['spectral_radius'] == 2e-1) &
                    (raw_data['ridge'] == 1e-3) &
                    (raw_data['seed'] == 34)]
    mdl = data['model'].iloc[0]
    mdl_dict = json.loads(mdl)
    mdl = ESN.fromdict(mdl_dict)
    y_sim_test = mdl.simulate(u=u_test, y=y_test, burnout=sim_settings['burnout_simulate'],
                              output_burnout=False)
    fig, ax = plt.subplots(figsize=(19, 9))
    ax.plot(y_test[sim_settings['burnout_simulate']:sim_settings['burnout_simulate'] + 1500, 0, 0], 'b-', linewidth=2)
    ax.plot(y_sim_test[0:1500, 0, 0], 'r-', linewidth=2)
    ax.set_ylabel(r'$y(k)$, $\hat{y}(k)$')
    ax.set_xlabel(r'$k$')
    ax.set_xlim([0, 150])
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'test_time_series_rho_2e-1_alpha_02_ridge_1e-3_din2.eps'))
    print('Done!')

    # creating histograms for testing MAPE accross monte carlo runs
    df = raw_data[(raw_data['n_features'] == 300) &
                  (raw_data['spectral_radius'] == 1e-1) &
                  (raw_data['leaky_factor'] == 0.7) &
                  (raw_data['ridge'] == 1e-3)]
    z_scores = stats.zscore(df['mape_testing'])
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3)
    n_outliers = sum(np.logical_not(filtered_entries))
    df = df[filtered_entries]
    print(f'A total of {n_outliers} outliers were removed.')
    df.hist(column='mape_testing')
    plt.show()

    # creating boxplot of MAPE under rho variation for 300 neurons
    plt.close('all')
    fig, ax = plt.subplots(figsize=(19, 9))
    df = raw_data[(round(raw_data['n_features'], 10) == 300) &
                  (round(raw_data['leaky_factor'], 10) == 0.3) &
                  (round(raw_data['ridge'], 10) == 1e-3)]
    rhos = run.get_hp_occurences(data=df, variable='spectral_radius')
    df_rhos = pd.DataFrame()
    for ind, rho in enumerate(rhos):
        head = str(ind)
        df_rhos[head] = df[df['spectral_radius'] == rho]['mape_testing'].reset_index(drop=True)
    ax = df_rhos.boxplot(return_type='axes')
    rhos_str = []
    for rho in rhos:
        rhos_str.append(f'{rho:.0e}')
    ax.set_xticklabels(rhos_str)
    ax.set_xticklabels(['1e-3', '1e-2', '7e-2', '1e-1', '2e-1', '5e-1', '1', '3', '1e+1'])
    ax.set_ylim([0, 110])
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel('MAPE')
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'rho_300_boxplot_din2.eps'))

    # creating boxplot of MAPE under alpha variation for 300 neurons
    plt.close('all')
    fig, ax = plt.subplots(figsize=(19, 9))
    df = raw_data[(round(raw_data['n_features'], 10) == 300) &
                  (round(raw_data['spectral_radius'], 10) == 1e-1) &
                  (round(raw_data['ridge'], 10) == 1e-3)]
    alphas = run.get_hp_occurences(data=df, variable='leaky_factor')
    df_alphas = pd.DataFrame()
    for ind, alpha in enumerate(alphas):
        head = str(ind)
        df_alphas[head] = df[df['leaky_factor'] == alpha]['mape_testing'].reset_index(drop=True)
    ax = df_alphas.boxplot(return_type='axes')
    alphas_str = []
    for alpha in alphas:
        alphas_str.append(f'{alpha:.1f}')
    ax.set_xticklabels(alphas_str)
    ax.set_ylim([0, 110])
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('MAPE')
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'alpha_300_boxplot_din2.eps'))




