import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from reservoir.model import ESN
from functools import partial
import json
from multiprocessing import Pool
import tqdm

plt.rcParams.update({'font.size': 42})
plt.ioff()


def embedding(x, delay):
    # x must be a 3d np.array with trajectory values in the 1st dimension
    # this function returns a 3d np.array containing both x and the delayed values concatenated along the 3rd dimension
    if delay < 0:
        delay = -1 * delay
        x_delayed = x[:-delay, ...]
        y = np.concatenate((x[delay:, ...], x_delayed), axis=2)
    else:
        x_delayed = x[delay:, ...]
        y = np.concatenate((x[:-delay, ...], x_delayed), axis=2)
    return y


def mape(y, y_hat):
    return float((100 * sum(abs(y[:, 0, 0] - y_hat[:, 0, 0]))) / (len(y) * abs(max(y) - min(y))))


def mse(y_true, y_pred):
    """Mean square error between samples."""
    n = y_pred.shape[0]
    return np.mean((y_true[-n:, ...] - y_pred) ** 2)


def train_model(n_features, seed, u_train, y_train, use_iterative=True, ridge=0.0,
                state_transform='quadratic', spectral_radius=1.2, density=0.2,
                input_scale=0.1, burnout_train=200, leaky_factor=0.6):
    nys = [[1]]  # unidimensional output (only x state)
    nus = [[1]]
    mdl = ESN(nus=nus, nys=nys, n_features=n_features, burnout=burnout_train,
              input_scale=input_scale, density=density,
              spectral_radius=spectral_radius, state_transform=state_transform,
              use_iterative=use_iterative, ridge=ridge, random_state=seed,
              leaky_factor=leaky_factor)
    mdl.fit(u_train, y_train)
    return mdl


def get_best_model(data, model_selected=0, show_k_best=False, k_best=10):
    k_smallest = np.argsort(data['mape_testing'])
    best_pos = k_smallest.iloc[model_selected]
    if show_k_best:
        mape_training = data['mape_training']
        mape_testing = data['mape_testing']
        mdl = data['model'].iloc[best_pos]
        mdl_dict = json.loads(mdl)
        mdl = ESN.fromdict(mdl_dict)
        print('The selected model metrics are: \n')
        print(f'n_features: {mdl.n_states:d}, ridge: {mdl.ridge:.3e},'
              f'radius: {mdl.spectral_radius:.3e}, input_scale: {mdl.input_scale:.3f},'
              f'leaky_factor: {mdl.leaky_factor:.2f}, density: {mdl.density:.2f}\n')
        print(f'Training: mape = {mape_training.iloc[best_pos]:.3f}\n')
        print(f'Testing: mape = {mape_testing.iloc[best_pos]:.3f}\n')
        print(f'The {k_best:d} best models are: \n')
        for k_index in range(0, k_best):
            mdl = data['model'].iloc[k_smallest.iloc[k_index]]
            mdl_dict = json.loads(mdl)
            mdl = ESN.fromdict(mdl_dict)
            pos = k_smallest.iloc[k_index]
            print(f'k_index: {k_index:d}, testing mape: {mape_testing.iloc[pos]}n_features: {mdl.n_states:d}, ridge: {mdl.ridge:.3e},'
                  f'radius: {mdl.spectral_radius:.3e}, input_scale: {mdl.input_scale:.3f},'
                  f'leaky_factor: {mdl.leaky_factor:.2f}, density: {mdl.density:.2f}\n')
    mdl = data['model'].iloc[best_pos]
    mdl_dict = json.loads(mdl)
    mdl = ESN.fromdict(mdl_dict)
    return mdl


def read_monte_carlo_data(results, monte_carlo_runs=30):
    # populating the dataframe
    data = []
    for k in range(0, len(results)):
        for j in range(0, monte_carlo_runs):
            parameters = list(results[k][0:6]).copy()
            parameters.append(results[k][7][j].random_state)
            parameters.append(results[k][7][j])
            parameters.append(results[k][8][j])
            parameters.append(results[k][9][j])
            parameters.append(results[k][10][j])
            parameters.append(results[k][11][j])
            data.append(parameters)
    raw_data = pd.DataFrame(data)
    raw_data.columns = ['n_features', 'ridge', 'spectral_radius', 'input_scale',
                        'leaky_factor', 'density', 'seed', 'model', 'mape_training',
                        'mse_training', 'mape_testing', 'mse_testing']
    return raw_data


def get_hp_occurences(data, variable):
    aux = data[variable].value_counts()
    variable_values = aux.index.values
    variable_values.sort()
    return variable_values


def results_by_seed(data, n_of_seeds):
    n_of_exp = len(data)
    chunks = np.arange(0, n_of_exp, n_of_seeds)
    data_by_seed = []
    for i, chunk in enumerate(chunks):
        df = data.iloc[chunk:chunk+n_of_seeds].copy()
        df.reset_index(inplace=True)
        mean_mape_testing = np.mean(df['mape_testing'])
        testing_std = np.std(df['mape_testing'])
        mean_mape_training = np.mean(df['mape_training'])
        training_std = np.std(df['mape_training'])
        row = list(df[['n_features', 'ridge', 'spectral_radius', 'input_scale', 'leaky_factor', 'density']].iloc[0])
        row.append(mean_mape_testing)
        row.append(testing_std)
        row.append(mean_mape_training)
        row.append(training_std)
        data_by_seed.append(row)
    data_by_seed = pd.DataFrame(data_by_seed, columns=['n_features', 'ridge', 'spectral_radius', 'input_scale',
                                                       'leaky_factor', 'density', 'mape_testing', 'testing_std',
                                                       'mape_training', 'training_std'])
    return data_by_seed


def parallel_esn_grid_search(n_features_list, ridge_list, spectral_radius_list, input_scale_list, leaky_factor_list, density_list, u_train,
                             y_train, u_test, y_test, seed_list, sim_settings):
    use_iteractive = sim_settings['use_iteractive']
    state_transform = sim_settings['state_transform']
    burnout_simulate = sim_settings['burnout_simulate']
    osa = sim_settings['osa']
    (n_features, ridge, radius, input_scale, leaky_factor, density) = (n_features_list, ridge_list, spectral_radius_list,
                                                                      input_scale_list, leaky_factor_list, density_list)
    monte_carlo = not(np.size(seed_list) == 1)
    if monte_carlo:
        models_list = []
        mape_training = []
        mape_testing = []
        mse_training = []
        mse_testing = []
        for seed in seed_list:
            # model training for each seed
            mdl = train_model(n_features=n_features, seed=seed, u_train=u_train, y_train=y_train,
                              use_iterative=use_iteractive, ridge=ridge,
                              state_transform=state_transform, spectral_radius=radius,
                              burnout_train=burnout_simulate, density=density,
                              input_scale=input_scale, leaky_factor=leaky_factor)
            # model testing for each seed
            y_sim_test = mdl.simulate(u_test, y_test, burnout=burnout_simulate,
                                      output_burnout=False, one_step_ahead=osa)
            models_list.append(mdl)
            # appending training metrics
            mape_training.append(
                mape(y_train[mdl.order + mdl.burnout_train:, ...], mdl.y_sim_train))
            mse_training.append(
                mse(y_train[mdl.order + mdl.burnout_train:, ...], mdl.y_sim_train))
            # appending testing metrics
            if osa:
                y_test_new = y_test[mdl.burnout_train:-1, ...]
            else:
                y_test_new = y_test[mdl.burnout_train:, ...]
            mape_testing.append(mape(y_test_new, y_sim_test))
            mse_testing.append(mse(y_test_new, y_sim_test))
        return n_features, ridge, radius, input_scale, leaky_factor, density, seed_list, \
               models_list, mape_training, mse_training, mape_testing, mse_testing
    else:
        # model training for chosen seed
        mdl = train_model(n_features, seed_list, u_train, y_train,
                          use_iterative=use_iteractive, ridge=ridge,
                          state_transform=state_transform, spectral_radius=radius,
                          burnout_train=burnout_simulate, density=density,
                          input_scale=input_scale, leaky_factor=leaky_factor)
        # model testing for chosen seed
        y_sim_test = mdl.simulate(u_test, y_test, burnout=burnout_simulate,
                                  output_burnout=False, one_step_ahead=osa)
        # computing training metrics
        mape_training = (
            mape(y_train[mdl.order + mdl.burnout_train:, ...], mdl.y_sim_train))
        mse_training = (mse(y_train[mdl.order + mdl.burnout_train:, ...], mdl.y_sim_train))
        # testing metrics
        if osa:
            y_test_new = y_test[mdl.burnout_train:-1, ...]
        else:
            y_test_new = y_test[mdl.burnout_train:, ...]
        mape_testing = (mape(y_test_new, y_sim_test))
        mse_testing = (mse(y_test_new, y_sim_test))
        return n_features, ridge, radius, input_scale, leaky_factor, density, mdl.random_state, \
               mdl, mape_training, mse_training, mape_testing, mse_testing


def adapt_to_parquet(raw_data, path, parquet_name):
    keys_to_remove = ['prepared_nys', 'prepared_nus', 'n_eff_inputs', 'order', 'rng', 'w_zx', 'w_xx',
                      'activation', 'y_sim_train']
    models_dict = raw_data['model'].copy()
    models_dict = [model.__dict__ for model in models_dict]
    for model in models_dict:
        for key in keys_to_remove:
            del model[key]
    for model in models_dict:
        model['n_features'] = model.pop('n_states')
        model['burnout'] = model.pop('burnout_train')
        model['w_xy'] = model['w_xy'].tolist()
    models_json = [json.dumps(model) for model in models_dict]
    raw_data['model'] = models_json
    raw_data.to_parquet(os.path.join(path, f'{parquet_name}.parquet.gzip'), compression='gzip')
    return None


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def run_experiment(path, u_train, y_train, u_test, y_test, parameters_list, seed_list, block, sim_settings, n_of_cpu=8):
    monte_carlo = not (np.size(seed_list) == 1)
    func = partial(parallel_esn_grid_search, sim_settings=sim_settings,
                   u_train=u_train, y_train=y_train, u_test=u_test, y_test=y_test,
                   seed_list=seed_list)
    print(f"Number of vCPU's: {n_of_cpu}/{os.cpu_count()}")
    with Pool(processes=n_of_cpu) as pool:
        results = pool.starmap(func, tqdm.tqdm(parameters_list, total=len(parameters_list)), chunksize=1)
    if not os.path.exists(path):
        os.makedirs(path)
    print(f"I'm done with the computations for block {block}!\n")
    print(f"But calm down... I'm still saving your results of block {block} as a dataframe. You're welcome :)\n")
    if monte_carlo:
        results_df = read_monte_carlo_data(results=results, monte_carlo_runs=len(seed_list))
    else:
        results_df = pd.DataFrame(results, columns=['n_features', 'ridge', 'spectral_radius', 'input_scale',
                                                    'leaky_factor', 'density', 'seed', 'model', 'mape_training',
                                                    'mse_training', 'mape_testing', 'mse_testing'])
    adapt_to_parquet(raw_data=results_df, path=path, parquet_name=f'results_dataframe_{block}')
    print(f"Done with block {block}!!!\n")
    return 0








