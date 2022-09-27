import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import seaborn as sns
import matplotlib.cm as cm
import json
from reservoir.model import ESN
from datasets import get_dataset
from functools import partial
from multiprocessing import Pool
import tqdm
from matplotlib.ticker import ScalarFormatter

plt.rcParams.update({'font.size': 42})
plt.ioff()


class ScalarFormatterClass(ScalarFormatter):
    def _set_format(self):
        self.format = "%.2f"


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


def collect_dataset(sim_settings, path):
    train_name = sim_settings['train_name']
    test_name = sim_settings['test_name']
    y_train, y_test, u_train, u_test = get_dataset(train_time=sim_settings['train_time'],
                                                   test_time=sim_settings['test_time'],
                                                   trans_time=sim_settings['trans_time'], dt=sim_settings['dt'],
                                                   decimate=sim_settings['decimate'],
                                                   ss_amp=sim_settings['ss_amp'],
                                                   plot_train_solution=sim_settings['plot_train'],
                                                   plot_test_solution=sim_settings['plot_test'])
    folder = path  # os.path.join(path, 'data')
    if train_name:
        np.save(os.path.join(folder, f'y_{train_name}.npy'), y_train)
        np.save(os.path.join(folder, f'u_{train_name}.npy'), u_train)
    if test_name:
        np.save(os.path.join(folder, f'y_{test_name}.npy'), y_test)
        np.save(os.path.join(folder, f'u_{test_name}.npy'), u_test)
    return y_train, u_train, y_test, u_test


def get_best_model(df, input_scale, density, model_selected=0, show_k_best=False, k_best=10):
    data = df[(df['input_scale'] == input_scale) & (df['density'] == density)]
    k_smallest = np.argsort(data['mape_testing'])
    best_pos = k_smallest.iloc[model_selected]
    if show_k_best:
        mape_training = data['mape_training']
        mape_testing = data['mape_testing']
        mse_training = data['mse_training']
        mse_testing = data['mse_testing']

        mdl = data['model'][best_pos]
        print('The selected model metrics are: \n')
        print(f'n_features: {mdl.n_states:d}, ridge: {mdl.ridge:.3e},'
              f'radius: {mdl.spectral_radius:.3e}, input_scale: {mdl.input_scale:.3f},'
              f'leaky_factor: {mdl.leaky_factor:.2f}, density: {mdl.density:.2f}\n')
        print(f'Training: mape = {mape_training[best_pos]:.3f},'
              f'mse = {mse_training[best_pos]:.6f}\n')
        print(f'Testing: mape = {mape_testing[best_pos]:.3f},'
              f'mse = {mse_testing[best_pos]:.6f}\n')

        print(f'The {k_best:d} best models are: \n')
        for k_index in range(0, k_best):
            mdl = data['model'][k_smallest[k_index]]
            pos = k_smallest.iloc[k_index]
            print(
                f'k_index: {k_index:d}, testing mape: {mape_testing[pos]}n_features: {mdl.n_states:d}, ridge: {mdl.ridge:.3e},'
                f'radius: {mdl.spectral_radius:.3e}, input_scale: {mdl.input_scale:.3f},'
                f'leaky_factor: {mdl.leaky_factor:.2f}, density: {mdl.density:.2f}\n')

    mdl = data['model'].iloc[best_pos]
    mdl_dict = json.loads(mdl)
    mdl = ESN.fromdict(mdl_dict)
    return mdl


def plot_time_series(sim_settings, y_ref, y_hat, path, save_name):
    fig, ax = plt.subplots(figsize=(19, 9))
    ax.plot(y_ref, color='blue', label='measured')
    ax.plot(y_hat, color=sim_settings['model_color'], label='predicted')
    ax.set_xlabel('k')
    ax.set_ylabel('x(k)')
    # ax.legend()
    # ax.set_title(save_name.replace(" ", "_"))
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(os.path.join(path, save_name))
    plt.close()
    return fig, ax


def plot_phase_portrait(x, number_of_samples, path, save_name, sim_settings, delay=4, y=None):
    if delay:
        y = embedding(x, delay)
    fig, ax = plt.subplots(figsize=(19, 9))
    if 'original' in save_name:
        ax.plot(y[-number_of_samples:, 0, 0], y[-number_of_samples:, 0, 1],
                color='blue', lw=0.5)
    else:
        ax.plot(y[-number_of_samples:, 0, 0], y[-number_of_samples:, 0, 1],
                color=sim_settings['model_color'], lw=0.5)
    ax.set_xlabel('x(k)')
    ax.set_ylabel('x(k+%i)' % delay)
    # ax.set_title(f'{save_name} attractor')
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(os.path.join(path, save_name))
    plt.close()
    return fig, ax


def plot_poincare_section(sim_settings, x, path, save_name, number_of_points=40, phase=0, delay=4, y=None, ax=None,
                          marker_size=0.25, heatmap=False, axlim=None):
    dt_decimated = sim_settings['dt'] * sim_settings['decimate']
    t_p = int(2 * np.pi / dt_decimated)
    aux = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(19, 9))
    if delay:
        y = embedding(x, delay)
    y = y[phase::t_p, ...]
    if 'original' in save_name:
        ax.scatter(y[-number_of_points:, :, 0], y[-number_of_points:, :, 1],
                   color='blue', s=20 * 2 ** marker_size)
    else:
        ax.scatter(y[-number_of_points:, :, 0], y[-number_of_points:, :, 1],
                   color=sim_settings['model_color'], s=20 * 2 ** marker_size)

    if heatmap:
        if np.size(axlim) != 1:
            ax.set_xlim(axlim[0], axlim[1])
            ax.set_ylim(axlim[2], axlim[3])
        #         ax.ticklabel_format(useOffset=False)
        yScalarFormatter = ScalarFormatterClass(useMathText=True, useOffset=False)
        yScalarFormatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(yScalarFormatter)
        ax.xaxis.set_major_formatter(yScalarFormatter)
    else:
        ax.set_xlabel('x(k)')
        ax.set_ylabel('x(k+%i)' % delay)
        plt.tight_layout()
        if save_name is not None:
            plt.savefig(os.path.join(path, save_name))
    return ax


def plot_alpha_ridge_heatmaps(df, input_scale, density, spectral_values, subplots_rows, subplots_cols, path,
                              clip_upper, save_name=None, annotation=None):
    df = df[(df['input_scale'] == input_scale) & (df['density'] == density)]
    df = df[df['spectral_radius'].isin(spectral_values)]
    df['mape_testing'] = df['mape_testing'].clip(upper=clip_upper)
    v_max = np.max(df['mape_testing'])  # setting max mape_testing to guarantee heatmaps scale consistency
    v_min = np.min(df['mape_testing'])  # setting min mape_testing to guarantee heatmaps scale consistency
    df_filtered = df.copy()
    # plots the heatmaps
    for ind, rho in enumerate(spectral_values):
        fig, ax = plt.subplots(figsize=(19, 10))  # separate figs
        position = np.unravel_index(ind, (subplots_rows, subplots_cols))
        df = df_filtered[df_filtered['spectral_radius'] == rho]
        df = df[['ridge', 'leaky_factor', 'mape_testing', 'model']]
        df = df.sort_values(["leaky_factor", "ridge"], ascending=[False, True])
        result = df.pivot(index='leaky_factor', columns='ridge', values='mape_testing')
        if annotation is None:
            sns.heatmap(result, annot=True, vmin=v_min, vmax=v_max, fmt=".1f", cmap=cm.gray_r, ax=ax)  # separate figs
        else:
            labels = np.flip(annotation[ind, :, :], 0)
            sns.heatmap(result, annot=labels, vmin=v_min, vmax=v_max, fmt="", cmap=cm.gray_r, ax=ax)  # separate figs
        ax.set_xlabel(r'$\gamma$')
        ax.set_ylabel(r'$\alpha$')
        ax.set_xticklabels(["{:.0e}".format(x) for x in result.columns.values])
        ax.invert_yaxis()
        plt.tight_layout()
        if save_name is not None:
            new_save = f'{ind}_{save_name}'
            plt.savefig(os.path.join(path, new_save))
        plt.close()
    return None


def plot_alpha_ridge_heatmap_poincares(df, input_scale, density, spectral_values, subplots_rows, subplots_cols,
                                       y_test, u_test, sim_settings, path, save_name=None, axlim=None):
    df = df[(df['input_scale'] == input_scale) & (df['density'] == density)]
    df = df[df['spectral_radius'].isin(spectral_values)]
    df_filtered = df.copy()
    for ind, rho in enumerate(spectral_values):
        fig, ax = plt.subplots(subplots_rows, subplots_cols, figsize=(38, 18))
        df = df_filtered[df_filtered['spectral_radius'] == rho]
        df = df[['ridge', 'leaky_factor', 'mape_testing', 'model']]
        df = df.sort_values(["leaky_factor", "ridge"], ascending=[False, True])
        for iterate in range(0, subplots_rows * subplots_cols):
            print(iterate)
            mdl = df['model'].iloc[iterate]
            mdl_dict = json.loads(mdl)
            mdl = ESN.fromdict(mdl_dict)
            y_sim_test = mdl.simulate(u=u_test, y=y_test, burnout=sim_settings['burnout_simulate'],
                                      output_burnout=False,
                                      one_step_ahead=sim_settings['osa'])
            phase = 107
            position = np.unravel_index(iterate, (subplots_rows, subplots_cols))
            ax[position] = plot_poincare_section(x=y_sim_test[:, ...],
                                                 number_of_points=500,
                                                 phase=107,
                                                 save_name=f'{ind:d}_{position[0]}_{position[1]}_{phase:d}_{save_name}.jpg',
                                                 sim_settings=sim_settings,
                                                 delay=sim_settings['embedd_delay'],
                                                 path=path,
                                                 marker_size=0.1,
                                                 ax=ax[position],
                                                 axlim=axlim[ind, iterate],
                                                 heatmap=True)
        new_save_name = f'{ind:d}_{save_name}'
        plt.tight_layout()
        if save_name is not None:
            plt.savefig(os.path.join(path, new_save_name))
        plt.close()
    return None


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


def parallel_esn_grid_search(n_features_list, ridge_list, spectral_radius_list, input_scale_list, leaky_factor_list,
                             density_list, u_train,
                             y_train, u_test, y_test, seed_list, sim_settings):
    use_iteractive = sim_settings['use_iteractive']
    state_transform = sim_settings['state_transform']
    burnout_simulate = sim_settings['burnout_simulate']
    osa = sim_settings['osa']
    (n_features, ridge, radius, input_scale, leaky_factor, density) = (
        n_features_list, ridge_list, spectral_radius_list,
        input_scale_list, leaky_factor_list, density_list)
    monte_carlo = not (np.size(seed_list) == 1)
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
