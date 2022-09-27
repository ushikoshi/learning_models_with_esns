import numpy as np
import os
import running_functions_duffing as run
import pandas as pd
import json
import matplotlib.pyplot as plt
from reservoir.model import ESN

plt.rcParams.update({'font.size': 42})  # 42

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
    results_folder = os.path.join(current_directory, experiment_name)

    if sim_settings['collect_data']:  # collecting dataset for training and testing
        y_train, u_train, y_test, u_test = run.collect_dataset(sim_settings=sim_settings, path=results_folder)
        np.save('data/y_test_validation_5000.npy', y_test)
        np.save('data/u_test_validation_5000.npy', u_test)
    else:  # using saved dataset for training and testing
        y_train = np.load('data/y_train.npy')
        u_train = np.load('data/u_train.npy')
        y_test = np.load('data/y_test_63.npy')
        u_test = np.load('data/u_test_63.npy')

    # plotting training input
    fig, ax = plt.subplots(figsize=(19, 9))
    ax.step(u_train[:, 0, 0], 'k-')
    ax.set_xlim(0, 1910)
    ax.set_xlabel(r'$k$')
    ax.set_ylim(-18, 18)
    ax.set_xticks([0, 955, 1910])
    ax.set_yticks([-17, 0, 17])
    plt.show()

    # plotting training output
    fig, ax = plt.subplots(figsize=(19, 9))
    ax.plot(y_train[:, 0, 0], 'k-')
    ax.set_xlim(0, 1910)
    ax.set_xlabel(r'$k$')
    ax.set_ylim(-5, 5)
    ax.set_xticks([0, 955, 1910])
    ax.set_yticks([-4.8, 0, 4.3])
    plt.show()

    # loading results dataframe
    raw_data = pd.read_parquet(os.path.join(results_folder, 'results_dataframe_complete.parquet.gzip'))
    if sim_settings['invert_alpha']:
        raw_data['leaky_factor'] = round(1 - raw_data.loc[:, 'leaky_factor'], 10)

    # Loading validation data
    y_validation = np.load('data/y_test_validation_5000.npy')
    u_validation = np.load('data/u_test_validation_5000.npy')
    seeds = [5]  # choosing seed

    # getting best model and its HPs
    if os.path.exists(os.path.join(results_folder, "best_results.txt")):
        os.remove(os.path.join(results_folder, "best_results.txt"))
    for _, seed in enumerate(seeds):
        data = raw_data[raw_data['seed'] == seed]
        best_model = run.get_best_model(df=data, input_scale=0.1, density=0.05, model_selected=0, show_k_best=False,
                                        k_best=10)
        best_rho = best_model.spectral_radius
        if sim_settings['invert_alpha']:
            best_alpha = round(1 - best_model.leaky_factor, 10)
        else:
            best_alpha = best_model.leaky_factor
        best_ridge = best_model.ridge
        with open(os.path.join(results_folder, 'best_results.txt'), 'a') as f:
            f.write(f'\n seed: {seed}, best rho: {best_rho:.3e}, '
                    f'best alpha: {best_alpha:.2f}, '
                    f'best ridge: {best_ridge:.3e}\n')

    # plotting best model training time series
    y_sim_train = best_model.simulate(u=u_train, y=y_train, burnout=sim_settings['burnout_simulate'],
                                      one_step_ahead=sim_settings['osa'], output_burnout=False)
    _, _ = run.plot_time_series(sim_settings=sim_settings,
                                y_ref=y_train[best_model.order + best_model.burnout_train:, 0, 0],
                                y_hat=y_sim_train[:, 0, 0], path=results_folder,
                                save_name=f'best_model_training_time_series_seed{seed}.eps')

    # plotting best model testing time series
    y_sim_test = best_model.simulate(u=u_test, y=y_test, burnout=sim_settings['burnout_simulate'],
                                     one_step_ahead=sim_settings['osa'], output_burnout=False)
    _, _ = run.plot_time_series(sim_settings=sim_settings,
                                y_ref=y_test[best_model.burnout_train+1500:best_model.burnout_train + 1500+1500, 0, 0],
                                y_hat=y_sim_test[1500:1500+1500, 0, 0], path=results_folder,
                                save_name=f'best_model_testing_time_series_seed{seed}.eps')

    # plotting the original phase portrait
    _, _ = run.plot_phase_portrait(x=y_validation[best_model.burnout_train:, ...], number_of_samples=5000,
                                   path=results_folder,
                                   save_name=f'original_phase_portrait.eps',
                                   sim_settings=sim_settings, delay=sim_settings['embedd_delay'],
                                   y=None)
    # plotting best model phase portraits (on test data)
    y_sim_test = best_model.simulate(u=u_test, y=y_test, burnout=sim_settings['burnout_simulate'],
                                     one_step_ahead=sim_settings['osa'], output_burnout=False)
    _, _ = run.plot_phase_portrait(x=y_sim_test[:, ...], number_of_samples=500, path=results_folder,
                                   save_name=f'best_model_phase_portrait_seed{seed}.eps',
                                   sim_settings=sim_settings,
                                   delay=sim_settings['embedd_delay'], y=None)

    # plotting original poincare section
    _ = run.plot_poincare_section(sim_settings=sim_settings, x=y_validation[best_model.burnout_train:, ...],
                                  path=results_folder,
                                  save_name=f'original_poincare_section.eps',
                                  number_of_points=5000, phase=107,
                                  delay=sim_settings['embedd_delay'], y=None, ax=None,
                                  marker_size=0.25)

    # Plotting best model Poincare section (on validation data)
    plt.close('all')
    chosen_seed = 1
    y_sim_test = best_model.simulate(u=u_validation, y=y_validation, burnout=sim_settings['burnout_simulate'],
                                     output_burnout=False, one_step_ahead=sim_settings['osa'])
    _ = run.plot_poincare_section(x=y_sim_test[:, ...],
                                  y=None,
                                  number_of_points=500,
                                  phase=107,
                                  save_name=f'best_model_poincare_section_seed{chosen_seed}.jpg',
                                  sim_settings=sim_settings,
                                  delay=sim_settings['embedd_delay'],
                                  path=results_folder,
                                  marker_size=0.25,
                                  ax=None,
                                  heatmap=False)

    # Creating the alpha-ridge heatmap plots for chosen seed and density
    chosen_seed = 5
    data = raw_data[raw_data['seed'] == chosen_seed]
    radius_values = [0.1, 0.3, 0.6, 0.9]
    # # seed 1
    # # annotation = np.asarray([
    # #     [['P2*', 'S*', 'S*', 'P1*', 'P1*'],
    # #      ['S*', 'P2*', 'P1*', 'P1*', 'P1'],
    # #      ['P1*', 'P1*', 'P3', 'P1', 'P1'],
    # #      ['C', 'C', 'P7', 'P1', 'P1'],
    # #      ['P2', 'C', 'P1', 'P1', 'P1']],
    # #     [['P7*', 'S*', 'Q*', 'P1*', 'P1*'],
    # #      ['P2*', 'P1*', 'P1*', 'P1*', 'P1'],
    # #      ['P2*', 'P1*', 'P1*', 'P1*', 'P1'],
    # #      ['P2', 'C', 'P4', 'P1', 'P1'],
    # #      ['C*', 'S', 'P1', 'P1', 'P1']],
    # #     [['P2*', 'P2*', 'P1*', 'P1*', 'P1*'],
    # #      ['P1*', 'P1*', 'P1*', 'P1*', 'P1'],
    # #      ['P1*', 'P1*', 'P1*', 'P1*', 'P1'],
    # #      ['P1*', 'P1*', 'P1', 'P4', 'P1'],
    # #      ['P12*', 'S*', 'P1*', 'P1', 'P1']],
    # #     [['P1*', 'P1*', 'P1*', 'P1*', 'P1'],
    # #      ['P1*', 'P1*', 'P1*', 'P1*', 'P1'],
    # #      ['P1*', 'P1*', 'P1*', 'P1', 'P1'],
    # #      ['Q*', 'Q*', 'S*', 'P1*', 'P1*'],
    # #      ['P2*', 'Q*', 'S*', 'P1', 'P1']]
    # # ])
    # seed 5
    annotation = np.asarray([
        [['P1*', 'P1*', 'Q*', 'P1*', 'P1*'],
         ['P1*', 'P1*', 'P1*', 'P1*', 'P1'],
         ['P1*', 'P1*', 'Q', 'P1', 'P1'],
         ['C', 'P2', 'Q*', 'P1', 'P1'],
         ['P1', 'C', 'P5', 'P1', 'P1']],
        [['P1*', 'Q*', 'P1*', 'P1*', 'P1*'],
         ['P1*', 'P5*', 'P1*', 'P1*', 'P1'],
         ['P1*', 'P1*', 'P1*', 'P1', 'P1'],
         ['C', 'C', 'P1', 'P1', 'P1'],
         ['P1*', 'P1', 'P1', 'P1', 'P1']],
        [['P1*', 'P1*', 'P1*', 'P1*', 'P1*'],
         ['P1*', 'P1*', 'P1*', 'P1*', 'P1'],
         ['P1*', 'P1*', 'P1*', 'P1*', 'P1'],
         ['C', 'C', 'C', 'P1', 'P1'],
         ['C*', 'P4*', 'C', 'P1', 'P1']],
        [['P1*', 'P1*', 'P1*', 'P1*', 'P1'],
         ['P1*', 'P1*', 'P1*', 'P1*', 'P1'],
         ['P1*', 'P1*', 'P1*', 'P1', 'P1'],
         ['Q*', 'Q*', 'P3*', 'P1*', 'P1*'],
         ['P2', 'C', 'P1*', 'P1*', 'P1*']]
    ])
    run.plot_alpha_ridge_heatmaps(df=data,
                                  input_scale=0.1,
                                  density=0.05,
                                  spectral_values=radius_values,
                                  subplots_rows=2,
                                  subplots_cols=2,
                                  path=results_folder,
                                  clip_upper=100,
                                  save_name=f'heatmap_seed{chosen_seed}.eps',
                                  annotation=annotation)

    # Validation poincare section plots (for every frame on the heatmaps)
    chosen_seed = 5
    data = raw_data[raw_data['seed'] == chosen_seed]
    radius_values = [0.1, 1.2, 2, 3, 0.9]
    # axis limits seed 1
    # axlim = np.asarray([
    #     [[None], [None], [None], [-2.18e1, -2.16e1, -21.7, -21.68], [3.3e-1, 3.32e-1, 0.47, 0.49],
    #      [None], [None], [1.04e1, 1.06e1, 10.45, 10.47], [None], [4.43e-1, 4.45e-1, 0.56, 0.58],
    #      [1.22e1, 1.24e1, 12.33, 12.35], [6.11, 6.13, 6.11, 6.13], [None], [1.30, 1.32, 0.91, 0.93], [6.47e-1, 6.49e-1, 0.73, 0.75],
    #      [None], [None], [None], [1.73, 1.75, 1.20, 1.22], [1.03, 1.05, 1.03, 1.05],
    #      [None], [None], [2.62, 2.64, 2.71, 2.73], [1.04, 1.06, 1.10, 1.12], [1.16, 1.18, 1.14, 1.16]],
    #     [[None], [None], [None], [2.54e1, 2.56e1, 25.44, 25.46], [-1.04e1, -1.02e1, -10.3, -10.28],
    #      [None], [None], [1.38e1, 1.4e1, 13.86, 13.88], [1.35e1, 1.37e1, 13.52, 13.54], [9.17e-1, 9.19e-1, 0.88, 0.90],
    #      [None], [9.80, 9.82, 9.82, 9.84], [5.29, 5.31, 5.24, 5.26], [8.90e-1, 8.92e-1, -0.09, -0.07], [1.04, 1.06, 0.92, 0.94],
    #      [None], [None], [None], [3.17, 3.19, 2.17, 2.19], [1.00, 1.02, 0.79, 0.81],
    #      [-2.1, -1.9, -2.8, -2.6], [None], [1.65, 1.67, 3.16, 3.18], [1.07, 1.09, 0.98, 1.00], [9.08e-1, 9.10e-1, 0.81, 0.83]],
    #     [[None], [None], [2.80e1, 2.82e1, 28.13, 28.15], [2.75e1, 2.77e1, 27.59, 27.61], [-1.34e1, -1.32e1, -13.25, -13.23],
    #      [None], [None], [1.85e1, 1.87e1, 18.6, 18.62], [1.55e1, 1.57e1, 15.55, 15.57], [0.99, 1.01, 0.86, 0.88],
    #      [1.27e1, 1.29e1, 12.74, 12.76], [1.24e1, 1.26e1, 12.45, 12.47], [8.28, 8.30, 8.17, 8.19], [3.55, 3.57, 4.04, 4.06], [9.4e-1, 9.42e-1, 0.74, 0.76],
    #      [4.07e-1, 4.09e-1, -1.27, -1.25], [-3.19e-2, -3.17e-2, -1.58, -1.56], [2.31, 2.33, 3.67, 3.69], [None], [1.07, 1.09, 0.84, 0.86],
    #      [None], [None], [4.56, 4.58, 3.19, 3.21], [1.39, 1.41, 2.43, 2.45], [7.18e-1, 7.2e-1, 0.73, 0.75]],
    #     [[3.03e1, 3.05e1, 30.46, 30.48], [3.03e1, 3.05e1, 30.46, 30.48], [3.17e1, 3.19e1, 31.81, 31.83], [2.37e1, 2.39e1, 23.79, 23.81], [1.02, 1.04, 1.39, 1.41],
    #      [2.39e1, 2.41e1, 24.05, 24.07], [2.39e1, 2.41e1, 24.06, 24.08], [2.57e1, 2.59e1, 25.85, 25.87], [1.61e1, 1.63e1, 16.28, 16.30], [5.48e-1, 5.5e-1, 1.13, 1.15],
    #      [1.83e1, 1.85e1, 18.14, 18.16], [1.83e1, 1.85e1, 18.16, 18.18], [1.84e1, 1.86e1, 18.53, 18.55], [1.15, 1.17, 1.24, 1.26], [-4.26e-2, -4.24e-2, 0.63, 0.65],
    #      [None], [None], [None], [4.48, 4.5, 3.94, 3.96], [2.09e-1, 2.11e-1, 0.26, 0.28],
    #      [None], [None], [None], [2.59e-2, 2.61e-2, 2.05, 2.07], [6.13e-1, 6.15e-1, 0.52, 0.54]],
    # ])
    # axis limits seed 1 - 0.3 and 0.6
    # axlim = np.asarray([
    #     [[None], [None], [None], [2.54e1, 2.56e1, 25.44, 25.46], [-1.04e1, -1.02e1, -10.3, -10.28],
    #      [None], [-8, -6, 10, 12], [1.38e1, 1.4e1, 13.86, 13.88], [1.35e1, 1.37e1, 13.52, 13.54], [9.17e-1, 9.19e-1, 0.88, 0.90],
    #      [None], [9.80, 9.82, 9.82, 9.84], [5.29, 5.31, 5.24, 5.26], [8.90e-1, 8.92e-1, -0.09, -0.07], [1.04, 1.06, 0.92, 0.94],
    #      [None], [None], [None], [3.17, 3.19, 2.17, 2.19], [1.00, 1.02, 0.79, 0.81],
    #      [-2.1, -1.9, -2.8, -2.6], [None], [1.65, 1.67, 3.16, 3.18], [1.07, 1.09, 0.98, 1.00], [9.08e-1, 9.10e-1, 0.81, 0.83]],
    #     [[None], [None], [2.80e1, 2.82e1, 28.13, 28.15], [2.75e1, 2.77e1, 27.59, 27.61], [-1.34e1, -1.32e1, -13.25, -13.23],
    #      [15, 17, 15, 17], [15, 17, 15, 17], [1.85e1, 1.87e1, 18.6, 18.62], [1.55e1, 1.57e1, 15.55, 15.57], [0.99, 1.01, 0.86, 0.88],
    #      [1.27e1, 1.29e1, 12.74, 12.76], [1.24e1, 1.26e1, 12.45, 12.47], [8.28, 8.30, 8.17, 8.19], [3.55, 3.57, 4.04, 4.06], [9.4e-1, 9.42e-1, 0.74, 0.76],
    #      [4.07e-1, 4.09e-1, -1.27, -1.25], [-3.19e-2, -3.17e-2, -1.58, -1.56], [2.31, 2.33, 3.67, 3.69], [None], [1.07, 1.09, 0.84, 0.86],
    #      [None], [None], [4.56, 4.58, 3.19, 3.21], [1.39, 1.41, 2.43, 2.45], [7.18e-1, 7.2e-1, 0.73, 0.75]],
    # ])
    # axis limits seed 5
    axlim = np.asarray([
        [[-1.61e2, -1.59e2, -160, -158], [4.71e1, 4.73e1, 4.71e1, 4.73e1], [None], [-2.42e1, -2.4e1, -24.06, -24.04], [3.2e-1, 3.4e-1, 0.47, 0.49],
         [-2.72e2, -2.7e2, -271.17, -271.15], [1.71e1, 1.73e1, 17.2, 17.22], [1.14e1, 1.16e1, 11.5, 11.52], [None], [4.41e-1, 4.43e-1, 0.57, 0.59],
         [1.11e1, 1.13e1, 11.24, 11.26], [6.88, 6.9, 6.88, 6.9], [None], [1.88, 1.9, 1.6, 1.62], [6.48e-1, 6.5e-1, 0.75, 0.77],
         [None], [None], [None], [1.98, 2, 1.82, 1.84], [1.02, 1.04, 1.04, 1.06],
         [2.8, 3, 3.6, 3.8], [None], [None], [1.05, 1.07, 1.13, 1.15], [1.14, 1.16, 1.13, 1.15]],
        [[3.78e1, 3.8e1, 37.98, 38], [None], [2.79e1, 2.81e1, 28, 28.02], [3.07e1, 3.09e1, 30.78, 30.8], [-1.14e1, -1.12e1, -11.31, -11.29],
         [2.29e1, 2.31e1, 22.9, 22.92], [None], [1.67e1, 1.69e1, 16.8, 16.82], [1.62e1, 1.64e1, 16.24, 16.26], [9.83e-1, 9.85e-1, 1.01, 1.03],
         [1.37e1, 1.39e1, 13.87, 13.89], [9.91, 9.93, 9.91, 9.93], [7.13, 7.15, 7.10, 7.12], [1.32, 1.35, -0.01, 0.01], [1.15, 1.17, 1.1, 1.12],
         [None], [None], [2, 3, 3, 4], [3.41, 3.43, 3.06, 3.08], [1.06, 1.08, 0.95, 0.97],
         [-2.05, -2.03, -2.20, -2.18], [1.2, 1.22, 0.89, 0.91], [1.42, 1.44, 2.52, 2.54], [1.17, 1.19, 1.17, 1.19], [8.39e-1, 8.41e-1, 0.78, 0.8]],
        [[3.36e1, 3.38e1, 33.72, 33.74], [3.38e1, 3.4e1, 33.92, 33.94], [3.7e1, 3.72e1, 37.13, 37.15], [3.03e1, 3.05e1, 30.42, 30.44], [-1.44e1, -1.42e1, -14.27, -14.25],
         [1.98e1, 2e1, 19.95, 19.97], [1.99e1, 2.01e1, 20.03, 20.05], [2.35e1, 2.37e1, 23.59, 23.61], [1.73e1, 1.75e1, 17.36, 17.38], [1.34, 1.36, 1.15, 1.17],
         [1.16e1, 1.18e1, 11.65, 11.67], [1.15e1, 1.17e1, 11.56, 11.58], [1.15e1, 1.17e1, 11.51, 11.53], [8.77e-1, 8.79e-1, 0.32, 0.34], [1.29, 1.31, 0.99, 1.01],
         [None], [None], [None], [3.14, 3.16, 1.25, 1.27], [1.54, 1.56, 1.31, 1.33],
         [None], [None], [None], [1.17, 1.19, 2.01, 2.03], [6.73e-1, 6.75e-1, 0.76, 0.78]],
        [[3.16e1, 3.18e1, 31.64, 31.66], [3.16e1, 3.18e1, 31.64, 31.66], [3.18e1, 3.2e1, 31.85, 31.87], [2.26e1, 2.28e1, 22.65, 22.67], [2.02, 2.04, 1.54, 1.56],
         [2.84e1, 2.86e1, 28.48, 28.5], [2.84e1, 2.86e1, 28.48, 28.5], [2.74e1, 2.76e1, 27.5, 27.52], [1.56e1, 1.58e1, 15.69, 15.71], [1.83, 1.85, 1.62, 1.64],
         [1.61e1, 1.63e1, 16.1, 16.12], [1.61e1, 1.63e1, 16.1, 16.12], [1.52e1, 1.54e1, 15.21, 15.23], [4, 6, 3, 5], [1.11, 1.13, 0.83, 0.85],
         [None], [None], [None], [0.99, 1.01, 0.24, 0.26], [1.06, 1.08, 0.28, 0.3],
         [None], [None], [-3.22, -3.2, -4.17, -4.15], [-2.14, -2.12, -0.42, -0.4], [4.02e-1, 4.04e-1, 0.43, 0.45]],
    ])
    run.plot_alpha_ridge_heatmap_poincares(df=data,
                                           input_scale=0.1,
                                           density=0.05,
                                           spectral_values=radius_values,
                                           subplots_rows=5,
                                           subplots_cols=5,
                                           y_test=y_validation,
                                           u_test=u_validation,
                                           sim_settings=sim_settings,
                                           path=results_folder,
                                           axlim=axlim,
                                           save_name=f'poincare_heatmap_seed{chosen_seed}.jpg')

    # plotting a chosen Poincare section and respective portrait phase
    chosen_seed = 5
    plt.close('all')
    data = raw_data[(raw_data['n_features'] == 200) &
                    (round(raw_data['leaky_factor'], 2) == 0.3) &
                    (raw_data['spectral_radius'] == 1e-1) &
                    (raw_data['ridge'] == 1e-6) &
                    (raw_data['seed'] == 1) &
                    (raw_data['density'] == 0.05) &
                    (raw_data['input_scale'] == 0.1)]
    mdl = data['model'].iloc[0]
    mdl_dict = json.loads(mdl)
    mdl = ESN.fromdict(mdl_dict)
    y_sim_test = mdl.simulate(u=u_validation, y=y_validation, burnout=sim_settings['burnout_simulate'],
                              output_burnout=False, one_step_ahead=sim_settings['osa'])
    _ = run.plot_poincare_section(x=y_sim_test[:, ...],
                                  y=None,
                                  number_of_points=5000,
                                  phase=107,
                                  save_name=f'poincare_rho_6e-1_alpha_03_ridge_1e-6_seed{chosen_seed}.eps',
                                  sim_settings=sim_settings,
                                  delay=sim_settings['embedd_delay'],
                                  path=results_folder,
                                  marker_size=0.25,
                                  ax=None,
                                  heatmap=False)
    _, _ = run.plot_phase_portrait(x=y_sim_test[:, ...], number_of_samples=5000, path=results_folder,
                                   save_name=f'good_model_phase_portrait_seed{chosen_seed}.eps',
                                   sim_settings=sim_settings,
                                   delay=sim_settings['embedd_delay'], y=None)