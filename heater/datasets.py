import math
import numpy as np
import scipy.signal
from scipy.integrate import solve_ivp
import h5py
import matplotlib.pyplot as plt
import csv


def dvDuffing(x, u, t):
    k = 0.1
    xd = np.zeros((2, 1))
    xd[0] = x[1]
    xd[1] = - k * x[1] - x[0] ** 3 + u
    return xd


def rkDuffing(x0, u, t, dt=np.pi/60):
    # 1st call
    xd = dvDuffing(x0, u, t)
    savex0 = x0.reshape((-1, 1))
    phi = xd
    x0 = savex0 + np.array(0.5 * dt * xd)
    # 2nd call
    xd = dvDuffing(x0, u, t + 0.5 * dt)
    phi = phi + 2 * xd
    x0 = savex0 + 0.5 * dt * xd
    # 3rd call
    xd = dvDuffing(x0, u, t + 0.5 * dt)
    phi = phi + 2 * xd
    x0 = savex0 + dt * xd
    # 4th call
    xd = dvDuffing(x0, u, t + dt)
    x = savex0 + (phi + xd) * dt / 6
    return x


def inp_duffing(t, train_time=200, test_time=100, trans_time=200, dt=np.pi/3000, ss_amp=6.5):
    n_trans = int(trans_time / dt)
    n_train = int(train_time / dt)
    n_test = int(test_time / dt)
    n_u = n_train + n_test + 2 * n_trans
    u = np.zeros((n_u + 1, 1))
    for i in range(len(u)):
        noise_flag = 0
        if i <= n_trans:
            u[i] = ss_amp * np.cos(t[i])
        elif i <= n_trans + int(n_train/2):
            if noise_flag == 0:
                noise = np.random.normal(0, math.sqrt(2), 1)
                noise_flag = 3
            noise_flag = noise_flag - 1
            amp = 0.5 + 12.5 * ((t[i] - t[n_trans]) / (t[n_trans + int(n_train/2)] - t[n_trans]))
            u[i] = amp * scipy.signal.square(t[i], duty=0.5) + noise
        elif i <= n_trans + n_train:
            if noise_flag == 0:
                noise = np.random.normal(0, math.sqrt(2), 1)
                noise_flag = 3
            noise_flag = noise_flag - 1
            amp = 13.5 - 13 * ((t[i] - t[n_trans + int(n_train/2)]) / (t[n_trans + n_train]
                                                                       - t[n_trans + int(n_train/2)]))
            u[i] = amp * scipy.signal.square(t[i], duty=0.5) + noise
        else:
            u[i] = ss_amp * np.cos(t[i])
    return u


def get_dataset(train_time=200, test_time=100, trans_time=200, dt=np.pi/3000, decimate=50,
                ss_amp=6.5, plot_train_solution=True, plot_test_solution=True):
    n_trans = int(trans_time / dt)
    n_train = int(train_time / dt)
    n_test = int(test_time / dt)
    n_total = n_train + n_test + 2 * n_trans
    t_f = n_total
    t = np.arange(0, t_f + 1, 1) * dt
    points = np.linspace(1, len(t)-1, len(t)-1, dtype='int64')

    # initial conditions
    # np.random.seed(0)
    # x_0 = np.random.rand(2) - 0.5
    x_0 = np.array([0.1, 0.1])

    # trajectory array. Row 0 is variable x and row is variable y
    x = np.array(np.zeros((len(x_0), len(t))))
    x[:, 0] = x_0

    # external signal
    # u = np.array(np.zeros((1, len(t))))
    # u = 6.5 * np.cos(t)
    u = inp_duffing(t, train_time, test_time, trans_time, dt, ss_amp)

    # fourth-order Runge-Kutta integration
    for k in points:
        x[:, [k]] = rkDuffing(x[:, k - 1], u[k], t[k], dt)

    # Put it into the correct shape
    # n_transient = int(transient_time / dt)
    # t = t[n_transient:]
    y = x[0, :]
    # u = u.reshape((-1, 1))
    # u = u[n_transient:, :]
    y = y.reshape((-1, 1, 1))
    u = u.reshape((-1, 1, 1))

    # Get training and test output
    train_time = t[n_trans+1:n_trans+1+n_train+1]
    test_time = t[n_trans+1+n_train+1+n_trans+1:]
    y_train = y[n_trans+1:n_trans+1+n_train+1, :, :]
    y_test = y[n_trans+1+n_train+1+n_trans+1:, :, :]
    u_train = u[n_trans+1:n_trans+1+n_train+1, :, :]
    u_test = u[n_trans+1+n_train+1+n_trans+1:, :, :]

    # training_data = np.concatenate((u_train, y_train), axis=1)
    # training_data = training_data[:, :, 0]
    # np.save("duffing_training_data.npy", training_data)
    # np.savetxt('duffing_training_data.csv', training_data, delimiter=',')

    # Get training and test output
    # n_train = int(train_ratio * len(y))
    # train_time = t[:n_train]
    # test_time = t[n_train:]
    # y_train = y[:n_train, :, :]
    # y_test = y[n_train:, :, :]
    # u_train = u[:n_train, :, :]
    # u_test = u[n_train:, :, :]

    # Decimation
    train_time = train_time[::decimate]
    test_time = test_time[::decimate]
    y_train = y_train[::decimate]
    y_test = y_test[::decimate]
    u_train = u_train[::decimate]
    u_test = u_test[::decimate]

    # plotting training and testing series
    if plot_train_solution is True:
        fig, ax = plt.subplots(2)
        ax[0].plot(train_time, u_train[:, :, 0])
        ax[1].plot(train_time, y_train[:, :, 0])
        plt.show()
    if plot_test_solution is True:
        fig, ax = plt.subplots(2)
        ax[0].plot(test_time, u_test[:, :, 0])
        ax[1].plot(test_time, y_test[:, :, 0])
        plt.show()
    return y_train, y_test, u_train, u_test

