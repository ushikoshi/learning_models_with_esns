import numpy as np
import scipy.linalg as linalg
from warnings import warn
from scipy import linalg as linalg
import scipy.sparse as sps
from ..dynamic_systems import prepare_ns, get_order, formulate_problem, simulate_nlss, \
    FeedbackOutputs
from typing import Union, List, Optional
from scipy.stats import uniform


class ESN(object):
    EPS = 1e-8

    def __init__(self, nus: Union[List[int], List[List[int]]], nys: Union[List[int], List[List[int]]],
                 n_features: int = 100, nonlinearity: str = 'tanh',
                 leaky_factor: float = 0.6, input_scale: float = 0,
                 spectral_radius: float = 0.95, density: float = 0.2,
                 ridge: float = 0.01, burnout: int = 0, state_transform: Optional[str] = None,
                 use_iterative: bool = False, random_state: int = 0,
                 n_inputs: Optional[int] = None, n_outputs: Optional[int] = None,
                 w_xy: Optional[np.ndarray] = None):
        # Save inputs
        self.nys = nys  # number of outputs
        self.nus = nus  # number of inputs
        self.prepared_nys = prepare_ns(nys)  # if nys = [[3]] then prepared_nys = [(0, 3)]. I do not know why
        self.prepared_nus = prepare_ns(nus)  # if nus = [[3]] then prepared_nus = [(0, 3)]. I do not know why
        self.n_eff_inputs = len(self.prepared_nus + self.prepared_nys)  # effective number of inputs to the model
        self.n_states = n_features  # the number of state variables is the number of reservoir neurons
        self.nonlinearity = nonlinearity  # uses tanh as default
        self.state_transform = state_transform  # applies a transformation to the model state, if desired
        self.leaky_factor = leaky_factor  # leaky factor from the literature
        self.spectral_radius = spectral_radius  # spectral radius from the literature
        self.density = density  # reservoir density (0 <= density <= 1). Fully connected = 1
        self.ridge = ridge  # penalization used in the ridge regression
        self.order = get_order(self.prepared_nus, self.prepared_nys)  # did not get it
        self.burnout_train = burnout  # it is a good practice to discard predictions during the training
        self.random_state = random_state  # defines a seed for the rng
        self.input_scale = input_scale  # input scale from the literature
        self.use_iterative = use_iterative  # did not get it
        # Get random states
        self.rng = np.random.RandomState(random_state)  # set rng seed
        # Set internal matrices
        # TODO: implement alternative scales (i.e. check neural network literatures)
        self.w_zx = self.get_input_to_state_matrix(self.rng, self.n_eff_inputs, self.n_states,
                                                   self.input_scale)  # randomly initialize Win uniformly
        error = True
        while error:  # redefines seed until there are no convergence issues
            # randomly init. sparse W uniformly in [-1, 1] also readjust the spectral radius as desired
            self.w_xx, error = self.get_state_to_state_matrix(self.rng, self.n_states, self.density,
                                                              self.spectral_radius)
            if error:
                self.random_state = self.random_state + 100
                self.rng = np.random.RandomState(self.random_state)

        # Get activation function
        if nonlinearity == 'tanh':
            self.activation = np.tanh
        else:
            raise ValueError('Invalid nonlinearity {}.'.format(nonlinearity))
        # Readouts
        self.w_xy = w_xy if isinstance(w_xy, np.ndarray) else np.array(w_xy)  # Will be assigned after fit...
        self.n_outputs = n_outputs  # Will be assigned after fit...
        self.n_inputs = n_inputs  # Will be assigned after fit...
        self.y_sim_train = None  # Will be assigned after fit...

    @classmethod
    def fromdict(cls, d):
        df = {k: v for k, v in d.items()}
        return cls(**df)

    @classmethod
    def get_input_to_state_matrix(cls, rng, n_eff_inputs, n_states, input_scale):
        """Intialize input to state matrix

        Initizalize input matrix the same way as in [1]: that is each reservoir
        node receive the input from a single input.
        [1] Lu, Zhixin, et al. "Reservoir observers: Model-free inference of
        unmeasured variables in chaotic systems." Chaos: An Interdisciplinary
        Journal of Nonlinear Science 27.4 (2017): 041102.
        """
        q = int(np.floor(n_states / n_eff_inputs))
        w = np.zeros((n_eff_inputs, n_states))
        for i in range(n_eff_inputs):
            w[i, i * q:(i + 1) * q] = input_scale * 2 * (rng.rand(1, q) - 0.5)
        return w

    @classmethod
    def get_state_to_state_matrix(self, rng, n_states, density, spectral_radius):
        w = sps.random(n_states, n_states, density=density, format='csc', random_state=rng,
                       data_rvs=lambda *x: uniform(loc=-1, scale=2).rvs(*x, random_state=rng))
        # Use iterative procedures to compute eigenvalue.
        # One alternative, more expensive but maybe more precise
        # is to use "radius = np.max(np.abs(linalg.eigvals(w.todense())))"!
        error = False
        try:
            radius = np.abs(sps.linalg.eigs(w, 1, which='LM', return_eigenvectors=False))[0]
        except sps.linalg.ArpackNoConvergence as e:
            warn("ArPack failed to completely converge. Don't readjust radius.")
            error = True
            return w, error
        if radius < self.EPS:
            warn("spectral radius too small (~{}). To avoid numerical errors the matrix is "
                 "not scaled and the specified spectral radius of {} will not be atained.".format(radius,
                                                                                                  spectral_radius))
            error = True
            return w, error
        # Limit spectral radius
        w = round(w * (spectral_radius / radius), 10)
        return w, error

    def get_x0(self, n_seq, n_states):
        return np.zeros((n_seq, n_states))

    # TODO: add x0 check here as well...
    def _check_input(self, u, y):
        try:
            seq_len1, n_seq1, n_inputs = u.shape
            if y is not None:
                seq_len2, n_seq2, n_outputs = y.shape
            else:
                seq_len2, n_seq2, n_outputs = seq_len1, n_seq1, self.n_outputs
        except:
            raise ValueError('data should be a tensor with dim = 3!')
        if seq_len1 != seq_len2:
            raise ValueError("The sequence length (the tensor dimension 0) "
                             "of the input u should be the same as that of the output y! "
                             "y.shape[0] = {} != u.shape[0] = {}"
                             .format(seq_len1, seq_len2))
        if n_seq1 != n_seq2:
            raise ValueError("The number of sequences (the tensor dimension 1) "
                             "of the input u should be the same as that of the output y! "
                             "y.shape[1] = {} != u.shape[1] = {}"
                             .format(n_seq1, n_seq2))
        if self.n_inputs is None:
            self.n_inputs = n_inputs
        elif n_inputs != self.n_inputs:
            raise ValueError("The number of inputs (the u tensor dimension 2) "
                             "does not match the expected dimension. "
                             "Expected dimension = {}. Received dimension = {}."
                             .format(n_inputs, self.n_inputs))
        if self.n_outputs is None:
            self.n_outputs = n_outputs
        elif n_outputs != self.n_outputs:
            raise ValueError("The number of outputs (the y tensor dimension 2) "
                             "does not match the expected dimension. "
                             "Expected dimension = {}. Received dimension = {}."
                             .format(n_outputs, self.n_outputs))

    def state_fn(self, x, z, identity_readouts=True):
        pre_activation = x @ self.w_xx + z @ self.w_zx  # eq.(1) of [1]
        post_activation = self.activation(pre_activation)
        if self.leaky_factor > 0:
            x_next = self.leaky_factor * x + (1 - self.leaky_factor) * post_activation
        else:
            x_next = post_activation
        y = self.extend_states(x, z) @ self.w_xy if (not identity_readouts and self.w_xy is not None) else x
        return y, x_next

    def extend_states(self, x, _z):
        if self.state_transform is None or self.state_transform == 'identity':
            return x
        elif self.state_transform == 'quadratic':
            x_new = x.copy()
            for i in range(1, x_new.shape[-1] - 1):
                if i % 2 != 0:
                    x_new[..., i] = x[..., i] * x[..., i]
            return x_new
        else:
            raise ValueError('Invalid state_transform')

    def fit(self, u, y, x0=None):
        self._check_input(u, y)
        # Estimate states
        z = formulate_problem(u, y, self.prepared_nus, self.prepared_nys)  # rearrange all the regressors
        seq_len, n_seq, n_combined_inputs = z.shape
        x0 = self.get_x0(n_seq, self.n_states) if x0 is None else x0  # initialize matrix to store the state trajectory
        x, _ = simulate_nlss(lambda x, z, k: self.state_fn(x, z, identity_readouts=True), z, x0)  # I did not get it
        # Reshape
        x, y = x.reshape(-1, self.n_states), y.reshape(-1, self.n_outputs)
        # Remove burnout
        x, y = x[self.burnout_train:, ...], y[self.order + self.burnout_train:, ...]
        # Extend states
        x = self.extend_states(x, z)
        # Estimate readout matrix
        self.w_xy, self.y_sim_train = solve_least_squares(x, y, ridge=self.ridge, use_lsmr=self.use_iterative)
        return self

    def predict(self, u, y=None, x0=None):
        self._check_input(u, y)
        # get initial state
        seq_len, n_seq, n_combined_inputs = u.shape
        x0 = self.get_x0(n_seq, self.n_states) if x0 is None else x0
        # Predict
        z = formulate_problem(u, y, self.prepared_nus, self.prepared_nys)
        x, _ = simulate_nlss(lambda x, z, k: self.state_fn(x, z, identity_readouts=True), z, x0)
        # Predict
        y_hat = self.extend_states(x, z) @ self.w_xy
        return y_hat

    def simulate(self, u, y=None, x0=None, one_step_ahead=False, burnout=200, output_burnout=True,
                 restart_every_n_sample=None):
        self._check_input(u, y)
        seq_len, n_seq, n_combined_inputs = u.shape
        burnout_simulate = burnout  # Just to distinguish more clearly from burnout train

        # In case of restarting every n use recursive implementation
        if restart_every_n_sample is not None:
            if y is None:
                raise ValueError('restart_every_n_sample functionality not implemented'
                                 'when y is none.')
            sample_from_before_start = max(burnout_simulate, self.order)
            list_outputs = []
            for start in range(sample_from_before_start, seq_len, restart_every_n_sample):
                # In the first iteration output the burnout period according to the variable output_burnout
                # in all the others, it doens't so it gets the right shape
                output_burnout_i = output_burnout if not list_outputs else False
                y_hat_i = self.simulate(u[start - sample_from_before_start:start + restart_every_n_sample],
                                        y[start - sample_from_before_start:start + restart_every_n_sample],
                                        x0, burnout=burnout_simulate, output_burnout=output_burnout_i)
                list_outputs.append(y_hat_i)
            return np.concatenate(list_outputs, axis=0)

        x0 = self.get_x0(n_seq, self.n_states) if x0 is None else x0
        # Predict first half of the data (burnout period)
        if burnout_simulate > self.order:
            z = formulate_problem(u[:burnout_simulate], y[:burnout_simulate], self.prepared_nus, self.prepared_nys)
            x, x_next = simulate_nlss(lambda x, z, k: self.state_fn(x, z, identity_readouts=True), z, x0)
        else:
            x_next = x0
        # Simulate second half of the data (burnout period)

        if one_step_ahead:
            z = formulate_problem(u[burnout_simulate:], y[burnout_simulate:], self.prepared_nus, self.prepared_nys)
            y_hat, _ = simulate_nlss(lambda x, z, k: self.state_fn(x, z, identity_readouts=False), z, x_next)
        else:
            freerun = FeedbackOutputs(lambda x, z, k: self.state_fn(x, z, identity_readouts=False),
                                      self.prepared_nus, self.prepared_nys, self.n_states)
            # Get initial state
            u0 = np.zeros((self.order, n_seq, self.n_inputs))
            y0 = np.zeros((self.order, n_seq, self.n_outputs))
            if burnout_simulate > 0:
                position = max(burnout_simulate - self.order, 0)
                n_terms = min(self.order, burnout_simulate)
                u0[-n_terms:, ...] = u[position:position + n_terms]  # last input used in burnout simulation
                y0[-n_terms:, ...] = y[position:position + n_terms]  # last output used in burnout simulation
            initial_states = freerun.get_initial_state(u0, y0, x_next)  # x_next is the last burnout state
            # simulate on test window
            y_hat, _ = simulate_nlss(freerun, u[burnout_simulate:, ...], initial_states)  # freerun: only input
        if output_burnout and burnout_simulate > self.order:
            y_hat0 = self.extend_states(x, z) @ self.w_xy
            y_hat = np.concatenate((y_hat0, y_hat), axis=0)
        return y_hat

    def __repr__(self):
        return "{}({},{},{},'{}','{}','{}','{}',{},{},{},{},{},{})".format(
            type(self).__name__, self.nus, self.nys, self.n_states, self.nonlinearity,
            self.leaky_factor, self.input_scale, self.spectral_radius, self.density,
            self.ridge, self.burnout_train, self.state_transform, self.use_iterative, self.random_state)


def solve_least_squares(X, y, ridge=0.0, use_lsmr=True):
    estim_param_list = []
    y_sim_train = []
    if use_lsmr:  # Use interative method
        for yi in y.T:
            estim_param_i = sps.linalg.lsmr(X, yi, damp=ridge, atol=1e-15, btol=1e-15, maxiter=10000)[0]
            estim_param_list.append(estim_param_i)
        estim_param = np.stack(estim_param_list, axis=-1)
        return estim_param

    if ridge <= 0:  # min norm solution
        estim_param, _resid, _rank, _s = linalg.lstsq(X, y)
    else:  # SVD implementation of ridge regression
        u, s, vh = linalg.svd(X, full_matrices=False, compute_uv=True)  # X = u diag(s) v.H
        prod_aux = s / (ridge + s ** 2)  # diag(prod_aux) = inv(diag(s).T diag(s) + ridge * I) diag(s).T)
        # estim_param = V diag(prod_aux) U.T y  (which is equivalent to what is bellow, but would be less efficient)
        estim_param = ((prod_aux * (y.T @ u)) @ vh).T
        y_sim_train = X @ estim_param
        y_sim_train = y_sim_train.reshape((-1, 1, 1))
    return estim_param, y_sim_train
