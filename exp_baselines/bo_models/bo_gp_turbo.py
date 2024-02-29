from bo_gp_class import BO_optimization
from dataclasses import dataclass
import math
import torch
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.models import SingleTaskGP
import pandas as pd

MAX_CHOLESKY_SiZE = float("inf")

def update_candidates(candidates, kw_type):
    for idx, key in enumerate(kw_type.keys()):
        if type(kw_type[key]) == list or kw_type[key] == 'int':
            candidates[:, idx] = candidates[:, idx].round()
        elif kw_type[key] == 'float':
            candidates[:, idx] = candidates[:, idx]
    return candidates

def list_to_numpy(list_config, kw_type):
    new_list = []
    for this_dict in list_config:
        new_dict = {}
        for key in kw_type.keys():
            if not (type(this_dict[key]) == str):
                new_dict[key] = this_dict[key]
            else:
                new_dict[key] = kw_type[key].index(this_dict[key])
        new_list += [new_dict]
    return new_list

@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )

def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state

class BO_optimization_Turbo(BO_optimization):
    def __init__(self, fun_to_evaluate, config_space, order_list, seed = 0):
        super().__init__(fun_to_evaluate, config_space, order_list, seed)

    def get_model(self, init_x, init_y):
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
            MaternKernel(
                nu=2.5, ard_num_dims=init_x.shape[-1], lengthscale_constraint=Interval(0.005, 1.0)
            )
        )
        model = SingleTaskGP(init_x, init_y, covar_module=covar_module, likelihood=likelihood)
        return model, likelihood

    def update_bounds_turbo(self, init_x, init_y, model, state):
        x_center = init_x[init_y.argmax(), :].clone()
        weights  = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights  = weights / weights.mean()
        weights  = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb    = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
        tr_ub    = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)
        return torch.stack([tr_lb, tr_ub])

    def get_next_points(self, init_x, init_y, best_init_y, state):
        init_x, init_y, bounds, x_max, x_min =  self.prepare_data_input(init_x, init_y, best_init_y)

        single_model, likelihood = self.get_model(init_x, init_y)
        with gpytorch.settings.max_cholesky_size(MAX_CHOLESKY_SiZE):
            single_model             = self.maximize_likelihood(single_model, likelihood)
            bounds                   = self.update_bounds_turbo(init_x, init_y, single_model, state)
            acq_function             = self.obtain_acq(single_model, best_init_y)
            candidates               = self.obtain_candidates(acq_function, bounds)
            candidates               = self.prepare_data_output(candidates, x_max, x_min)
        return candidates

    def optimize(self, n_runs, n_init, config_init = None):
        self.fun_to_evaluate.reset_results()
        init_x, init_y, best_init_y = self.generate_initial_data(n_init = n_init,
                                        train_x = pd.DataFrame(self.bo_utils.list_to_numpy(config_init[:n_init])).to_numpy())

        dim        = init_x.shape[-1]
        batch_size = 1
        state      = TurboState(dim, batch_size=batch_size)
        x_before   = torch.tensor([])
        for i in range(n_runs):
            new_candidates              = self.get_next_points(init_x, init_y, best_init_y, state)
            x_before                    = torch.cat([x_before, new_candidates])
            new_candidates              = self.bo_utils.update_candidates(new_candidates)
            new_results                 = self.loop_model(new_candidates)

            state                       = update_state(state=state, Y_next = new_results)
            init_x                      = torch.cat([init_x, new_candidates])
            init_y                      = torch.cat([init_y, torch.tensor(new_results).reshape(1,1)]).float()
            best_init_y                 = init_y.max().item()
        return init_y, pd.DataFrame(self.fun_to_evaluate.all_results)