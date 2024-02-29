import numpy as np
import pandas as pd
import torch
from botorch.models import SingleTaskGP, MixedSingleTaskGP 
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from botorch import fit_gpytorch_model
from sampler import sample_configurations
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.models import ExactGP
from gpytorch.constraints import Interval

from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.optim import optimize_acqf_mixed
from torch import nn


class CustomGP(ExactGP, GPyTorchModel):
        _num_outputs = 1  # to inform GPyTorchModel API
        num_outputs = 1  # to inform GPyTorchModel API

        def __init__(self, train_x, train_y, likelihood):
            super(CustomGP, self).__init__(train_x, train_y.squeeze(-1), likelihood)
            self.mean_module = ConstantMean()
            self.covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                MaternKernel(
                    nu=2.5, ard_num_dims = train_x.shape[-1], lengthscale_constraint=Interval(0.005, 4.0)
                )
            )            
        def forward(self, x):
            mean_x      = self.mean_module(x)
            covar_x     = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)


class BO_utils:
    def __init__(self, config_space, order_list):
        self.hyp_opt, self.hyp_type, self.hyp_log = self.extract_space_hyp(config_space, order_list)

    def extract_space_hyp(self, config_space, order_list):
        import ConfigSpace as CS
        config_info = []
        hyp_opt   = {}
        hyp_type  = {}
        hyp_log   = {}
        for hp_name in order_list:
            hp = config_space.get_hyperparameter(hp_name)
            if isinstance(hp, CS.CategoricalHyperparameter):
                choices            = hp.choices
                hyp_opt[hp_name]   = list(np.arange(len(choices)))
                hyp_type[hp_name]  = list(choices)
                hyp_log[hp_name]   = False
            elif isinstance(hp, CS.UniformFloatHyperparameter):
                hyp_opt[hp_name]   = [hp.lower, hp.upper]
                hyp_type[hp_name]  = 'float'
                hyp_log[hp_name]   = hp.log
            elif isinstance(hp, CS.UniformIntegerHyperparameter):
                hyp_opt[hp_name]   = [hp.lower, hp.upper]
                hyp_type[hp_name]  = 'int'
                hyp_log[hp_name]   = hp.log
        return hyp_opt, hyp_type, hyp_log        

    def obtain_idx(self):
        list_idx = []
        for idx, key in enumerate(self.hyp_type.keys()):
            if type(self.hyp_type[key]) == list:
                list_idx += [idx]
        return list_idx

    def create_feature_list(self):
        fixed_feature_list = []
        for idx_x, key in enumerate(self.hyp_type.keys()):
            if type(self.hyp_type[key]) == list:
                for idx_y in range(len(self.hyp_type[key])):
                    fixed_feature_list += [{idx_x: idx_y}]
        return fixed_feature_list

    def numpy_to_kwargs(self, X_next):
        kwargs_new = {}
        for idx, key in enumerate(self.hyp_type.keys()):
            if type(self.hyp_type[key]) == list:
                kwargs_new[key] = self.hyp_type[key][round(X_next[idx].item())]           
            if self.hyp_type[key] == 'int':
                kwargs_new[key] = round(X_next[idx].item())
            elif self.hyp_type[key] == 'float':
                kwargs_new[key] = float(X_next[idx])
        return kwargs_new

    def list_to_numpy(self, list_config):
        new_list = []
        for this_dict in list_config:
            new_dict = {}
            for key in self.hyp_type.keys():
                if not (type(this_dict[key]) == str):
                    new_dict[key] = this_dict[key]
                else:
                    new_dict[key] = self.hyp_type[key].index(this_dict[key])
            new_list += [new_dict]
        return new_list

    def obtain_bounds(self):
        bounds = []
        for idx, key in enumerate(self.hyp_opt.keys()):
            bounds += [torch.tensor([self.hyp_opt[key][0], self.hyp_opt[key][-1]])]
        return torch.stack(bounds, 1).float()

    def update_candidates(self, candidates):
        for idx, key in enumerate(self.hyp_type.keys()):
            if type(self.hyp_type[key]) == list or self.hyp_type[key] == 'int':
                candidates[:, idx] = candidates[:, idx].round()
            elif self.hyp_type[key] == 'float':
                candidates[:, idx] = candidates[:, idx]
        return candidates

class BO_optimization(nn.Module):
    def __init__(self, fun_to_evaluate, config_space, order_list, seed = 0, n_candidates = None):
        self.fun_to_evaluate    = fun_to_evaluate
        self.bo_utils           = BO_utils(config_space, order_list)
        self.config_space       = config_space

        self.cat_dims           = self.bo_utils.obtain_idx()
        self.fixed_feature_list = self.bo_utils.create_feature_list()
        self.bool_tensor        = None
        self.seed               = seed
        if self.bo_utils.hyp_log is not None:
            self.bool_tensor    = torch.tensor(list(self.bo_utils.hyp_log.values()))
        self.n_candidates       = n_candidates
        if self.n_candidates is None:
            self.obtain_candidates = self.obtain_candidates_opt
        else:
            self.obtain_candidates = self.obtain_candidates_no_opt
    
    def prepare_x_input(self, init_x):
        bounds         = self.bo_utils.obtain_bounds()
        init_x         = init_x.double()
        num_feat       = init_x.shape[-1]

        bool_tensor = None
        if self.bo_utils.hyp_log is not None:
            bounds         = torch.where(self.bool_tensor.unsqueeze(0), bounds.log(), bounds)
            init_x         = torch.where(self.bool_tensor.unsqueeze(0), init_x.log(), init_x)

        # Normalize x
        x_min,  x_max             = bounds[0, :][None,:], bounds[1, :][None,:]
        x_max[..., self.cat_dims] = 1
        x_min[..., self.cat_dims] = 0
        init_x                    = (init_x - x_min) / (x_max - x_min)
        return init_x

    def prepare_data_input(self, init_x, init_y, best_init_y):
        bounds         = self.bo_utils.obtain_bounds()
        init_x, init_y = init_x.double(), init_y.double()
        num_feat       = init_x.shape[-1]

        bool_tensor = None
        if self.bo_utils.hyp_log is not None:
            bounds         = torch.where(self.bool_tensor.unsqueeze(0), bounds.log(), bounds)
            init_x         = torch.where(self.bool_tensor.unsqueeze(0), init_x.log(), init_x)

        # Normalize x
        x_min,  x_max             = bounds[0, :][None,:], bounds[1, :][None,:]
        x_max[..., self.cat_dims] = 1
        x_min[..., self.cat_dims] = 0
        init_x                    = (init_x - x_min) / (x_max - x_min)

        # Normalize y
        y_mean, y_std       = init_y.mean(0).unsqueeze(0), init_y.std(0).unsqueeze(0)
        if np.abs(y_std.sum()) != 0:
            init_y  = (init_y - y_mean) / y_std 
        bounds              = torch.cat([torch.zeros(1, num_feat), torch.ones(1, num_feat)])
        return init_x, init_y, bounds, x_max, x_min

    def get_model(self, init_x, init_y):
        likelihood = None
        if len(self.cat_dims) > 0:
            single_model = MixedSingleTaskGP(init_x, init_y, cat_dims = cat_dims)
        else:
            single_model = SingleTaskGP(init_x, init_y)
        return single_model, likelihood

    def maximize_likelihood(self, single_model, likelihood):
        mll          = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
        fit_gpytorch_model(mll)
        return single_model

    def obtain_acq(self, single_model, best_init_y):
        return qExpectedImprovement(model = single_model, best_f = best_init_y)

    def obtain_candidates_no_opt(self, acq_function, bounds = None):
        sampled_cfg   = sample_configurations(self.config_space, self.seed, self.n_candidates)        
        init_x        = torch.tensor(pd.DataFrame(self.bo_utils.list_to_numpy(sampled_cfg)).to_numpy())
        init_x        = self.prepare_x_input(init_x)
        init_x        = init_x.reshape(init_x.shape[0], 1, -1)
        acq           = acq_function(torch.tensor(init_x))
        return init_x[acq.argmax()]

    def obtain_candidates_opt(self, acq_function, bounds):
        if len(self.cat_dims) > 0:
            candidates, _ = optimize_acqf_mixed(acq_function = acq_function,
                                                bounds       = bounds,
                                                q            = 1,
                                                num_restarts = 20, #num_restarts = 200,
                                                raw_samples  = 100, #raw_samples = 1024
                                                fixed_features_list = self.fixed_feature_list,
                                                options = {"batch_limit": 5, "maxiter": 200})
        else:
            candidates, _ = optimize_acqf(acq_function = acq_function,
                                            bounds = bounds,
                                            q = 1,
                                            num_restarts = 20,
                                            raw_samples = 100,
                                            options = {"batch_limit": 5, "maxiter": 200})
        return candidates

    def prepare_data_output(self, candidates, x_max, x_min):
        final_result = candidates * (x_max - x_min) + x_min
        if self.bo_utils.hyp_log is not None:
            final_result = torch.where(self.bool_tensor.unsqueeze(0), final_result.exp(), final_result)
        return final_result

    def get_next_points(self, init_x, init_y, best_init_y, n_points = 1):
        init_x, init_y, bounds, x_max, x_min = self.prepare_data_input(init_x, init_y, best_init_y)

        single_model, likelihood = self.get_model(init_x, init_y)
        single_model             = self.maximize_likelihood(single_model, likelihood)

        acq_function             = self.obtain_acq(single_model, best_init_y)
        candidates               = self.obtain_candidates(acq_function, bounds)
        candidates               = self.prepare_data_output(candidates, x_max, x_min)
        return candidates

    def loop_model(self, train_x):
        all_y            = []
        list_add_metrics = []
        for idx in range(len(train_x)):
            kwargs                = self.bo_utils.numpy_to_kwargs(train_x[idx, :])
            result                = self.fun_to_evaluate(kwargs)
            all_y                += [result]
        return np.array(all_y) * -1

    def generate_initial_data(self, n_init, train_x):
        exact_obj               = self.loop_model(train_x)
        best_observed_value     = exact_obj.max().item()
        return torch.tensor(train_x).float(), torch.tensor(exact_obj).reshape(-1, 1).float(), best_observed_value

    def optimize(self, n_runs, n_init, config_init):
        self.fun_to_evaluate.reset_results()
        init_x, init_y, best_init_y = self.generate_initial_data(n_init = n_init,
                            train_x = pd.DataFrame(self.bo_utils.list_to_numpy(config_init[:n_init])).to_numpy())

        x_before = torch.tensor([])
        for i in range(n_runs):
            print(f"Nr. of optimization run: {i}")
            new_candidates              = self.get_next_points(init_x, init_y, best_init_y, 1)
            x_before                    = torch.cat([x_before, new_candidates])
            new_candidates              = self.bo_utils.update_candidates(new_candidates)
            new_results                 = self.loop_model(new_candidates)
            init_x                      = torch.cat([init_x, new_candidates])
            init_y                      = torch.cat([init_y, torch.tensor(new_results).reshape(1,1)]).float()
            best_init_y                 = init_y.max().item()
        return init_y, pd.DataFrame(self.fun_to_evaluate.all_results)

class BO_optimization_random(BO_optimization):
    def __init__(self, fun_to_evaluate, config_space, order_list, seed = 0, n_candidates = None):
        super().__init__(fun_to_evaluate, config_space, order_list, seed)

    def optimize(self, n_runs, n_init, config_init):
        self.fun_to_evaluate.reset_results()
        init_x, init_y, best_init_y = self.generate_initial_data( n_init = n_init,
                            train_x = pd.DataFrame(self.bo_utils.list_to_numpy(config_init[:n_init])).to_numpy())

        x_before = torch.tensor([])
        for i in range(n_runs):
            sampled_config = sample_configurations(self.config_space, i + self.seed * 100, 1)
            new_candidates = torch.tensor(pd.DataFrame(self.bo_utils.list_to_numpy(sampled_config)).to_numpy()).float()
            x_before                    = torch.cat([x_before, new_candidates])
            new_candidates              = self.bo_utils.update_candidates(new_candidates)
            new_results                 = self.loop_model(new_candidates)
            init_x                      = torch.cat([init_x, new_candidates])
            init_y                      = torch.cat([init_y, torch.tensor(new_results).reshape(1,1)]).float()
            best_init_y                 = init_y.max().item()
        return init_y, pd.DataFrame(self.fun_to_evaluate.all_results)
