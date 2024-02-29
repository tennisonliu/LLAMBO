from bo_models.bo_gp_class import BO_optimization
import sys
sys.path.append('exp_baselines/bo_models')
from pybnn import DNGO
from botorch.acquisition.analytic import ExpectedImprovement
import torch
import numpy as np

# Defining a class for BO optimization using the DNGO model
class BO_optimization_DNGO(BO_optimization):
    def __init__(self, fun_to_evaluate, config_space, order_list, seed = 0, n_candidates = 20):
        super().__init__(fun_to_evaluate, config_space, order_list, seed, n_candidates = n_candidates)

    def get_model(self, init_x, init_y):
        return DNGO(X_train = init_x, y_train = init_y, do_mcmc=False), None
        
    def obtain_acq(self, single_model, best_init_y):
        return ExpectedImprovement(model = single_model, best_f = best_init_y)

    def maximize_likelihood(self, single_model, likelihood):
        print("likelihood")
        single_model.train(do_optimize=True)
        return single_model
    
    def prepare_data_input(self, init_x, init_y, best_init_y):
        bounds         = self.bo_utils.obtain_bounds()
        init_x, init_y = init_x.float(), init_y.float()
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
