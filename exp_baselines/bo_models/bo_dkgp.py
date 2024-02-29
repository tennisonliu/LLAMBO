import torch
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal

from bo_models.bo_gp_class import BO_optimization
import gpytorch

from botorch.models.gpytorch import GPyTorchModel
from gpytorch.models import ExactGP

NUM_DIM_EXTRACTOR = 4
class SmallFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim):
        super(SmallFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 32))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(32, 32))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(32, 32))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(32, NUM_DIM_EXTRACTOR))

class DeepKernelLearning(ExactGP, GPyTorchModel):
        _num_outputs = 1  # to inform GPyTorchModel API
        num_outputs = 1  # to inform GPyTorchModel API

        def __init__(self, train_x, train_y, likelihood):
            super(DeepKernelLearning, self).__init__(train_x, train_y.squeeze(-1), likelihood)
            #super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
            self.mean_module = ConstantMean()
            self.covar_module = ScaleKernel(
                base_kernel=RBFKernel(ard_num_dims=NUM_DIM_EXTRACTOR),
            )

            self.feature_extractor = SmallFeatureExtractor(train_x.shape[-1])
            # This module will scale the NN features so that they're nice values
            
        def forward(self, x):
            # We're first putting our data through a deep net (feature extractor)
            projected_x = self.feature_extractor(x)
            mean_x      = self.mean_module(projected_x)
            covar_x     = self.covar_module(projected_x)
            return MultivariateNormal(mean_x, covar_x)

class BO_optimization_DKL(BO_optimization):
    def __init__(self, fun_to_evaluate, config_space, order_list, seed = 0):
        super().__init__(fun_to_evaluate, config_space, order_list, seed)

    def get_model(self, init_x, init_y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = DeepKernelLearning(init_x, init_y, likelihood)
        model = model.float()
        return model, likelihood

    def maximize_likelihood(self, single_model, likelihood):
        def train_model(train_x, train_y, model, likelihood):
            training_iterations = 100
            # Find optimal model hyperparameters
            model.train()
            likelihood.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam([
                {'params': model.feature_extractor.parameters()},
                {'params': model.covar_module.parameters()},
                {'params': model.mean_module.parameters()},
                {'params': model.likelihood.parameters()},
            ], lr = 0.01)

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            def train():
                #iterator = tqdm.notebook.tqdm(range(training_iterations))
                iterator = range(training_iterations)
                for i in iterator:
                    # Zero backprop gradients
                    optimizer.zero_grad()
                    # Get output from model
                    output = model(train_x)
                    # Calc loss and backprop derivatives
                    loss = -mll(output, train_y).mean()
                    loss.backward()
                    #iterator.set_postfix(loss=loss.item())
                    optimizer.step()
            train()
            model.eval()
            likelihood.eval()
            return model
        train_x = single_model.train_inputs[0]
        train_y = single_model.train_targets
        return train_model(train_x, train_y, single_model, likelihood)