import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")
# standard import
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import grad

from collections import OrderedDict
from tqdm import tqdm


@dataclass
class args:
    data_path = "../2D_field_eqs/allen_cahn/u_allen_cahn_128_Ny_128_timestep_1e-05.npy"
    np_seed = 1234
    torch_seed = 0
    n_t = 200  # number of timesteps
    saved_every = 10
    delta_t = 1e-5
    L_x = 1.0
    L_y = 1.0
    n_x = 128
    n_y = 128

    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    n_epoch = 1000
    learning_rate = 1e-2
    train_split = 0.8  # This is not implemented yet
    batch_size = 1024  # This is not implemented yet
    device = "cpu"


device = torch.device(args.device)
# Fix the random seeds
np.random.seed(args.np_seed)
torch.manual_seed(args.torch_seed)

# Load data
u = np.load(args.data_path)
x = np.linspace(0, args.L_x, args.n_x, dtype=np.float32)
y = np.linspace(0, args.L_y, args.n_y, dtype=np.float32)
t_saved = args.delta_t * args.saved_every
t = np.linspace(t_saved, args.n_t * t_saved, args.n_t, dtype=np.float32)
tv, xv, yv = np.meshgrid(t, x, y, indexing="ij")

X = torch.tensor(np.c_[tv.ravel(), xv.ravel(), yv.ravel()], requires_grad=True).to(
    device
)
u = torch.tensor(u.ravel().reshape(-1, 1), requires_grad=True).to(device)

# the deep neural network


class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ("layer_%d" % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(("activation_%d" % i, self.activation()))

        layer_list.append(
            ("layer_%d" % (self.depth - 1),
             torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


# the physics-guided neural network
class PhysicsInformedNN:
    def __init__(self, X, u, layers):
        # data
        self.t = X[:, 0].view(-1, 1)
        self.x = X[:, 1].view(-1, 1)
        self.y = X[:, 2].view(-1, 1)
        self.u = u.view(-1, 1)

        # parameter to infer
        self.lambda_1 = torch.tensor([2.0], requires_grad=True).to(device)
        self.lambda_1 = torch.nn.Parameter(self.lambda_1)

        self.lambda_2 = torch.tensor([2.0], requires_grad=True).to(device)
        self.lambda_2 = torch.nn.Parameter(self.lambda_2)

        self.lambda_3 = torch.tensor([2.0], requires_grad=True).to(device)
        self.lambda_3 = torch.nn.Parameter(self.lambda_3)

        # deep neural networks
        self.dnn = DNN(layers).to(device)
        self.dnn.register_parameter("lambda_1", self.lambda_1)
        self.dnn.register_parameter("lambda_2", self.lambda_2)
        self.dnn.register_parameter("lambda_3", self.lambda_3)

        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-8,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )

        self.optimizer_Adam = torch.optim.Adam(
            self.dnn.parameters(), lr=args.learning_rate)
        self.iter = 0

    def net_u(self, txy):
        t = txy[:, 0].view(-1, 1)
        x = txy[:, 1].view(-1, 1)
        y = txy[:, 2].view(-1, 1)
        u = self.dnn(torch.cat([t, x, y], dim=1))
        return u

    def net_f(self, txy):
        """The pytorch autograd version of calculating residual"""
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        lambda_3 = self.lambda_3
        u = self.net_u(txy)
        u_t = grad(u, txy, create_graph=True, retain_graph=True,
                   grad_outputs=torch.ones_like(u))[0][:, 0]
        u_x = grad(u, txy, create_graph=True, retain_graph=True,
                   grad_outputs=torch.ones_like(u))[0][:, 1]
        u_y = grad(u, txy, create_graph=True, retain_graph=True,
                   grad_outputs=torch.ones_like(u))[0][:, 2]
        u_xx = grad(u_x, txy, retain_graph=True,
                    grad_outputs=torch.ones_like(u_x))[0][:, 1]
        u_yy = grad(u_y, txy, retain_graph=True,
                    grad_outputs=torch.ones_like(u_y))[0][:, 2]

        f = u_t + lambda_1 * (u_xx + u_yy) + lambda_2 * u + lambda_3 * (u**3)
        return f

    def loss_func(self):
        u_pred = self.net_u(torch.hstack([self.t, self.x, self.y]))
        f_pred = self.net_f(torch.hstack([self.t, self.x, self.y]))
        loss = torch.mean((self.u - u_pred) ** 2) + torch.mean(f_pred**2)
        self.optimizer.zero_grad()
        loss.backward()
        self.iter += 1
        if self.iter % 1 == 0:
            print(
                "Loss: %e, lambda1: %.5f, lambda2: %.5f , lambda3: %.5f"
                % (
                    loss.item(),
                    self.lambda_1.item(),
                    self.lambda_2.item(),
                    self.lambda_3.item(),
                )
            )

        return loss

    def train(self, nIter):
        self.dnn.train()
        pbar = tqdm(range(1, nIter), desc="Epoch")
        for epoch in pbar:
            u_pred = self.net_u(torch.hstack([self.t, self.x, self.y]))
            f_pred = self.net_f(torch.hstack([self.t, self.x, self.y]))
            loss = torch.mean((self.u - u_pred) ** 2) + torch.mean(f_pred**2)

            # Backward and optimize
            self.optimizer_Adam.zero_grad()
            loss.backward()
            self.optimizer_Adam.step()
            pbar.set_postfix(
                {
                    "train_loss": "{0:2.3g}".format(loss),
                    "lambda1": "{0:2.3g}".format(self.lambda_1.item()),
                    "lambda2": "{0:2.3g}".format(self.lambda_2.item()),
                    "lambda3": "{0:2.3g}".format(self.lambda_3.item()),
                }
            )
        # Backward and optimize
        print("\n=> start the LBFGS optimizition")
        self.optimizer.step(self.loss_func)

    # def predict(self, X):
    #     x = torch.tensor(X[:, 0].reshape(-1, 1), requires_grad=True).float().to(device)
    #     y = torch.tensor(X[:, 1].reshape(-1, 1), requires_grad=True).float().to(device)
    #     t = torch.tensor(X[:, 2].reshape(-1, 1), requires_grad=True).float().to(device)

    #     self.dnn.eval()
    #     u = self.net_u(x, y, t)
    #     f = self.net_f(x, y, t)
    #     u = u.detach().cpu().numpy()
    #     f = f.detach().cpu().numpy()
    #     return u, f


# This need to be changed for the PYTORCH DATALOADER WITH BATCH SIZE AND SPLIT
n_data = 5_000
# Trained on the last data since the effect of initial condition is resolved.
model = PhysicsInformedNN(X[-n_data:, :], u[-n_data:], args.layers)
model.train(args.n_epoch)
