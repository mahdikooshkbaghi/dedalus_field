from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")
# standard import
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch
from torch import ge, nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar


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
    max_epoch = 10
    learning_rate = 1e-4
    train_split = 0.8
    device = "cpu"
    batch_size = 1024


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

X = torch.tensor(np.c_[xv.ravel(), yv.ravel(), tv.ravel()].astype(np.float32))
u = torch.tensor(u.ravel().reshape(-1, 1).astype(np.float32))
data = TensorDataset(X, u)

# Split train and validation
train_set_size = int(len(u) * args.train_split)
valid_set_size = len(u) - train_set_size


class PINN(pl.LightningModule):
    def __init__(self, data, layers, lr=1e-3):
        """Initialize Network layers and Trainable parameters"""
        super().__init__()
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
            ("layer_%d" % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.dnn = nn.Sequential(layerDict)

        # parameters of the RHS
        # lambda1
        self.lambda_1 = torch.tensor([0.0])
        self.lambda_1 = torch.nn.Parameter(self.lambda_1, requires_grad=True)
        self.register_parameter("lambda_1", self.lambda_1)
        # lambda2
        self.lambda_2 = torch.tensor([0.0])
        self.lambda_2 = torch.nn.Parameter(self.lambda_2, requires_grad=True)
        self.register_parameter("lambda_2", self.lambda_2)
        # lambda3
        self.lambda_3 = torch.tensor([0.0])
        self.lambda_3 = torch.nn.Parameter(self.lambda_3, requires_grad=True)
        self.register_parameter("lambda_3", self.lambda_3)

        # learning rate
        self.lr = lr

        # Iteration for tunning step
        self.iter = 0

        # Data
        self.data = data

    def forward(self, x, y, t):
        """Forward pass (x,y,t) -> u"""
        out = self.dnn(torch.cat([x, y, t], dim=1))
        return out

    def net_derivatives(self, func, var):
        """Calculate Network derivatives func with respect to var \par func/\par var"""
        dfunc_dvar = torch.autograd.grad(
            func,
            var,
            grad_outputs=torch.ones_like(func),
            retain_graph=True,
            create_graph=True,
        )[0]
        return dfunc_dvar

    def net_f(self, x, y, t):
        """Find the RHS of the PDE"""
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        lambda_3 = self.lambda_3

        u = self.forward(x, y, t)
        u_t = self.net_derivatives(u, t)
        u_x = self.net_derivatives(u, x)
        u_y = self.net_derivatives(u, y)
        u_xx = self.net_derivatives(u_x, x)
        u_yy = self.net_derivatives(u_y, y)

        f = u_t + lambda_1 * (u_xx + u_yy) + lambda_2 * u + lambda_3 * (u**3)
        return f

    def training_step(self, train_batch, batch_idx):
        """Training step for each batch passed to lightning"""
        X = train_batch[0]
        u = train_batch[1]
        self.x = torch.tensor(X[:, 0].reshape(-1, 1), requires_grad=True)
        self.y = torch.tensor(X[:, 1].reshape(-1, 1), requires_grad=True)
        self.t = torch.tensor(X[:, 2].reshape(-1, 1), requires_grad=True)
        self.u = torch.tensor(u, requires_grad=True)
        u_pred = self.forward(self.x, self.y, self.t)
        f_pred = self.net_f(self.x, self.y, self.t)
        # loss function |u-u_pred|^2+|f-f_pred|^2
        loss = torch.mean((self.u - u_pred) ** 2) + torch.mean(f_pred**2)
        # Add the lambda_1 to the progress bar in training
        self.log("lam1", self.lambda_1, prog_bar=True, on_step=False, on_epoch=True)
        self.log("lam2", self.lambda_2, prog_bar=True, on_step=False, on_epoch=True)
        self.log("lam3", self.lambda_3, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_loss", loss, prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # Configure the first optimizer used in training
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]

    def validation_step(self, batch, batch_idx):
        X = batch[0]
        u = batch[1]
        x = X[:, 0].reshape(-1, 1)
        y = X[:, 1].reshape(-1, 1)
        t = X[:, 2].reshape(-1, 1)
        u_pred = self.forward(x, y, t)
        # loss function |u-u_pred|^2+|f-f_pred|^2
        loss = torch.mean((u - u_pred) ** 2)
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.data_train, self.data_val = random_split(
                data, [train_set_size, valid_set_size]
            )

        if stage == "test" or stage is None:
            self.data_test = self.data_val

        if stage == "predict":
            self.data_test = self.data_val

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=args.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=args.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=args.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.data_test, batch_size=len(self.data_test))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X = batch[0]
        u = batch[1]
        x = X[:, 0].reshape(-1, 1)
        y = X[:, 1].reshape(-1, 1)
        t = X[:, 2].reshape(-1, 1)
        u_pred = self.forward(x, y, t)
        return u, u_pred

    def on_train_end(self) -> None:
        # Fine tune PINN with LBFGS
        print("Fine Tune Model Using LBFGS")
        self.pinn_optimizer().step(self.pinn_loss)
        return super().on_train_end()

    def pinn_loss(self):
        # Loss function to tune coefficients
        u_pred = self.forward(self.x, self.y, self.t)
        f_pred = self.net_f(self.x, self.y, self.t)
        loss = torch.mean((self.u - u_pred) ** 2) + torch.mean(f_pred**2)
        self.pinn_optimizer().zero_grad()
        loss.backward()
        self.iter += 1
        if self.iter % 1 == 0:
            print(
                f"Iter: {self.iter}, lam1: {self.lambda_1.item(): .8f}, lam2: {self.lambda_2.item(): .8f}, lam3: {self.lambda_3.item(): .8f}, loss: {loss: .3e}"
            )
        return loss

    def pinn_optimizer(self):
        # optimizers: using the same settings
        return torch.optim.LBFGS(
            self.parameters(),
            lr=1.0,
            max_iter=5000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-8,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )


# model
model = PINN(data=data, layers=args.layers, lr=args.learning_rate)
trainer = pl.Trainer(accelerator=args.device, max_epochs=args.max_epoch)
# training
trainer.fit(model)

# predict
predict_step = trainer.predict(model)
u = predict_step[0][0]
u_pred = predict_step[0][1]

# fig, ax = plt.subplots(1, 1)
# xlim = [-1, 1]
# ax.scatter(u, u_pred, alpha=0.1, label='test data', rasterized=True)
# ax.plot(xlim, xlim, c='r')
# ax.set_xlabel(r'$u_{test}$')
# ax.set_ylabel(r'$u_{pred}$')
# Rsq = np.corrcoef(u_pred.ravel(), u[:, -1].ravel())[0, 1]**2
# ax.set_title(f'$R^2$={Rsq:.3}')
# ax.legend()
# plt.tight_layout()
# plt.show()
