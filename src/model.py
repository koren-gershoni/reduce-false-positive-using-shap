import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from config import config

device = 'cuda' if torch.cuda.is_available() else "cpu"

class AEEncoder(nn.Module):
    def __init__(self, config, input_dim, iter=0, new_model=True):
        super(AEEncoder, self).__init__()
        self.config = config
        if config.random_architecture_enabled and not new_model:
            self.layers = []
            for output, input in config.encoder_dims[:-1]:
                cur_layer = nn.Linear(input, output).to(device)
                self.layers.append(cur_layer)
            last_output, last_input = config.encoder_dims[-1]
            self.fc_mean = nn.Linear(last_input, last_output).to(device)
            self.fc_var = nn.Linear(last_input, last_output).to(device)

        elif config.random_architecture_enabled and new_model:
            self.layers = []
            last_linear_dim = input_dim
            config.decoder_dims = []
            self.num_layers = np.random.randint(config.min_num_layers, config.max_num_layers)
            for i in range(self.num_layers):
                if i == 0:
                    cur_linear_dim = np.random.randint(input_dim // 2, last_linear_dim)
                else:
                    cur_linear_dim = np.random.randint(self.num_layers - i + 1, last_linear_dim)
                cur_layer = nn.Linear(last_linear_dim, cur_linear_dim).to(device)
                config.decoder_dims.insert(0, (cur_linear_dim, last_linear_dim))
                last_linear_dim = cur_linear_dim
                self.layers.append(cur_layer)
            rand_latent = np.random.randint(1, last_linear_dim)
            config.decoder_dims.insert(0, (rand_latent, last_linear_dim))
            self.last_nn = nn.Linear(last_linear_dim, rand_latent).to(device)
            with open(f'{config.saving_dir}/{iter}/decoder_dims.txt', "w") as f:
                f.write(str(config.decoder_dims))
        else:
            self.fc_1 = nn.Linear(input_dim, input_dim // 2)
            self.fc_2 = nn.Linear(input_dim // 2, input_dim // 4)
            self.fc_out = nn.Linear(input_dim // 4, 2)

        self.activation = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x):
        if config.random_architecture_enabled:
            for i in range(len(self.layers)):
                x = self.activation(self.layers[i](x))
                if config.dropout_enabled:
                    if i != (len(self.layers) - 1):
                        x = nn.Dropout(self.config.dropout_prob)(x)
            x = self.last_nn(x)
        else:
            x = self.activation(self.fc_1(x))
            x = self.activation(self.fc_2(x))
            x = self.fc_out(x)
        return x

class AEDecoder(nn.Module):
    def __init__(self, config, output_dim):
        super(AEDecoder, self).__init__()
        self.config = config
        self.layers = []
        self.norm_layers = []
        if config.random_architecture_enabled:
            dims = config.decoder_dims
            # print(dims)
            self.fc_out = nn.Linear(*dims[-1]).to(device)
            for input, output in dims[:-1]:
                cur_layer = nn.Linear(input, output).to(device)
                self.layers.append(cur_layer)
                self.norm_layers.append(nn.LayerNorm(output).to(device))
        else:
            self.fc_h1 = nn.Linear(2, output_dim // 4)
            self.fc_h2 = nn.Linear(output_dim // 4, output_dim // 2)
            self.fc_out = nn.Linear(output_dim // 2, output_dim)

        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        if config.random_architecture_enabled:
            for i in range(len(self.layers)):
                x = self.activation(self.layers[i](x))
                if config.dropout_enabled:
                    if i != range(len(self.layers)):
                        x = nn.Dropout(config.dropout_prob)(x)
        else:
            x = self.activation(self.fc_h1(x))
            x = self.activation(self.fc_h2(x))

        x_hat = torch.sigmoid(self.fc_out(x))
        return x_hat

class AEModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(AEModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_hat = self.decoder(x_encoded)

        return x_hat

    def predict(self, x):
        self.eval()
        if isinstance(x, pd.DataFrame):
            x = x.values
        x = torch.tensor(x).float().to(device)
        with torch.no_grad():
            x_hat = self.forward(x)
        return x_hat.detach().cpu().numpy()


def load_models():
    models = []
    for i in range(config.n_estimators + 1):
        if torch.cuda.is_available():
            model = torch.load(f"{config.saving_dir}/{i}/model.pt")
            model.eval()
            models.append(model)
        else:
            model = torch.load(f"{config.saving_dir}/{i}/model.pt",  map_location='cpu')
            model.eval()
            models.append(model)
    return models