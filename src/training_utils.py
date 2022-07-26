import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from tqdm import tqdm

from data_utils import handle_categorical_columns, handle_df_examples, split_x_y, split_train_val, \
    split_train_test_credit, split_train_test_kdd
from model import AEEncoder, AEDecoder, AEModel

device = 'cuda' if torch.cuda.is_available() else "cpu"

from torch.optim import Adam
BCE_loss = nn.BCELoss()
MSE_loss = nn.MSELoss()
def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


def train_model_AE(config, train_loader, val_loader, input_dim, i):
    cur_patience = 0
    encoder = AEEncoder(input_dim, i).to(device)
    decoder = AEDecoder(input_dim).to(device)
    model = AEModel(encoder, decoder).to(device)
    optimizer = Adam(model.parameters(), lr=config.lr)
    params = model.parameters()
    best_loss = 1e10
    print(f"Start training VAE...\n")
    model.train()
    for epoch in range(config.epochs):
        overall_loss = 0
        for batch_idx, x in tqdm(enumerate(train_loader)):
            x = x.float().to(device)

            optimizer.zero_grad()

            x_hat = model(x)
            x_hat.to(device)
            # loss = loss_function(x, x_hat, mean, log_var)
            loss = MSE_loss(x, x_hat)
            overall_loss += loss.item()

            loss.backward()
            # nn.utils.clip_grad_norm_(params, 0.1)
            optimizer.step()

        val_loss = 0
        with torch.no_grad():
            for batch_idx, x in tqdm(enumerate(val_loader)):
                x = x.float().to(device)

            x_hat = model(x)
            x_hat.to(device)
            # loss = loss_function(x, x_hat, mean, log_var)
            loss = MSE_loss(x, x_hat)

            val_loss += loss.item()

        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*config.batch_size))
        if val_loss < best_loss:
            print(f"prev val loss:{best_loss}\nnew val loss: {val_loss}")
            best_loss = val_loss
            cur_patience = 0
            torch.save(encoder, f'{config.saving_dir}/{i}/encoder.pt')
            torch.save(decoder, f'{config.saving_dir}/{i}/decoder.pt')
            torch.save(model, f'{config.saving_dir}/{i}/model.pt')
            # torch.save(encoder.state_dict(), f'{config.saving_dir}/{i}/encoder.pt')
            # torch.save(model.state_dict(), f'{config.saving_dir}/{i}/model.pt')
            # torch.save(encoder.state_dict(), f'{config.saving_dir}/{i}/encoder.pt')
            # torch.save(decoder.state_dict(), f'{config.saving_dir}/{i}/decoder.pt')
        else:
            cur_patience += 1
            if cur_patience == config.patience:
                break


def run_training_flow(config, df, dataset, seed, label_map=None, AE=False):
    # save_df = not (config.n_features == len(X_df.columns)) or \
    #           config.sample_with_replacement
    save_df = True
    save_columns = config.random_num_of_features_enabled
    Path(f'{config.saving_dir}').mkdir(parents=True, exist_ok=True)
    json.dump(vars(config), open(f'{config.saving_dir}/config.json', 'w'), indent=2)
    for i in range(config.starting_model, config.n_estimators + 1):
        print(f"****************model {i}****************")
        cur_df = handle_categorical_columns(df)
        if i != config.n_estimators:
            cur_df = handle_df_examples(config, cur_df)
        Path(f'{config.saving_dir}/{i}/').mkdir(parents=True, exist_ok=True)
        if save_df:
            cur_df.to_csv(f'{config.saving_dir}/{i}/df.csv')
        if save_columns:
            json.dump(list(cur_df.columns), open(f'{config.saving_dir}/{i}/columns.json', 'w'))
        cur_X_df, cur_y_df = split_x_y(cur_df)
        if dataset == "kdd":
            X_train, y_train, X_test, y_test = split_train_test_kdd(cur_X_df, cur_y_df, label_map)
        else:
            X_train, y_train, X_test, y_test = split_train_test_credit(cur_X_df, cur_y_df)
            print(len(X_train), len(y_train), len(X_test), len(y_test))
        X_train, X_val, y_train, y_val = split_train_val(X_train, y_train, seed)
        # if dataset == "kdd" or dataset == "creditcard":
        scaler = MinMaxScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_val_scaled[X_val_scaled > 1] = 1
        # else:
        #     X_train_scaled = X_train.values
        #     X_val_scaled = X_val.values
        train_loader = DataLoader(X_train_scaled, batch_size=config.batch_size)
        val_loader = DataLoader(X_val_scaled, batch_size=config.batch_size // 2)
        if AE:
            train_model_AE(config, train_loader, val_loader, X_train_scaled.shape[1], i)
        # else:
        #     train_model(train_loader, val_loader, X_train_scaled.shape[1], i)