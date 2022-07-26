import os
import pickle
from collections import defaultdict
from functools import partial
from config import config
import numpy as np
import pandas as pd
from time import time
import torch

def get_example_errors(explainer, X_train, X, model, example_num):
    return explainer.explain_unsupervised_data(pd.DataFrame(X_train), pd.DataFrame(X[example_num].reshape(1, -1)), autoencoder=model, return_shap_values=True)

def dump_rec_and_shap(config, model_rec_list, model_shap_list, split):
    try:
        pickle.dump(model_rec_list, open(f"{config.saving_dir}/{split}_rec_list.pkl", 'wb'))
        pickle.dump(model_shap_list, open(f"{config.saving_dir}/{split}_shap_list.pkl", 'wb'))
    except:
        for i in range(config.n_estimators):
            pickle.dump(model_rec_list[i], open(f"{config.saving_dir}/{split}_rec_list{i}.pkl", 'wb'))
            pickle.dump(model_shap_list[i], open(f"{config.saving_dir}/{split}_shap_list{i}.pkl", 'wb'))

def get_rec_and_shap(config, explainer, X_train, models, X, split):
    model_rec_list = []
    model_shap_list = []
    start = time()
    dump = False
    for model in models[:-1]:
        func = partial(get_example_errors, explainer, X_train, X, model)
        rec_results = []
        shap_results = []
        shap_path = ""
        rec_path = ""
        rec_path_array = [filename for filename in os.listdir(f'{config.saving_dir}') if filename.startswith(f"rec_{split}_{model}")]
        starting_index = 0
        if rec_path_array:
            starting_index = int(rec_path_array[0].split(".")[0].split("_")[-1]) + 1
            rec_path = os.path.join(config.saving_dir, rec_path_array[0])
            rec_results = pickle.load(open(rec_path, 'rb'))
            shap_path = [filename for filename in os.listdir(f'{config.saving_dir}') if filename.startswith(f"shap_{split}_{model}")][0]
            shap_path = os.path.join(config.saving_dir, shap_path)
            shap_results = pickle.load(open(shap_path, 'rb'))
        for i in range(starting_index, len(X)):
            dump = True
            rec, shap = func(i)
            rec_results.append(rec)
            shap_results.append(shap)
            if len(rec_results) % 100 == 0 or len(rec_results) == len(X):
                pickle.dump(rec_results, open(f"{config.saving_dir}/rec_{split}_{model}_{i}.pkl", 'wb'))
                pickle.dump(shap_results, open(f"{config.saving_dir}/shap_{split}_{model}_{i}.pkl", 'wb'))
                if rec_path:
                    os.remove(rec_path)
                    os.remove(shap_path)
                rec_path = f"{config.saving_dir}/rec_{split}_{model}_{i}.pkl"
                shap_path = f"{config.saving_dir}/shap_{split}_{model}_{i}.pkl"
        model_rec_list.append(rec_results)
        model_shap_list.append(shap_results)

    end = time()
    # print(end - start)
    if dump:
        dump_rec_and_shap(config, model_rec_list, model_shap_list, split)
    return model_rec_list, model_shap_list


def load_rec_and_shap(config, split, partitions=False):
    if partitions:
        rec_list = []
        for i in range(config.n_estimators):
            rec_list_temp = pickle.load(open(f"{config.saving_dir}/{split}_rec{i}.pkl", 'rb'))
            rec_list.append(rec_list_temp[0])
        return rec_list
    else:
        return pickle.load(open(f"{config.saving_dir}/{split}_rec_list.pkl", 'rb')), pickle.load(open(f"{config.saving_dir}/{split}_shap_list.pkl", 'rb'))


# def calculate_mse_train(model, example_idx):
#     with torch.no_grad():
#         y_pred = model(torch.Tensor(X_train[example_idx]).to(device)).detach().cpu().numpy()
#     return ((y_pred - X_train[example_idx]) ** 2).mean()
# vec_calculate_mse_train = np.vectorize(calculate_mse_train)
# def calculate_mse_val(model, example_idx):
#     with torch.no_grad():
#         y_pred = model(torch.Tensor(X_val[example_idx]).to(device)).detach().cpu().numpy()
#     return ((y_pred - X_val[example_idx]) ** 2).mean()
# vec_calculate_mse_val = np.vectorize(calculate_mse_val)
# def calculate_mse_test(model, example_idx):
#     with torch.no_grad():
#         y_pred = model(torch.Tensor(X_test[example_idx]).to(device)).detach().cpu().numpy()
#     return ((y_pred - X_test[example_idx]) ** 2).mean()
# vec_calculate_mse_test = np.vectorize(calculate_mse_test)

def calculate_counter_list_helper(rec_list, error_threshold, ex):
    num_features = config.num_columns
    counter_array = np.zeros(num_features)
    feature_model_agree = [[] for _ in range(num_features)]
        # np.zeros((num_features, len(rec_list)))
    for model_num in range(len(rec_list)):
        error_dict = rec_list[model_num][ex]
        for feature_number, error in error_dict.items():
            if error > error_threshold:
                feature_model_agree[feature_number].append(model_num)
                counter_array[feature_number] += 1
    return counter_array, feature_model_agree

def calculate_counter_list(rec_list, error_threshold):
    num_examples = len(rec_list[0])
    #foreach example, each model increase by 1 the i-feature if the error is above some threshold
    func = partial(calculate_counter_list_helper, rec_list, error_threshold)
    counter_array = []
    feature_model_agree = []
    # todo - change
    for i in range(num_examples):
        temp_counter_array, temp_feature_model_agree = func(i)
        counter_array.append(temp_counter_array)
        feature_model_agree.append(temp_feature_model_agree)

    return np.array(counter_array), feature_model_agree


def get_anomaly_list_shap(X_anomalies, counter_list, shap_values, feature_model_agree, num_models_needed, shap_threshold, rec_err, include_anomaly_feature_shap=False):
    is_anomaly_list_feature_in_rec = np.zeros(len(counter_list))
    is_anomaly_list_feature = np.zeros(len(counter_list))
    # go over all examples
    for i, ls in enumerate(counter_list):
        # contain the shap values agreement
        shap_agreement_in_rec = np.zeros(X_anomalies.shape[1])
        shap_agreement_not_in_rec = np.zeros(X_anomalies.shape[1])
        # go over all features in example i
        for j, models_agree in enumerate(ls):
            # if not enough models agree then we don't need to continue
            if models_agree >= num_models_needed:
                shap_above_threshold_counter_in_rec = np.zeros(X_anomalies.shape[1])
                shap_above_threshold_counter = np.zeros(X_anomalies.shape[1])
                # go over all models that agree on feature j
                for model_index in feature_model_agree[i][j]:
                    model_rec_err = rec_err[model_index][i]
                    model_shap = shap_values[model_index][i].toarray()[j]
                    model_shap_abs = np.abs(model_shap)
                    sorted_indices = np.argsort(model_shap_abs)
                    for feature in sorted_indices[-5:]:
                        # whether or not include the feature j shap (feature that all models agree on as anomaly)
                        if not include_anomaly_feature_shap:
                            if feature != j and model_shap_abs[feature] > shap_threshold:
                                shap_above_threshold_counter[feature] += 1
                            # feature must be in the features with the high reconstruction error
                            if feature != j and model_shap_abs[feature] > shap_threshold and feature in model_rec_err:
                                shap_above_threshold_counter_in_rec[feature] += 1
                        else:
                            if model_shap_abs[feature] > shap_threshold:
                                shap_above_threshold_counter[feature] += 1
                            # feature must be in the features with the high reconstruction error
                            if model_shap_abs[feature] > shap_threshold and feature in model_rec_err:
                                shap_above_threshold_counter_in_rec[feature] += 1

                shap_agreement_in_rec[np.where(shap_above_threshold_counter_in_rec)] += 1
                shap_agreement_not_in_rec[np.where(shap_above_threshold_counter)] += 1

            if shap_agreement_in_rec.max() >= num_models_needed:
                is_anomaly_list_feature_in_rec[i] = 1
            if shap_agreement_not_in_rec.max() >= num_models_needed:
                is_anomaly_list_feature[i] = 1

    return is_anomaly_list_feature_in_rec, is_anomaly_list_feature


def get_mse_per_feature_error(models, X):
    mse_per_feature = defaultdict(list)
    for index, model in enumerate(models[:-1]):
        for sample in X:
            rec_calculate = np.power(model(torch.Tensor(sample)).detach().numpy() - sample, 2)
            mse_per_feature[index].append(rec_calculate)
    return mse_per_feature

def get_mse_per_example_error(models, X):
    mse_per_example = defaultdict(list)
    for index, sample in enumerate(X):
        for model in models[:-1]:
            rec_calculate = np.power(model(torch.Tensor(sample)).detach().numpy() - sample, 2)
            mse_per_example[index].append(rec_calculate.mean())
    return mse_per_example

def get_ensemble_anomalies(mses, threshold, num_models_needed):
    results = []
    for index in mses:
        result = (np.array(mses[index]) > threshold).sum() > num_models_needed
        results.append(result)
    return np.array(results)

