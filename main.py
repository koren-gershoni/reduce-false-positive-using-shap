import os
import argparse
import neptune.new as neptune
from collections import defaultdict

from ExplainAnomaliesUsingSHAP import ExplainAnomaliesUsingSHAP
from config import config
from data_utils import load_data
from general_utils import update_global_seed
from model import load_models, AEModel, AEEncoder, AEDecoder
from shap_rec_utils import get_rec_and_shap, load_rec_and_shap, get_mse_per_example_error, get_ensemble_anomalies, \
    get_anomaly_list_shap, calculate_counter_list
from training_utils import run_training_flow

model_frac = 0.8


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--dataset", required=True, type=str, help="dataset name")
parser.add_argument("--seed", required=True, type=int, help="seed to be used")
parser.add_argument('--do_train', dest='do_train', action='store_true')
parser.set_defaults(do_train=False)

args = parser.parse_args()

seed = args.seed
update_global_seed(seed)

dataset = args.dataset

os.makedirs(f"./output/{dataset}", exist_ok=True)

config.saving_dir = f"./output/{dataset}/{dataset}_{seed}_{model_frac}"
config.model_frac = model_frac

original_df, X_train, y_train, X_val, y_val, X_test, y_test, label_map = load_data(dataset, seed)
config.num_columns = X_train.shape[1]

if args.do_train:
    run_training_flow(config, original_df, dataset, seed, label_map, AE=True)
models = load_models()
explainer = ExplainAnomaliesUsingSHAP(num_anomalies_to_explain=1, explanation="shap")

################### ensemble #############################
val_mses = get_mse_per_example_error(models, X_val)
test_mses = get_mse_per_example_error(models, X_test)

test_rec, test_shap = load_rec_and_shap(config, "test", partitions=False)
val_rec, val_shap = load_rec_and_shap(config, "val", partitions=False)



def run_neptune():
    run = neptune.init(project="",
                       api_token="",
    )  # your credentials
    return run
# run['dataset'] = dataset
# run['shap_threshold'] = args.shap_threshold
# run['seed'] = args.seed


def save_to_neptune(run, shap_threshold, val_anomaly_list_ensemble, test_anomaly_list_ensemble, val_anomaly_list_a, val_anomaly_list_b, test_anomaly_list_c, test_anomaly_list_d, val_anomaly_list_e, val_anomaly_list_f, test_anomaly_list_g, test_anomaly_list_h):
    run['dataset'] = args.dataset
    run['seed'] = args.seed
    run['shap_threshold'] = shap_threshold
    run['val_ensemble'] = val_anomaly_list_ensemble.sum().astype(int)
    run['test_ensemble'] = test_anomaly_list_ensemble.sum().astype(int)
    run['val_our_nofeatureshap_inrec'] = val_anomaly_list_a.sum().astype(int)
    run['val_our_nofeatureshap'] = val_anomaly_list_b.sum().astype(int)
    run['test_our_nofeatureshap'] = test_anomaly_list_c.sum().astype(int)
    run['test_our_nofeatureshap'] = test_anomaly_list_d.sum().astype(int)
    run['val_our_inrec'] = val_anomaly_list_e.sum().astype(int)
    run['val_our'] = val_anomaly_list_f.sum().astype(int)
    run['test_our_inrec'] = test_anomaly_list_g.sum().astype(int)
    run['test_our'] = test_anomaly_list_h.sum().astype(int)


thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
shap_thresholds = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08]
for threshold in thresholds:
    for num_models_agree_needed in range(1, 11):
        ################### ensemble #############################
        val_anomaly_list_ensemble = get_ensemble_anomalies(val_mses, threshold, num_models_agree_needed)
        test_anomaly_list_ensemble = get_ensemble_anomalies(test_mses, threshold, num_models_agree_needed)
        print("num models needed", num_models_agree_needed, "threshold", threshold, "sum val:", val_anomaly_list_ensemble.sum(), "sum test:", test_anomaly_list_ensemble.sum())
        for shap_threshold in shap_thresholds:
            run = run_neptune()
            ######################## feature level #############################
            counter_array_val, feature_model_agree_val = calculate_counter_list(val_rec, threshold)
            counter_array_test, feature_model_agree_test = calculate_counter_list(test_rec, threshold)

            # first output = count the shap value for just for features with high rec (inside our high rec list)
            # second output = count the shap value for all features
            # if include_anomaly_feature_shap == False then do not count for the shap of the anomaly feature itself
            # if include_anomaly_feature_shap == True then count also for the shap of the anomaly feature itself

            # don't allow the anomaly feature to be also in the shap anomaly count
            val_anomaly_list_a, val_anomaly_list_b = get_anomaly_list_shap(X_val, counter_array_val, val_shap, feature_model_agree_val, num_models_agree_needed, shap_threshold=shap_threshold, rec_err=val_rec, include_anomaly_feature_shap=False)
            test_anomaly_list_c, test_anomaly_list_d = get_anomaly_list_shap(X_test, counter_array_test, test_shap, feature_model_agree_test, num_models_agree_needed, shap_threshold=shap_threshold, rec_err=test_rec, include_anomaly_feature_shap=False)
            # allow the anomaly feature to be also in the shap anomaly count
            val_anomaly_list_e, val_anomaly_list_f = get_anomaly_list_shap(X_val, counter_array_val, val_shap, feature_model_agree_val, num_models_needed=5, shap_threshold=shap_threshold, rec_err=val_rec, include_anomaly_feature_shap=True)
            test_anomaly_list_g, test_anomaly_list_h = get_anomaly_list_shap(X_test, counter_array_test, test_shap, feature_model_agree_test, num_models_needed=5, shap_threshold=shap_threshold, rec_err=test_rec, include_anomaly_feature_shap=True)
            save_to_neptune(run, shap_threshold, val_anomaly_list_ensemble, test_anomaly_list_ensemble, val_anomaly_list_a, val_anomaly_list_b, test_anomaly_list_c, test_anomaly_list_d, val_anomaly_list_e, val_anomaly_list_f, test_anomaly_list_g, test_anomaly_list_h)





