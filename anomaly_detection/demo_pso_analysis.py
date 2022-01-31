import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

import models.utils as mutils
from sensei.sensei import Sensei
import data_utils.utils as dutils
from data_utils.dataset_wrapper import TimeSeriesDataset

# hidx = 5

# pso_name = 'train/pso_16_10_20220115-022127_gru_mae_tanh'
# pso_name = 'pso_16_10_20220114-030724_gru_rmse_tanh'
# pso_name = 'pso_16_10_20220114-150227_lstm_mse_tanh'
# pso_name = 'pso_16_10_20220114-164608_rnn_mse_tanh'
# pso_name = 'val/pso_32_16_20220117-121814_gru_mae_tanh'
# pso_name = 'val/pso_32_16_20220117-134940_gru_mae_tanh'
# pso_name = 'val/pso_4_4_20220117-182736_gru_mae_tanh'
# pso_name = '4-report_draft/val/pso_16_10_20220118-015555_gru_mae_tanh'
# pso_name = '4-report_draft/train/pso_16_10_20220118-033403_gru_mae_tanh'


# pso_name = '27-report_draft/train/pso_16_10_20220118-135738_gru_mae_tanh'
# pso_name = '27-report_draft/val/pso_16_10_20220118-120915_gru_mae_tanh'

# pso_name = 'ok/4-report_draft/val/pso_16_10_20220119-175854_gru_mae_tanh'
# pso_name = 'val_shuffled/4-report_draft/val/pso_16_10_20220119-185935_gru_mae_tanh'

# pso_name = '4/not_shuffled/val/pso_16_10_20220120-035734_lstm_mse_tanh'


def plot_best_models_comaprison(x, y, score, loss_fn, title="Result"):
    plt.close()
    c = ["darkred", "red", "lightcoral", "white", "palegreen", "green", "darkgreen"]
    v = [0, .15, .4, .5, 0.6, .9, 1.]
    l = list(zip(v, c))
    cmap = LinearSegmentedColormap.from_list('rg', l, N=256)
    # fig,ax = plt.subplots(1)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()

    if score.eq(0).all():
        score = 'darkred'
    elif score.eq(1).all():
        score = 'darkgreen'

    # plt.scatter(x, y, s=100, c=score, cmap=cmap, vmin=0, vmax=1)
    plt.scatter(x, y, s=100, c=score, cmap=cmap)
    plt.title(title)
    plt.xlabel(f'Validation Set Regression [{loss_fn}]', fontsize=18)
    plt.ylabel(f'Training Set Regression [{loss_fn}]', fontsize=16)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label='competition score [%]')
    plt.legend()
    plt.show()


# model_performance_filename = 'model_performance_shifted.csv'
# train_filename = 'summary_train_threshold_shifted.csv'
# test_filename = 'summary_test_threshold_shifted.csv'

model_performance_filename = 'model_performance.csv'
train_filename = 'summary_train_threshold.csv'
test_filename = 'summary_test_threshold.csv'

def analyze_pso(pso_name, loss_fn):
    # loss_fn = mutils.get_loss_fn(loss_fn)
    models, model_dir_names = mutils.get_all_models_from_dir(dir_name=pso_name)
    senseis = []
    metrics = []
    ad = []
    plot = False
    for i, (model, model_params, training_params, loss_history) in enumerate(models):
        print(f'--------- model no. {i} ----------')
        sensei = Sensei.from_trained(model, loss_fn, loss_history=loss_history)
        ts, dataset_description = TimeSeriesDataset.from_enum(dutils.Dataset.HEXAGON, int(training_params['hexagon_id']), 'minmax')

        dataloaders = ts.get_dataloaders(batch_size=training_params['batch_size'],
                                         input_size=model_params['input_size'])

        results_train = sensei.evaluate(test_loader=dataloaders.train_loader_one,
                                        plot=plot, title="Train Set Regression", scaler=ts.scaler)
        metrics_train_set = results_train[-1]

        results_val = sensei.evaluate(test_loader=dataloaders.val_loader_one,
                                      plot=plot, title='Validation Set Regression', scaler=ts.scaler)
        metrics_val_set = results_val[-1]

        results_test = sensei.evaluate(test_loader=dataloaders.test_loader_one,
                                       plot=plot, title='Test Set Regression', scaler=ts.scaler)
        metrics_test_set = results_test[-1]

        metrics.append((metrics_train_set, metrics_val_set, metrics_test_set))

        print(f'>>> train rmse: {metrics_train_set.rmse}')
        print(f'>>> val rmse: {metrics_val_set.rmse}')
        print(f'>>> test rmse: {metrics_test_set.rmse}')
        #
        # fig, ax = plt.subplots()
        # ax.plot(loss_history['train'], label="mean train loss per epoch")
        # ax.plot(loss_history['val'], label="mean val loss per epoch")
        # plt.legend()
        # # plt.show()

        if plot:
            sensei.plot_losses()

        results_ad_test_threshold = sensei.detect_anomalies_test_set_based_threshold(ts,
                                                                                     dataloaders.test_loader_one,
                                                                                     input_size=model_params[
                                                                                         'input_size'],
                                                                                     plot=plot)

        results_ad_train_threshold = sensei.detect_anomalies_train_set_based_threshold(ts,
                                                                                       dataloaders.train_loader_one,
                                                                                       dataloaders.test_loader_one,
                                                                                       input_size=model_params[
                                                                                           'input_size'],
                                                                                       plot=plot)

        ad.append((results_ad_test_threshold, results_ad_train_threshold))
        print(f'--------- /{i} ----------')
    from itertools import islice
    import pandas as pd
    # get best comp score, train set & val set losses
    scores_per_model_test_threshold = []
    scores_per_model_extended_test_threshold = []
    scores_per_model_train_threshold = []
    scores_per_model_extended_train_threshold = []
    train_evals = []
    val_evals = []
    test_evals = []

    quantile_no = 0
    highest_residue_no = 1
    second_highest_residue_no_in_dict = 2

    dict_no = highest_residue_no
    for i, (res_test, res_train) in enumerate(ad):
        scores_per_model_test_threshold.append(res_test[next(
            islice(iter(res_test), dict_no, dict_no + 1))][
                                                   1])  # competition score for
        scores_per_model_extended_test_threshold.append(res_test[next(
            islice(iter(res_test), dict_no, dict_no + 1))][2])
        scores_per_model_train_threshold.append(res_train[next(
            islice(iter(res_train), dict_no, dict_no + 1))][
                                                    1])  # competition score for
        scores_per_model_extended_train_threshold.append(res_train[next(
            islice(iter(res_train), dict_no, dict_no + 1))][2])
        # comp_score_per_model_test_threshold.append(res[next(islice(iter(res), quantile_no, quantile_no + 1))][1])
        # comp_score_per_model_extended_test_threshold.append(res[next(islice(iter(res), highest_residue_no, highest_residue_no + 1))][2])
        # comp_score_per_model_extended_test_threshold.append(res[next(islice(iter(res), second_highest_residue_no_in_dict, second_highest_residue_no_in_dict + 1))][2])

        train_evals.append(metrics[i][0])
        val_evals.append(metrics[i][1])
        test_evals.append(metrics[i][2])
    scores_per_model_test_threshold = pd.DataFrame(scores_per_model_test_threshold)
    # for stt in scores_per_model_test_threshold
    scores_per_model_extended_test_threshold = pd.DataFrame(scores_per_model_extended_test_threshold)
    scores_per_model_train_threshold = pd.DataFrame(scores_per_model_train_threshold)
    scores_per_model_extended_train_threshold = pd.DataFrame(scores_per_model_extended_train_threshold)
    train_evals = pd.DataFrame(train_evals)
    val_evals = pd.DataFrame(val_evals)
    test_vals = pd.DataFrame(test_evals)
    models_performance = pd.DataFrame(
        {
            'train_mae': train_evals.mae,
            'train_mse': train_evals.mse,
            'train_rmse': train_evals.rmse,
            'train_r2': train_evals.r2,
            'val_mae': val_evals.mae,
            'val_mse': val_evals.mse,
            'val_rmse': val_evals.rmse,
            'val_r2': val_evals.r2,
            'test_mae': test_vals.mae,
            'test_mse': test_vals.mse,
            'test_rmse': test_vals.rmse,
            'test_r2': test_vals.r2,
        })
    ad_summary_test_set_based = pd.DataFrame(
        {
            'comp_score': scores_per_model_test_threshold["competition_score"],
            'comp_score_extended': scores_per_model_extended_test_threshold["competition_score"],
            'ad_confusion_matrix': scores_per_model_test_threshold['confusion_matrix'],
            'ad_confusion_matrix_extended': scores_per_model_extended_test_threshold['confusion_matrix'],
            'most_prob_anomaly_score': scores_per_model_test_threshold['mp_competition_score'],
            'most_prob_anomaly_score_extended': scores_per_model_extended_test_threshold['mp_competition_score']
        })
    ad_summary_train_set_based = pd.DataFrame(
        {
            'comp_score': scores_per_model_train_threshold["competition_score"],
            'comp_score_extended': scores_per_model_extended_train_threshold["competition_score"],
            'ad_confusion_matrix': scores_per_model_train_threshold['confusion_matrix'],
            'ad_confusion_matrix_extended': scores_per_model_extended_train_threshold['confusion_matrix'],
            'most_prob_anomaly_score': scores_per_model_train_threshold['mp_competition_score'],
            'most_prob_anomaly_score_extended': scores_per_model_extended_train_threshold['mp_competition_score']
        })
    models_performance.insert(loc=0, column="name", value=model_dir_names, allow_duplicates=True)
    ad_summary_test_set_based.insert(loc=0, column="name", value=model_dir_names, allow_duplicates=True)
    ad_summary_train_set_based.insert(loc=0, column="name", value=model_dir_names, allow_duplicates=True)
    models_performance.to_csv(
        path_or_buf=mutils.get_trained_models_dir_path().joinpath(pso_name).joinpath(model_performance_filename),
        index=False)
    ad_summary_test_set_based.to_csv(
        path_or_buf=mutils.get_trained_models_dir_path().joinpath(pso_name).joinpath(test_filename),
        index=False)
    ad_summary_train_set_based.to_csv(
        path_or_buf=mutils.get_trained_models_dir_path().joinpath(pso_name).joinpath(train_filename),
        index=False)

    return models_performance, ad_summary_test_set_based, ad_summary_train_set_based

# root = './'
# pso_name = 'test_pso_rnn_mse_tanh'

# root = './4/not_shuffled/train'
# pso_name = 'pso_16_10_20220120-140622_gru_mse_tanh'

# root = './4/not_shuffled/train'
# pso_name = 'pso_16_10_20220120-140622_gru_mse_tanh'

# root = './27/not_shuffled/val'
# pso_name = 'pso_16_10_20220120-184158_lstm_mae_tanh'

# root = './90/not_shuffled/train'
# pso_name = 'pso_16_10_20220121-005151_gru_mse_tanh'

# root = './90/not_shuffled/val'
# pso_name = 'pso_16_10_20220121-021230_gru_mse_tanh'

# root = './90/not_shuffled/val'
# pso_name = 'pso_16_10_20220121-045114_gru_mae_tanh'

# root = './5/not_shuffled/train'
# pso_name = 'pso_16_10_20220121-133127_gru_mse_tanh'

# root = './5/not_shuffled/val'
# pso_name = 'pso_16_10_20220121-211343_gru_mse_tanh'

# root = './90/not_shuffled/train'
# pso_name = 'pso_16_10_20220122-044126_lstm_mse_tanh'

# root = './151/not_shuffled/val'
# pso_name = 'pso_16_10_20220122-235729_gru_mse_tanh'

# root = './151/not_shuffled/train'
# pso_name = 'pso_16_10_20220122-235751_gru_mse_tanh'

# root = './34/not_shuffled/train'
# pso_name = 'pso_16_10_20220122-235751_gru_mse_tanh'

root = 'yet_another_careless_mistake_fixed/29/val'
pso_name = 'pso_16_10_20220124-161650_gru_mae_tanh'


model_name, loss_fn, af = pso_name.split('_')[-3:]

abs_pso_name = mutils.get_trained_models_dir_path().joinpath(f'{root}/{pso_name}')

if all(x in dutils.get_all_filenames(abs_pso_name) for x in [model_performance_filename, train_filename, test_filename]):
    models_performance = pd.read_csv(abs_pso_name.joinpath(model_performance_filename))
    train_summary = pd.read_csv(abs_pso_name.joinpath(train_filename))
    test_summary = pd.read_csv(abs_pso_name.joinpath(test_filename))

else:
    models_performance, train_summary, test_summary = analyze_pso(f'{root}/{abs_pso_name.stem}', loss_fn=loss_fn)



plot_best_models_comaprison(x=models_performance.val_mae,
                            y=models_performance.train_mae,
                            score=train_summary.comp_score,
                            title=pso_name,
                            loss_fn=loss_fn)
plot_best_models_comaprison(x=models_performance.val_mae,
                            y=models_performance.train_mae,
                            score=train_summary.comp_score_extended,
                            title=pso_name,
                            loss_fn=loss_fn)

plot_best_models_comaprison(x=models_performance.val_mae,
                            y=models_performance.train_mae,
                            score=test_summary.most_prob_anomaly_score,
                            title=pso_name,
                            loss_fn=loss_fn)
plot_best_models_comaprison(x=models_performance.val_mae,
                            y=models_performance.train_mae,
                            score=test_summary.most_prob_anomaly_score_extended,
                            title=pso_name,
                            loss_fn=loss_fn)




# plot_best_models_comaprison(x=models_performance.val_mae,
#                             y=models_performance.train_mae,
#                             score=ad_summary_train_set_based.comp_score,
#                             title=pso_name)
# plot_best_models_comaprison(x=models_performance.val_mae,
#                             y=models_performance.train_mae,
#                             score=ad_summary_train_set_based.comp_score_extended,
#                             title=pso_name)
#
# plot_best_models_comaprison(x=models_performance.val_mae,
#                             y=models_performance.train_mae,
#                             score=ad_summary_test_set_based.most_prob_anomaly_score,
#                             title=pso_name)
# plot_best_models_comaprison(x=models_performance.val_mae,
#                             y=models_performance.train_mae,
#                             score=ad_summary_test_set_based.most_prob_anomaly_score_extended,
#                             title=pso_name)
#
# plt.gca().invert_xaxis()
# plt.plot(ad_summary_test_set_based.val_mae, ad_summary_test_set_based.comp_score_extended)

# plt.show()
print()

#%%
