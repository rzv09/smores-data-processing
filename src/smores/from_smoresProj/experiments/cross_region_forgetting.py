import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os

import data_processing.three_oec.process_3oec as process_3oec
import data_processing.three_oec.sequences_3oec as sequences_3oec
import data_processing.santa_barbara.process_sb as process_sb
from data_processing.data_utils import standardize_piece, create_sequences_single_step, make_sequence_tensor
import models.lstm.single_step as lstm_single_step
import plotting.plot_preds
import utils
import utils.set_seeds
from metrics.np.regression import MARE
from utils.write_metrics import write_csv, write_stats_mare, write_stats_crit
from plotting.plot_mare import plot_MARE
from plotting.plot_smoothl1loss import plot_smooth_l1_loss


class CrossRegionForgetting():
    def __init__(self, data_path_sb, data_path_fl, num_train_epochs, sampling_freq, sequence_len, device,
                 data_dir_smores=None, smores_salinity_psu=35.0, smores_deployment_start=None):
        self.data_path_sb = data_path_sb
        self.data_path_fl = data_path_fl
        self.num_train_epochs = num_train_epochs
        self.sampling_freq = sampling_freq
        self.sequence_len = sequence_len
        self.device = device
        # Optional SMORES BB folds (SMORES_preprocessing_plan_v2.md S8-S9).
        # data_dir_smores is a directory of raw SENS_<serial>_<chunk>_<Kind>.csv
        # chunks; leave None to reproduce the FL/SB-only experiment exactly.
        self.data_dir_smores = data_dir_smores
        self.smores_salinity_psu = smores_salinity_psu
        self.smores_deployment_start = smores_deployment_start

    def process_data(self):
        # preprocess SB data
        self.df_sb = process_sb.make_timeseries(self.data_path_sb)
        # preprocess FL data
        self.df_fl = process_3oec.resample(process_3oec.make_timeseries(self.data_path_fl), self.sampling_freq)

        pieces = {
            "sb1": process_sb.get_first_piece(self.df_sb),
            "sb2": process_sb.get_second_piece(self.df_sb),
            "sb3": process_sb.get_third_piece(self.df_sb),
            "sb4": process_sb.get_fourth_piece(self.df_sb),
            "fl1": process_3oec.get_first_piece(self.df_fl),
            "fl2": process_3oec.get_second_piece(self.df_fl),
            "fl3": process_3oec.get_third_piece(self.df_fl),
            "fl4": process_3oec.get_fourth_piece(self.df_fl),
        }

        if self.data_dir_smores is not None:
            from data_processing.smores.folds import build_pieces as build_smores_pieces
            pieces.update(build_smores_pieces(
                self.data_dir_smores,
                salinity_psu=self.smores_salinity_psu,
                deployment_start=self.smores_deployment_start,
            ))

        self.dataset_info = {}
        for label, piece in pieces.items():
            mean = piece.mean().values
            std = piece.std().values
            seqs, labels = create_sequences_single_step(standardize_piece(piece).values, self.sequence_len)
            seq_tensor = make_sequence_tensor(seqs, self.device)
            lbl_tensor = make_sequence_tensor(labels, self.device)
            self.dataset_info[label] = {
                "mean": mean,
                "std": std,
                "seq_tensor": seq_tensor,
                "lbl_tensor": lbl_tensor,
            }

    def set_train_data(self, train_label):
        self.train_seq = self.dataset_info[train_label]["seq_tensor"]
        self.train_lbls = self.dataset_info[train_label]["lbl_tensor"]

    def train_model(self, model: nn.Module, num_epochs: int, model_label: str, train_label):
        """
        Fine-tunes the model on a single dataset without batches.
        """
        self.set_train_data(train_label)

        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.SmoothL1Loss()

        train_losses = []

        for epoch in range(num_epochs):
            for k in range(len(self.train_seq)):
                model.train()
                optimizer.zero_grad()
                pred = model(self.train_seq[k])

                loss = criterion(pred, self.train_lbls[k])
                loss.backward()
                optimizer.step()

            train_losses.append(loss.item())
            if epoch % 10 == 0:
                print(f"Model {model_label}, Epoch {epoch}, Train Loss {loss.item()}")

        return model, train_losses

    def train_model_es(self, model: nn.Module, num_epochs: int, model_label: str, train_label, patience: int = 10):
        """
        Fine-tunes the model on a single dataset with early stopping.
        """
        self.set_train_data(train_label)

        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.SmoothL1Loss()

        train_losses = []

        best_train_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for k in range(len(self.train_seq)):
                optimizer.zero_grad()
                pred = model(self.train_seq[k])
                loss = criterion(pred, self.train_lbls[k])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(self.train_seq)
            train_losses.append(avg_train_loss)

            if epoch % 10 == 0:
                print(f"Model {model_label}, Epoch {epoch}, Avg Train Loss {avg_train_loss:.4f}")

            # ---- Early Stopping ----
            if avg_train_loss < best_train_loss - 1e-6:
                best_train_loss = avg_train_loss
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}. Best Train Loss: {best_train_loss:.4f}")
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return model, train_losses

    def rolling_average_model(self, test_seq, window_size):
        """
        Simple moving avg model that serves as a baseline
        """

        preds = []
        test_seq = test_seq.cpu().numpy()
        for i in range(len(test_seq)):
            pred = np.mean(test_seq[i:i + window_size])
            preds.append(pred)
        return preds

    def test_model(self, model: nn.Module, model_lbl: str, seed: int, test_seq: torch.FloatTensor,
                   test_lbl: torch.FloatTensor, prefix: str, ds_lbl: str, train_losses: list, save_dir: str):
        """
        Test a trained model
        """
        preds = []
        model.eval()

        with torch.no_grad():
            for i in range(len(test_seq)):
                pred = model(test_seq[i])

                if self.device != 'cpu':
                    preds.append(pred.cpu())
                else:
                    preds.append(pred)

        baseline_preds = self.rolling_average_model(test_seq, 12)
        plotting.plot_preds.plot_preds_from_device(preds, baseline_preds, test_lbl, filename_prefix=prefix, top_dir=save_dir)

        l1_smooth_error = nn.SmoothL1Loss()
        l1_smooth_value = l1_smooth_error(torch.FloatTensor(preds).to(device='cpu'), test_lbl.squeeze(1).to(device='cpu')).item()
        l1_smooth_value_baseline = l1_smooth_error(torch.FloatTensor(baseline_preds).to(device='cpu'), test_lbl.squeeze(1).to(device='cpu')).item()

        unnormalized_preds, unnormalized_lbls = self.unnormalize(preds, ds_lbl)
        unnormalized_preds_baseline, _ = self.unnormalize(baseline_preds, ds_lbl)
        error = MARE(unnormalized_preds, unnormalized_lbls)
        error_baseline = MARE(unnormalized_preds_baseline, unnormalized_lbls)
        csv_filepath = write_csv(model_lbl, ds_lbl, seed, error.item(), error_baseline.item(), l1_smooth_error,
                                 l1_smooth_value, l1_smooth_value_baseline, train_losses, save_dir)
        return csv_filepath

    def run_experiment(self, num_runs: int, save_dir: str, train_order, eval_all: bool = False,
                       early_stop: bool = False):
        os.mkdir(save_dir)
        l1_smooth_error = nn.SmoothL1Loss()
        self.process_data()

        train_order = self._normalize_labels(train_order)
        all_labels_order = list(self.dataset_info.keys())

        for i in range(num_runs):
            utils.set_seeds.set_experiment_seeds(i)
            model = lstm_single_step.create_lstm_single_step()

            for step_idx, train_label in enumerate(train_order, start=1):
                model_label = f"step{step_idx}_train_{train_label}"
                if early_stop:
                    model, train_losses = self.train_model_es(model, self.num_train_epochs, model_label, train_label)
                else:
                    model, train_losses = self.train_model(model, self.num_train_epochs, model_label, train_label)

                if eval_all:
                    eval_labels = all_labels_order
                else:
                    eval_labels = train_order[:step_idx]

                self.set_test_data(eval_labels)
                for test_seq, test_lbl, ds_lbl in zip(self.test_seqs, self.test_lbls, self.dataset_lbls):
                    prefix = f"{model_label},seed_{i},{ds_lbl}"
                    csv_filepath = self.test_model(model, model_label, i, test_seq, test_lbl, prefix, ds_lbl,
                                                   train_losses, save_dir)

        write_stats_mare(csv_filepath, save_dir)
        write_stats_crit(csv_filepath, save_dir, l1_smooth_error)
        plot_MARE(csv_filepath, save_dir)
        plot_smooth_l1_loss(csv_filepath, save_dir)

    def unnormalize(self, preds: torch.FloatTensor, ds_lbl: str):
        preds = torch.FloatTensor(preds)
        preds = preds.cpu()
        info = self.dataset_info.get(ds_lbl)
        if info is None:
            print('unnormalize_pred(): Incorrect dataset label')
            return False
        y_hat = preds.numpy() * info["std"] + info["mean"]
        unnormed_labels = info["lbl_tensor"].squeeze(1).cpu().numpy() * info["std"] + info["mean"]
        return y_hat, unnormed_labels

    def set_test_data(self, test_labels, test_only: bool = False):
        self.dataset_lbls = list(test_labels)
        self.test_seqs = [self.dataset_info[label]["seq_tensor"] for label in test_labels]
        self.test_lbls = [self.dataset_info[label]["lbl_tensor"] for label in test_labels]

    def _normalize_labels(self, labels):
        normalized = [label.lower() for label in labels]
        invalid = [label for label in normalized if label not in self.dataset_info]
        if invalid:
            raise ValueError(f"Unknown labels: {invalid}")
        return normalized
