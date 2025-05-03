from MAA_base import MAABase
import torch
import numpy as np
from functools import wraps
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

from utils.baseframe_trainer import train_baseframe
from typing import List, Optional
import models
import os
import time
import glob
from utils.evaluate_visualization import evaluate_best_models


def log_execution_time(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        func_name = func.__name__
        print(f"GCA_time_series - '{func_name}' elapse time: {elapsed_time:.4f} sec")
        return result

    return wrapper


def generate_labels(y):
    y = np.array(y).flatten()
    labels = [0]
    for i in range(1, len(y)):
        if y[i] > y[i - 1]:
            labels.append(2)
        elif y[i] < y[i - 1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(labels)


class base_time_series(MAABase):
    def __init__(self, args, N_pairs: int, batch_size: int, num_epochs: int,
                 generators_names: List, discriminators_names: Optional[List],
                 ckpt_dir: str, output_dir: str,
                 window_sizes: int,
                 initial_learning_rate: float = 2e-5,
                 train_split: float = 0.8,
                 do_distill_epochs: int = 1,
                 cross_finetune_epochs: int = 5,
                 precise=torch.float32,
                 device=None,
                 seed: int = None,
                 ckpt_path: str = None,
                 gan_weights=None,
                 ):

        super().__init__(N_pairs, batch_size, num_epochs,
                         generators_names, discriminators_names,
                         ckpt_dir, output_dir,
                         initial_learning_rate,
                         train_split,
                         precise,
                         do_distill_epochs, cross_finetune_epochs,
                         device,
                         seed,
                         ckpt_path)

        self.args = args
        self.window_sizes = window_sizes
        self.generator_dict = {}
        self.discriminator_dict = {"default": models.Discriminator3}

        for name in dir(models):
            obj = getattr(models, name)
            if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
                lname = name.lower()
                if "generator" in lname:
                    key = lname.replace("generator_", "")
                    self.generator_dict[key] = obj
                elif "discriminator" in lname:
                    key = lname.replace("discriminator", "")
                    self.discriminator_dict[key] = obj

        self.gan_weights = gan_weights

        self.init_hyperparameters()

    @log_execution_time
    def process_data(self, data_path, start_row, end_row,  target_columns, feature_columns):
        """
        Process the input data by loading, splitting, and normalizing it.

        Args:
            data_path (str): Path to the CSV data file
            target_columns (list): Indices of target columns
            feature_columns (list): Indices of feature columns

        Returns:
            tuple: (train_x, test_x, train_y, test_y, y_scaler)
        """
        print(f"Processing data with seed: {self.seed}")  # Using self.seed

        # Load data
        data = pd.read_csv(data_path)
        print(f'dataset name: {data_path}')
        # Select target columns
        y = data.iloc[start_row:end_row, target_columns].values
        target_column_names = data.columns[target_columns]
        print("Target columns:", target_column_names)

        # Select feature columns
        x = data.iloc[start_row:end_row, feature_columns].values
        feature_column_names = data.columns[feature_columns]
        print("Feature columns:", feature_column_names)

        # Data splitting using self.train_split
        train_size = int(data.iloc[start_row:end_row].shape[0] * self.train_split)
        train_x, test_x = x[:train_size], x[train_size:]
        train_y, test_y = y[:train_size], y[train_size:]

        # Normalization
        self.x_scaler = MinMaxScaler(feature_range=(0, 1))  # Store scaler as instance variable
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))  # Store scaler as instance variable

        self.train_x = self.x_scaler.fit_transform(train_x)
        self.test_x = self.x_scaler.transform(test_x)

        self.train_y = self.y_scaler.fit_transform(train_y)
        self.test_y = self.y_scaler.transform(test_y)

        self.train_labels = generate_labels(self.train_y)
        self.test_labels = generate_labels(self.test_y)
        print(self.train_y[:5])
        print(self.train_labels[:5])
        # ------------------------------------------------------------------

    def create_sequences_combine(self, x, y, label, window_size, start):
        x_ = []
        y_ = []
        y_gan = []
        label_gan = []
        for i in range(start, x.shape[0]):
            tmp_x = x[i - window_size: i, :]
            tmp_y = y[i]
            tmp_y_gan = y[i - window_size: i + 1]
            tmp_label_gan = label[i - window_size: i + 1]

            x_.append(tmp_x)
            y_.append(tmp_y)
            y_gan.append(tmp_y_gan)
            label_gan.append(tmp_label_gan)

        x_ = torch.from_numpy(np.array(x_)).float()
        y_ = torch.from_numpy(np.array(y_)).float()
        y_gan = torch.from_numpy(np.array(y_gan)).float()
        label_gan = torch.from_numpy(np.array(label_gan)).float()
        return x_, y_, y_gan, label_gan

    @log_execution_time
    def init_dataloader(self):

        # Sliding Window Processing
        train_data_list = [
            self.create_sequences_combine(self.train_x, self.train_y, self.train_labels, w, self.window_sizes[-1])
            for w in self.window_sizes
        ]

        test_data_list = [
            self.create_sequences_combine(self.test_x, self.test_y, self.test_labels, w, self.window_sizes[-1])
            for w in self.window_sizes
        ]

        self.train_x_all = [x.to(self.device) for x, _, _, _ in train_data_list]
        self.train_y_all = train_data_list[0][1]
        self.train_y_gan_all = [y_gan.to(self.device) for _, _, y_gan, _ in train_data_list]
        self.train_label_gan_all = [label_gan.to(self.device) for _, _, _, label_gan in train_data_list]

        self.test_x_all = [x.to(self.device) for x, _, _, _ in test_data_list]
        self.test_y_all = test_data_list[0][1]
        self.test_y_gan_all = [y_gan.to(self.device) for _, _, y_gan, _ in test_data_list]
        self.test_label_gan_all = [label_gan.to(self.device) for _, _, _, label_gan in test_data_list]

        assert all(torch.equal(train_data_list[0][1], y) for _, y, _, _ in train_data_list), "Train y mismatch!"
        assert all(torch.equal(test_data_list[0][1], y) for _, y, _, _ in test_data_list), "Test y mismatch!"

        self.dataloaders = []

        for i, (x, y_gan, label_gan) in enumerate(
                zip(self.train_x_all, self.train_y_gan_all, self.train_label_gan_all)):
            shuffle_flag = ("transformer" in self.generator_names[i])
            dataloader = DataLoader(
                TensorDataset(x, y_gan, label_gan),
                batch_size=self.batch_size,
                shuffle=shuffle_flag,
                generator=torch.manual_seed(self.seed),
                drop_last=True
            )
            self.dataloaders.append(dataloader)

    def init_model(self,num_cls):
        assert len(self.generator_names) == self.N, "Generators and Discriminators mismatch!"
        assert isinstance(self.generator_names, list)
        for i in range(self.N):
            assert isinstance(self.generator_names[i], str)

        self.generators = []
        self.discriminators = []

        for i, name in enumerate(self.generator_names):
            x = self.train_x_all[i]
            y = self.train_y_all[i]

            GenClass = self.generator_dict[name]
            if "transformer" in name:
                gen_model = GenClass(x.shape[-1], output_len=y.shape[-1]).to(self.device)
            else:
                gen_model = GenClass(x.shape[-1], y.shape[-1]).to(self.device)

            self.generators.append(gen_model)

            DisClass = self.discriminator_dict[
                "default" if self.discriminators_names is None else self.discriminators_names[i]]
            dis_model = DisClass(self.window_sizes[i], out_size=y.shape[-1], num_cls=3).to(self.device)
            self.discriminators.append(dis_model)

    def init_hyperparameters(self, ):
        self.init_GDweight = []
        for i in range(self.N):
            row = [0.0] * self.N
            row[i] = 1.0
            row.append(1.0)
            self.init_GDweight.append(row)

        if self.gan_weights is None:
            final_row = [round(1.0 / self.N, 3)] * self.N + [1.0]
            self.final_GDweight = [final_row[:] for _ in range(self.N)]
        else:
            pass

        self.g_learning_rate = self.initial_learning_rate
        self.d_learning_rate = self.initial_learning_rate
        self.adam_beta1, self.adam_beta2 = (0.9, 0.999)
        self.schedular_factor = 0.1
        self.schedular_patience = 16
        self.schedular_min_lr = 1e-7

    def train(self, logger):
        return train_baseframe(self.generators[0], self.dataloaders[0],
                                                    self.y_scaler, self.train_x_all[0], self.train_y_all, self.test_x_all[0],
                                                    self.test_y_all,
                                                    self.num_epochs,
                                                    self.output_dir,
                                                    self.device,
                                                    logger=logger)

    def save_models(self, best_model_state):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ckpt_dir = os.path.join(self.ckpt_dir, timestamp)
        gen_dir = os.path.join(ckpt_dir, "generators")
        disc_dir = os.path.join(ckpt_dir, "discriminators")
        os.makedirs(gen_dir, exist_ok=True)
        os.makedirs(disc_dir, exist_ok=True)

        for i in range(self.N):
            self.generators[i].load_state_dict(best_model_state[i])
            self.generators[i].eval()

        for i, gen in enumerate(self.generators):
            gen_name = type(gen).__name__
            save_path = os.path.join(gen_dir, f"{i + 1}_{gen_name}.pt")
            torch.save(gen.state_dict(), save_path)

        for i, disc in enumerate(self.discriminators):
            disc_name = type(disc).__name__
            save_path = os.path.join(disc_dir, f"{i + 1}_{disc_name}.pt")
            torch.save(disc.state_dict(), save_path)

        print("All models saved with timestamp and identifier.")

    def get_latest_ckpt_folder(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        all_subdirs = [d for d in glob.glob(os.path.join(self.ckpt_dir, timestamp[0] + "*")) if os.path.isdir(d)]
        if not all_subdirs:
            raise FileNotFoundError("❌ No checkpoint records!!")
        latest = max(all_subdirs, key=os.path.getmtime)
        print(f"📂 Auto loaded checkpoint file: {latest}")
        return latest

    def load_model(self):
        gen_path = os.path.join(self.ckpt_path, "g{gru}", "generator.pt")
        if os.path.exists(gen_path):
            self.generators[0].load_state_dict(torch.load(gen_path, map_location=self.device))
            print(f"✅ Loaded generator from {gen_path}")
        else:
            raise FileNotFoundError(f"❌ Generator checkpoint not found at: {gen_path}")

    def pred(self):
        if self.ckpt_path == "auto":
            self.ckpt_path = self.get_latest_ckpt_folder()

        print("Start predicting with all generators..")
        best_model_state = [None for _ in range(self.N)]
        current_path = os.path.join(self.ckpt_path, "generatorrs")

        for i, gen in enumerate(self.generators):
            gen_name = type(gen).__name__
            save_path = os.path.join(current_path, f"{i + 1}_{gen_name}.pt")
            best_model_state[i] = self.generator_dict[self.generator_names[i]].load_state_dict(torch.load(save_path))

        results = evaluate_best_models(self.generators, best_model_state, self.train_x_all, self.train_y_all,
                                       self.test_x_all, self.test_y_all, self.y_scaler,
                                       self.output_dir)
        return results

    def distill(self):
        pass

    def visualize_and_evaluate(self):
        pass

    def init_history(self):
        pass
