from MAA_base import MAABase
import torch
import numpy as np
from functools import wraps
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

# from utils.multiGAN_trainer import train_multi_gan
from utils.multiGAN_trainer_disccls import train_multi_gan
from typing import List, Optional
import models
import os
import time
import glob
from utils.evaluate_visualization import evaluate_best_models
from utils.util import compute_logdiff

def log_execution_time(func):
    """Decorator: Record the execution time of the function and dynamically get the function name"""

    @wraps(func)  # Preserve the original function's metadata (like __name__)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Execute the target function
        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time  # Calculate elapsed time

        # Dynamically get the function name (supports class methods and regular functions)
        func_name = func.__name__
        print(f"MAA_time_series - '{func_name}' elapse time: {elapsed_time:.4f} sec")
        return result

    return wrapper


def generate_labels(y):
    """
    Generate three-category labels based on whether each time step y is higher than the previous moment:
      - 2: Current value > Previous moment (Rising)
      - 0: Current value < Previous moment (Falling)
      - 1: Current value == Previous moment (Stable)
    For the first time step, the default value is 1 (Stable).

    Args:
        y: Array, shape (num_samples, ) or (num_samples, 1)
    Returns:
        labels: Generated label array, same length as y
    """
    y = np.array(y).flatten()  # Convert to a 1D array
    labels = [0]  # For the first sample, default to stable
    for i in range(1, len(y)):
        if y[i] > y[i - 1]:
            labels.append(2)
        elif y[i] < y[i - 1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(labels)


class MAA_time_series(MAABase):
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
        """
        Initialize necessary hyperparameters.

        :param N_pairs: Number of generators or discriminators
        :param batch_size: Batch size
        :param num_epochs: Scheduled training epochs
        :param initial_learning_rate: Initial learning rate
        :param generators_names: list object, including names of generators with different features
        :param discriminators_names: list object, including names of discriminators, default is the same if not provided
        :param ckpt_dir: Directory to save model checkpoints
        :param output_path: Output directory for visualization, loss function logs, etc.
        :param ckpt_path: Checkpoint saved during prediction
        """
        super().__init__(N_pairs, batch_size, num_epochs,
                         generators_names, discriminators_names,
                         ckpt_dir, output_dir,
                         initial_learning_rate,
                         train_split,
                         precise,
                         do_distill_epochs, cross_finetune_epochs,
                         device,
                         seed,
                         ckpt_path)  # Call parent class initialization

        self.args = args
        self.window_sizes = window_sizes
        # Initialize empty dictionaries
        self.generator_dict = {}
        self.discriminator_dict = {"default": models.Discriminator3}

        # Iterate through all attributes in the models module
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
    def process_data(self, data_path, start_row, end_row,  target_columns, feature_columns_list, log_diff):
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

        # Select target columns
        y = data.iloc[start_row:end_row, target_columns].values
        target_column_names = data.columns[target_columns]
        print("Target columns:", target_column_names)


        # # Select feature columns
        # x = data.iloc[start_row:end_row, feature_columns].values
        # feature_column_names = data.columns[feature_columns]
        # print("Feature columns:", feature_column_names)

        # Process each set of feature columns
        x_list = []
        feature_column_names_list = []
        self.x_scalers = []  # Store multiple x scalers

        for feature_columns in feature_columns_list:
            # Select feature columns
            x = data.iloc[start_row:end_row, feature_columns].values
            feature_column_names = data.columns[feature_columns]
            print("Feature columns:", feature_column_names)

            x_list.append(x)
            feature_column_names_list.append(feature_column_names)

        # —— Calculate and print the overall mean and variance of y ——
        print(f"Overall  Y mean: {y.mean():.4f}, var: {y.var():.4f}")

        # Data splitting using self.train_split
        train_size = int(data.iloc[start_row:end_row].shape[0] * self.train_split)
        # train_x, test_x = x[:train_size], x[train_size:]
        # Split each x in the list
        train_x_list = [x[:train_size] for x in x_list]
        test_x_list = [x[train_size:] for x in x_list]
        train_y, test_y = y[:train_size], y[train_size:]

        # —— Perform log differencing on train/test x and y ——
        if log_diff:
            train_x_list = [compute_logdiff(x) for x in train_x_list]
            test_x_list = [compute_logdiff(x) for x in test_x_list]
            train_y = compute_logdiff(train_y)
            test_y = compute_logdiff(test_y)

        # —— Calculate and print the mean and variance of train and test ——
        print(f"Train    Y mean: {train_y.mean():.4f}, var: {train_y.var():.4f}")
        print(f"Test     Y mean: {test_y.mean():.4f}, var: {test_y.var():.4f}")

        # Normalize each x set separately
        self.train_x_list = []
        self.test_x_list = []
        for train_x, test_x in zip(train_x_list, test_x_list):
            x_scaler = MinMaxScaler(feature_range=(0, 1))
            self.train_x_list.append(x_scaler.fit_transform(train_x))
            self.test_x_list.append(x_scaler.transform(test_x))
            self.x_scalers.append(x_scaler)  # Store all x scalers

        # Normalization
        self.x_scaler = MinMaxScaler(feature_range=(0, 1))  # Store scaler as instance variable
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))  # Store scaler as instance variable

        # self.train_x = self.x_scaler.fit_transform(train_x)
        # self.test_x = self.x_scaler.transform(test_x)

        self.train_y = self.y_scaler.fit_transform(train_y)
        self.test_y = self.y_scaler.transform(test_y)

        # Generate classification labels for the training set (generated directly on GPU)
        self.train_labels = generate_labels(self.train_y)
        # Generate classification labels for the test set
        self.test_labels = generate_labels(self.test_y)
        print(self.train_y[:5])
        print(self.train_labels[:5])
        # ------------------------------------------------------------------

    def create_sequences_combine(self, x_list, y, label, window_size, start):
        x_ = []
        y_ = []
        y_gan = []
        label_gan = []
        # Create sequences for each x in x_list
        for x in x_list:
            x_seq = []
            for i in range(start, x.shape[0]):
                tmp_x = x[i - window_size: i, :]
                x_seq.append(tmp_x)
            x_.append(np.array(x_seq))

        # Combine x sequences along feature dimension
        x_ = np.concatenate(x_, axis=-1)

        for i in range(start, y.shape[0]):
            # tmp_x = x[i - window_size: i, :]
            tmp_y = y[i]
            tmp_y_gan = y[i - window_size: i + 1]
            tmp_label_gan = label[i - window_size: i + 1]

            # x_.append(tmp_x)
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
        """Initialize data loaders for training and evaluation"""

        # Sliding Window Processing
        # Generate sequence data for different window_sizes separately
        train_data_list = [
            self.create_sequences_combine(self.train_x_list, self.train_y, self.train_labels, w, self.window_sizes[-1])
            for w in self.window_sizes
        ]

        test_data_list = [
            self.create_sequences_combine(self.test_x_list, self.test_y, self.test_labels, w, self.window_sizes[-1])
            for w in self.window_sizes
        ]

        # Extract x, y, y_gan separately and stack them
        self.train_x_all = [x.to(self.device) for x, _, _, _ in train_data_list]
        self.train_y_all = train_data_list[0][1]  # All y should be the same, take the first one, no cuda needed for eval
        self.train_y_gan_all = [y_gan.to(self.device) for _, _, y_gan, _ in train_data_list]
        self.train_label_gan_all = [label_gan.to(self.device) for _, _, _, label_gan in train_data_list]

        self.test_x_all = [x.to(self.device) for x, _, _, _ in test_data_list]
        self.test_y_all = test_data_list[0][1]  # All y should be the same, take the first one, no cuda needed for eval
        self.test_y_gan_all = [y_gan.to(self.device) for _, _, y_gan, _ in test_data_list]
        self.test_label_gan_all = [label_gan.to(self.device) for _, _, _, label_gan in test_data_list]

        assert all(torch.equal(train_data_list[0][1], y) for _, y, _, _ in train_data_list), "Train y mismatch!"
        assert all(torch.equal(test_data_list[0][1], y) for _, y, _, _ in test_data_list), "Test y mismatch!"

        """
        train_x_all.shape  # (N, N, W, F)  Different window_sizes will result in different W, can only stack when W is the same
        train_y_all.shape  # (N,)
        train_y_gan_all.shape  # (3, N, W+1)
        """

        self.dataloaders = []

        for i, (x, y_gan, label_gan) in enumerate(
                zip(self.train_x_all, self.train_y_gan_all, self.train_label_gan_all)):
            shuffle_flag = ("transformer" in self.generator_names[i])  # Set the last one to shuffle=True, others to False
            dataloader = DataLoader(
                TensorDataset(x, y_gan, label_gan),
                batch_size=self.batch_size,
                shuffle=shuffle_flag,
                generator=torch.manual_seed(self.seed),
                drop_last=True  # Drop the last batch if its size is less than batch_size
            )
            self.dataloaders.append(dataloader)

    def init_model(self,num_cls):
        """Model structure initialization"""
        assert len(self.generator_names) == self.N, "Generators and Discriminators mismatch!"
        assert isinstance(self.generator_names, list)
        for i in range(self.N):
            assert isinstance(self.generator_names[i], str)

        self.generators = []
        self.discriminators = []

        for i, name in enumerate(self.generator_names):
            # Get corresponding x, y
            x = self.train_x_all[i]
            y = self.train_y_all[i]

            # Initialize generator
            GenClass = self.generator_dict[name]
            if "transformer" in name:
                gen_model = GenClass(x.shape[-1], output_len=y.shape[-1]).to(self.device)
            else:
                gen_model = GenClass(x.shape[-1], y.shape[-1]).to(self.device)

            self.generators.append(gen_model)

            # Initialize discriminator (default to Discriminator3 only)
            DisClass = self.discriminator_dict[
                "default" if self.discriminators_names is None else self.discriminators_names[i]]
            dis_model = DisClass(self.window_sizes[i], out_size=y.shape[-1], num_cls=num_cls).to(self.device)
            self.discriminators.append(dis_model)

    def init_hyperparameters(self, ):
        """Initialize hyperparameters required for training"""
        # Initialization: 1 on the diagonal, 0 otherwise, last column is 1.0
        self.init_GDweight = []
        for i in range(self.N):
            row = [0.0] * self.N
            row[i] = 1.0
            row.append(1.0)  # Last column is scale
            self.init_GDweight.append(row)

        if self.gan_weights is None:
            # Final: Equal division, last column is 1.0
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
        results, best_model_state = train_multi_gan(self.args, self.generators, self.discriminators, self.dataloaders,
                                                    self.window_sizes,
                                                    self.y_scaler, self.train_x_all, self.train_y_all, self.test_x_all,
                                                    self.test_y_all, self.test_label_gan_all,
                                                    self.do_distill_epochs,self.cross_finetune_epochs,
                                                    self.num_epochs,
                                                    self.output_dir,
                                                    self.device,
                                                    init_GDweight=self.init_GDweight,
                                                    final_GDweight=self.final_GDweight,
                                                    logger=logger)

        self.save_models(best_model_state)
        return results

    def save_models(self, best_model_state):
        """
        Save the model parameters of all generators and discriminators, including timestamp, model name or number.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ckpt_dir = os.path.join(self.ckpt_dir, timestamp)
        gen_dir = os.path.join(ckpt_dir, "generators")
        disc_dir = os.path.join(ckpt_dir, "discriminators")
        os.makedirs(gen_dir, exist_ok=True)
        os.makedirs(disc_dir, exist_ok=True)

        # Load models and set to eval mode
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
            raise FileNotFoundError("No checkpoint records!!")
        latest = max(all_subdirs, key=os.path.getmtime)
        print(f"Auto loaded checkpoint file: {latest}")
        return latest

    def load_model(self):
        gen_path = os.path.join(self.ckpt_path, "g{gru}", "generator.pt")
        if os.path.exists(gen_path):
            self.generators[0].load_state_dict(torch.load(gen_path, map_location=self.device))
            print(f"Loaded generator from {gen_path}")
        else:
            raise FileNotFoundError(f"Generator checkpoint not found at: {gen_path}")

    def pred(self):
        if self.ckpt_path == "auto":
            self.ckpt_path = self.get_latest_ckpt_folder()

        print("Start predicting with all generators..")
        best_model_state = [None for _ in range(self.N)]
        current_path = os.path.join(self.ckpt_path, "generators")

        for i, gen in enumerate(self.generators):
            gen_name = type(gen).__name__
            save_path = os.path.join(current_path, f"{i + 1}_{gen_name}.pt")
            state_dict = torch.load(save_path, map_location=self.device)
            gen.load_state_dict(state_dict)
            best_model_state[i] = state_dict

        results = evaluate_best_models(self.generators, best_model_state, self.train_x_all, self.train_y_all,
                                       self.test_x_all, self.test_y_all, self.y_scaler,
                                       self.output_dir)

        # —— New: Iterate through each generator and save the true/predicted values from "normalized" to "original price" into CSV ——
        with torch.no_grad():
            for i, gen in enumerate(self.generators):
                gen.eval()
                # Prepare input, true y
                x_test = self.test_x_all[i]  # Tensor on device, shape=(N, W, F)
                y_true_norm = self.test_y_all.cpu().numpy()  # shape=(N,)
                # Forward prediction (after normalization)
                y_pred_norm = gen(x_test)[0].cpu().numpy().reshape(-1, 1)  # (N,1)
                # Inverse normalize back to original values
                y_true = self.y_scaler.inverse_transform(y_true_norm.reshape(-1, 1)).flatten()
                y_pred = self.y_scaler.inverse_transform(y_pred_norm).flatten()

                df = pd.DataFrame({
                    'true': y_true,
                    'pred': y_pred
                })
                csv_save_path = "true2pred"
                if not os.path.exists(csv_save_path):
                    os.makedirs(csv_save_path)
                out_path = os.path.join(csv_save_path, f'predictions_gen{i + 1}.csv')
                df.to_csv(out_path, index=False)
                print(f"Saved true vs pred for generator {i + 1} at: {out_path}")

        return results

    def distill(self):
        """Evaluate model performance and visualize results"""
        pass

    def visualize_and_evaluate(self):
        """Evaluate model performance and visualize results"""
        pass

    def init_history(self):
        """Initialize the metric recording structure during training"""
        pass