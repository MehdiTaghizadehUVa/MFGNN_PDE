# Import required libraries
import os
import pickle
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import json
from dgl.dataloading import GraphDataLoader
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from dgl.nn import EdgeConv
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from dgl import graph as dgl_graph
from torch.nn.functional import mse_loss
from pathlib import Path
from typing import List, Tuple, Dict, Any
from torch.optim import Optimizer

# Import custom modules (ensure these modules are properly defined in your project)
from modulus.models.meshgraphnet import MeshGraphNet
from modulus.distributed.manager import DistributedManager
from modulus.launch.logging import PythonLogger, initialize_wandb, RankZeroLoggingWrapper
from modulus.launch.utils import load_checkpoint, save_checkpoint
from constants_MFGNN_H import Constants

try:
    import wandb as wb
except ImportError:
    wb = None

# Suppress warnings from deprecated features
import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

# Constants and global configurations
C = Constants()

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)  # For all GPUs
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 22}
plt.rc('font', **font)

plt.rcParams.update({
    'font.size': 22,              # Global default font size
    'axes.titlesize': 24,        # Axes title fontsize
    'axes.labelsize': 24,        # X and Y label fontsize
    'xtick.labelsize': 24,       # X-tick label fontsize
    'ytick.labelsize': 24,       # Y-tick label fontsize
    'legend.fontsize': 22,       # Legend fontsize
    'figure.titlesize': 24       # Figure title fontsize
})

def preprocess_data(training_data: List[Tuple]) -> Tuple[List[dgl.DGLGraph], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Preprocesses the training data by creating graphs and normalizing features and labels using min-max normalization.

    Args:
        training_data (list): A list containing training data tuples.

    Returns:
        tuple: A tuple containing the processed graphs, feature normalization parameters, and label normalization parameters.
    """
    graphs = []
    all_features = []  # Used for feature normalization
    all_labels = []  # Used for label normalization

    for geometry, mesh, input_features, target_labels in training_data:
        element_nodes = mesh['element_nodes'] - 1
        nodes = torch.tensor(mesh['nodes'], dtype=torch.float)
        node_coordinates = nodes.clone()
        edges = [(elements[i], elements[(i + 1) % len(elements)]) for elements in element_nodes for i in range(len(elements))]
        src, dst = np.array(edges).T
        g = dgl_graph((src, dst))
        rel_distances = nodes[dst] - nodes[src]
        rel_distances_norm = torch.norm(rel_distances, dim=1, keepdim=True)
        edge_features = torch.cat([rel_distances, rel_distances_norm], dim=1)
        boundary_conditions = torch.tensor(input_features['boundary_conditions'], dtype=torch.float)
        node_features = torch.cat((node_coordinates, boundary_conditions), dim=1)
        all_features.append(node_features)
        target = torch.tensor(target_labels['von_mises_stress'], dtype=torch.float)
        all_labels.append(target)

        g.ndata['x'] = node_features
        g.edata['x'] = edge_features
        g.ndata['y'] = target.unsqueeze(1)
        graphs.append(g)

    all_features = torch.cat(all_features, dim=0)
    feature_min = all_features.min(dim=0, keepdim=True).values
    feature_max = all_features.max(dim=0, keepdim=True).values
    feature_range = feature_max - feature_min

    all_labels = torch.cat(all_labels)
    label_min = all_labels.min()
    label_max = all_labels.max()
    label_range = label_max - label_min

    for g in graphs:
        g.ndata['x'] = (g.ndata['x'] - feature_min) / feature_range
        g.ndata['y'] = (g.ndata['y'] - label_min) / label_range

    return graphs, (feature_min, feature_max), (label_min, label_max)

def preprocess_test_data(test_data: List[Tuple], data_type: str) -> List[dgl.DGLGraph]:
    """Preprocesses the test data by creating graphs and normalizing features and labels using min-max normalization.

    Args:
        test_data (list): A list containing test data tuples.
        data_type (str): The type of the data (e.g., 'LF' or 'HF').

    Returns:
        list: A list containing the processed and normalized graphs.
    """
    graphs = []

    for geometry, mesh, input_features, target_labels in test_data:
        element_nodes = mesh['element_nodes'] - 1
        nodes = torch.tensor(mesh['nodes'], dtype=torch.float)
        node_coordinates = nodes.clone()
        edges = [(elements[i], elements[(i + 1) % len(elements)]) for elements in element_nodes for i in range(len(elements))]
        src, dst = np.array(edges).T
        g = dgl_graph((src, dst))
        rel_distances = nodes[dst] - nodes[src]
        rel_distances_norm = torch.norm(rel_distances, dim=1, keepdim=True)
        edge_features = torch.cat([rel_distances, rel_distances_norm], dim=1)
        boundary_conditions = torch.tensor(input_features['boundary_conditions'], dtype=torch.float)
        node_features = torch.cat((node_coordinates, boundary_conditions), dim=1)
        target = torch.tensor(target_labels['von_mises_stress'], dtype=torch.float)
        g.ndata['x'] = node_features
        g.edata['x'] = edge_features
        g.ndata['y'] = target.unsqueeze(1)
        graphs.append(g)

    stat_path = os.path.join(C.ckpt_path, C.ckpt_name_hf, f'normalization_params_{data_type}.json')
    with open(stat_path, 'r') as f:
        normalization_params = json.load(f)

    label_min = torch.tensor(normalization_params['label_min'])
    label_max = torch.tensor(normalization_params['label_max'])
    label_range = label_max - label_min

    feature_min = torch.tensor(normalization_params['feature_min'])
    feature_max = torch.tensor(normalization_params['feature_max'])
    feature_range = feature_max - feature_min

    for g in graphs:
        g.ndata['x'] = (g.ndata['x'] - feature_min) / feature_range
        g.ndata['y'] = (g.ndata['y'] - label_min) / label_range

    return graphs

class MGNTrainer:
    """Trainer class for MeshGraphNet model.

    Attributes:
        best_val_loss (float): Best validation loss.
        low_fidelity_losses (dict): Dictionary to store training and validation losses.
        wb (object): Weights and Biases object for logging.
        dist (DistributedManager): Distributed manager for handling multi-GPU training.
        rank_zero_logger (PythonLogger): Logger for rank zero processes.
    """

    def __init__(self, wb: object, dist: DistributedManager, rank_zero_logger: PythonLogger):
        self.best_val_loss = float('inf')
        self.low_fidelity_losses = {'train': [], 'val': []}
        self.wb = wb
        self.dist = dist
        self.rank_zero_logger = rank_zero_logger
        self._load_dataset()
        self._initialize_model()

    def _load_dataset(self) -> None:
        """Loads the training dataset and preprocesses it."""
        self.rank_zero_logger.info("Loading the training dataset...")
        with open(f'{C.data_dir}/MF_train_data_LF.pkl', 'rb') as f:
            training_data = pickle.load(f)
        training_data_LF = training_data[:C.N_LF]
        low_fidelity_data, (feature_min, feature_max), (label_min, label_max) = preprocess_data(training_data_LF)
        self.normalization_params = {
            'label_min': label_min.item(),
            'label_max': label_max.item(),
            'feature_min': feature_min.tolist(),
            'feature_max': feature_max.tolist()
        }

        stat_path = os.path.join(C.ckpt_path, C.ckpt_name_hf)
        if not Path(stat_path).is_dir():
            Path(stat_path).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(stat_path, 'normalization_params_LF.json'), 'w') as f:
            json.dump(self.normalization_params, f)

        self.dataset, self.validation_dataset = train_test_split(low_fidelity_data, test_size=0.1, random_state=C.random_seed)
        self.dataloader = GraphDataLoader(self.dataset, batch_size=C.batch_size, shuffle=True, drop_last=True,
                                          pin_memory=True, use_ddp=self.dist.world_size > 1)
        self.validation_dataloader = GraphDataLoader(self.validation_dataset, batch_size=C.batch_size, shuffle=False,
                                                     drop_last=True, pin_memory=True, use_ddp=False)

    def _initialize_model(self) -> None:
        """Initializes the MeshGraphNet model, optimizer, scheduler, and scaler."""
        self.model = MeshGraphNet(
            C.input_dim_nodes, C.input_dim_edges, C.output_dim, aggregation=C.aggregation, processor_size= C.processor_size,
            hidden_dim_node_encoder=C.hidden_dim_node_encoder, hidden_dim_edge_encoder=C.hidden_dim_edge_encoder,
            hidden_dim_node_decoder=C.hidden_dim_node_decoder, hidden_dim_processor = C.hidden_dim_processor
        ).to(self.dist.device)
        if self.dist.world_size > 1:
            self.model = DistributedDataParallel(self.model, device_ids=[self.dist.local_rank],
                                                 output_device=self.dist.device)
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=C.lr, weight_decay=C.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: C.lr_decay_rate ** (epoch / len(self.dataloader))
        )
        self.scaler = GradScaler()
        self.epoch_init = load_checkpoint(
            os.path.join(C.ckpt_path, C.ckpt_name_lf),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.dist.device,
        )

    def get_lr(self) -> float:
        """Returns the current learning rate."""
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def denormalize(self, normalized_labels: torch.Tensor) -> torch.Tensor:
        """Denormalizes the labels based on the provided normalization parameters.

        Args:
            normalized_labels (torch.Tensor): Normalized labels.

        Returns:
            torch.Tensor: Denormalized labels.
        """
        label_min = self.normalization_params['label_min']
        label_max = self.normalization_params['label_max']
        denormalized_labels = normalized_labels * (label_max - label_min) + label_min
        return denormalized_labels

    def train_epoch(self) -> float:
        """Trains the model for one epoch and returns the average training loss.

        Returns:
            float: Average training loss.
        """
        total_loss = 0
        for graph in self.dataloader:
            graph = graph.to(self.dist.device)
            self.optimizer.zero_grad()
            with autocast(enabled=C.amp):
                pred = self.model(graph.ndata["x"], graph.edata["x"], graph)
                diff_norm = torch.norm(
                    torch.flatten(pred) - torch.flatten(graph.ndata["y"]), p=2
                )
                y_norm = torch.norm(torch.flatten(graph.ndata["y"]), p=2)
                loss = diff_norm / y_norm
            if C.amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item()
        self.scheduler.step()
        return total_loss / len(self.dataloader)

    @torch.no_grad()
    def validation(self) -> float:
        """Validates the model on the validation dataset and returns the validation loss.

        Returns:
            float: Validation loss.
        """
        error = 0
        loss_agg = 0
        for graph in self.validation_dataloader:
            graph = graph.to(self.dist.device)
            pred = self.model(graph.ndata["x"], graph.edata["x"], graph)
            pred_de = self.denormalize(pred)
            gt = self.denormalize(graph.ndata["y"])
            error += (
                torch.mean(torch.norm(pred_de - gt, p=2) / torch.norm(gt, p=2))
                .cpu()
                .numpy()
            )
            loss = torch.mean((pred - graph.ndata["y"]) ** 2)
            loss_agg += loss.cpu().numpy()
        error = error / len(self.validation_dataloader) * 100
        self.wb.log({"val_error (%)": error})
        self.rank_zero_logger.info(f"validation error (%): {error}")
        loss_agg = loss_agg / len(self.validation_dataloader)
        if loss_agg < self.best_val_loss and self.dist.rank == 0:
            self.best_val_loss = loss_agg
            self.save_best_model()
        return loss_agg

    def save_best_model(self) -> None:
        """Saves the best model based on validation loss."""
        best_model_path = os.path.join(C.ckpt_path, C.ckpt_name_lf, "best_model_LF")
        save_checkpoint(
            best_model_path,
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=0,
        )
        self.rank_zero_logger.info("Best model saved with validation loss: {:.3e}".format(self.best_val_loss))

    def log_losses(self, train_loss: float, val_loss: float) -> None:
        """Logs the training and validation losses."""
        self.low_fidelity_losses['train'].append(train_loss)
        self.low_fidelity_losses['val'].append(val_loss)

def load_model(model_path: str, device: torch.device, model_type: str) -> MeshGraphNet:
    """Loads a trained model from a specified path.

    Args:
        model_path (str): Path to the saved model file.
        device (torch.device): Device to load the model onto.
        model_type (str): Type of the model to load ('MeshGraphNet' or 'GNN').

    Returns:
        The loaded model, either MeshGraphNet or GNN.
    """
    if model_type == 'MeshGraphNet':
        model = MeshGraphNet(
            C.input_dim_nodes, C.input_dim_edges, C.output_dim, aggregation=C.aggregation, processor_size= C.processor_size,
            hidden_dim_node_encoder=C.hidden_dim_node_encoder, hidden_dim_edge_encoder=C.hidden_dim_edge_encoder,
            hidden_dim_node_decoder=C.hidden_dim_node_decoder, hidden_dim_processor = C.hidden_dim_processor
        ).to(device)
        model.load(model_path)
    elif model_type == 'GNN':
        input_dim = 10
        hidden_dim = 64
        output_dim = 1
        num_gnn_layers = 3
        use_residual = True
        model = GNN(input_dim, hidden_dim, output_dim, num_gnn_layers, residual=use_residual).to(device)
        model.load_state_dict(torch.load(model_path))
    else:
        raise ValueError("Unsupported model type. Choose 'MeshGraphNet' or 'GNN'.")
    model.eval()
    return model

def update_data(high_fidelity_graphs: List[dgl.DGLGraph], low_fidelity_graphs: List[dgl.DGLGraph], low_model: MeshGraphNet, device: torch.device, method: str = 'kn', k_neighbors: int = 5, epsilon: float = 5000.0) -> List[dgl.DGLGraph]:
    """Updates high-fidelity graphs with predictions from the low-fidelity model.

    Args:
        high_fidelity_graphs (list): List of high-fidelity graphs.
        low_fidelity_graphs (list): List of low-fidelity graphs.
        low_model (MeshGraphNet): Trained low-fidelity model.
        device (torch.device): Device to perform computations on.
        method (str): Interpolation method ('kn' for k-nearest neighbors, 'rbf' for radial basis function).
        k_neighbors (int): Number of neighbors for k-nearest neighbors interpolation.
        epsilon (float): Hyperparameter for radial basis function interpolation.

    Returns:
        list: List of updated high-fidelity graphs.
    """
    updated_graphs = []
    low_model.eval()
    total_time = 0
    with torch.no_grad():
        for high_g, low_g in zip(high_fidelity_graphs, low_fidelity_graphs):
            start_time = time.time()
            low_g, high_g = low_g.to(device), high_g.to(device)
            low_nodes, high_nodes = low_g.ndata['x'][:, :2], high_g.ndata['x'][:, :2]
            low_output = low_model(low_g.ndata['x'], low_g.edata['x'], low_g).detach()
            distances = torch.cdist(high_nodes, low_nodes)

            if method == 'kn':
                k_nearest_distances, k_nearest_indices = distances.topk(k_neighbors, largest=False)
                weights = 1 / (k_nearest_distances + 1e-8)
                normalized_weights = weights / torch.sum(weights, dim=1, keepdim=True)
                interpolated_output = torch.sum(low_output[k_nearest_indices] * normalized_weights.unsqueeze(-1), dim=1)
            elif method == 'rbf':
                rbf_weights = torch.exp(-epsilon * distances ** 2)
                normalized_weights = rbf_weights / torch.sum(rbf_weights, dim=1, keepdim=True)
                interpolated_output = torch.mm(normalized_weights, low_output)

            high_g.ndata['x'] = torch.cat([high_g.ndata['x'], interpolated_output], dim=1)
            updated_graphs.append(high_g)
            end_time = time.time()
            total_time += end_time - start_time

    average_time = total_time / len(high_fidelity_graphs)
    print(f'Average time for updating data: {average_time:.4f} seconds')
    return updated_graphs

class GNN(nn.Module):
    """Graph Neural Network (GNN) model.

    Attributes:
        encoder1 (nn.Linear): First encoder layer.
        encoder2 (nn.Linear): Second encoder layer.
        gnn_layers (nn.ModuleList): List of GNN layers.
        decoder (nn.Sequential): Decoder layers.
        residual (bool): Whether to use residual connections.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_gnn_layers: int, residual: bool = False) -> None:
        super(GNN, self).__init__()
        self.encoder1 = nn.Linear(input_dim, hidden_dim)
        self.encoder2 = nn.Linear(hidden_dim, hidden_dim)
        self.gnn_layers = nn.ModuleList([
            EdgeConv(in_feat=hidden_dim, out_feat=hidden_dim, batch_norm=False, allow_zero_in_degree=True)
            for _ in range(num_gnn_layers)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.residual = residual

    def forward(self, g: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GNN model.

        Args:
            g (dgl.DGLGraph): Input graph.
            features (torch.Tensor): Node features.

        Returns:
            torch.Tensor: Model predictions.
        """
        x_residual = 0
        if self.residual:
            x_residual = features[:, -1].view(-1, 1)

        x_encoded = F.relu(self.encoder1(features))
        x = F.relu(self.encoder2(x_encoded))
        for gnn_layer in self.gnn_layers:
            x = F.relu(gnn_layer(g, x))
        x_decoded = self.decoder(x)
        return x_decoded + x_residual

def make_prediction(model: MeshGraphNet, graph: dgl.DGLGraph, device: torch.device) -> torch.Tensor:
    """Makes a prediction using a trained MeshGraphNet model and denormalizes it.

    Args:
        model (MeshGraphNet): The trained MeshGraphNet model.
        graph (dgl.DGLGraph): The graph data for prediction.
        device (torch.device): The device to perform the prediction on.

    Returns:
        torch.Tensor: The denormalized prediction tensor.
    """
    stat_path = os.path.join(C.ckpt_path, C.ckpt_name_hf, 'normalization_params_LF.json')
    with open(stat_path, 'r') as f:
        normalization_params = json.load(f)

    label_min = torch.tensor(normalization_params['label_min'])
    label_max = torch.tensor(normalization_params['label_max'])

    graph = graph.to(device)
    with torch.no_grad():
        prediction = model(graph.ndata['x'], graph.edata['x'], graph)

    return prediction * (label_max - label_min) + label_min

def plot_mesh(ax: plt.Axes, element_nodes: np.ndarray, nodes: np.ndarray, values: np.ndarray, cmap: str, title: str, xlabel: str, ylabel: str, cbar_label: str) -> None:
    """Plots a mesh with the given parameters.

    Args:
        ax: The matplotlib axis to plot on.
        element_nodes: Element nodes data.
        nodes: Nodes data.
        values: Values to color the nodes with.
        cmap: Colormap for node coloring.
        title: Title of the plot.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        cbar_label: Label for the colorbar.
    """
    values_array = np.array(values)

    for element in element_nodes:
        element = element - 1
        vertices = nodes[element, :2]
        polygon = Polygon(vertices, edgecolor='k', lw=0.8, fill=False)
        ax.add_patch(polygon)

    vmin = 2771346
    vmax = 32435959

    scatter = ax.scatter(nodes[:, 0], nodes[:, 1], c=values_array, cmap=cmap, s=10)

    ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max())
    ax.set_ylim(nodes[:, 1].min(), nodes[:, 1].max())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', location='bottom')
    cbar.set_label(cbar_label)
    ax.set_aspect('equal')
    ax.set_title(title)

def compute_metrics_on_test_set(model: GNN, test_data: List[dgl.DGLGraph], device: torch.device) -> Tuple[float, float, float]:
    """Computes the Mean Relative Error (MRE) and Mean Squared Error (MSE) on the test set.

    Args:
        model (MeshGraphNet): The trained MeshGraphNet model.
        test_data (list): The test dataset.
        device (torch.device): The device to perform computations on.

    Returns:
        tuple: A tuple containing the Mean Relative Error, Mean Squared Error, and Relative L2 Norm Error.
    """
    stat_path = os.path.join(C.ckpt_path, C.ckpt_name_hf, 'normalization_params_HF.json')
    with open(stat_path, 'r') as f:
        normalization_params = json.load(f)

    label_min = torch.tensor(normalization_params['label_min'])
    label_max = torch.tensor(normalization_params['label_max'])

    total_mse = 0.0
    total_mre = 0.0
    total_relative_l2_error = 0.0
    total_samples = 0
    total_test_time = 0.0

    for sample_data in test_data:
        start_time = time.time()
        graph = sample_data.to(device)

        with torch.no_grad():
            prediction = model(graph, graph.ndata['x'])
        target = graph.ndata['y']

        end_time = time.time()
        total_test_time += (end_time - start_time)

        prediction = prediction * (label_max - label_min) + label_min
        target = target * (label_max - label_min) + label_min

        mse = mse_loss(prediction, target).item()
        total_mse += mse

        relative_error = torch.abs(prediction - target) / torch.clamp(target, min=1e-6)
        mre = torch.mean(relative_error).item()
        total_mre += mre

        l2_error = torch.norm(prediction - target, p=2)
        target_l2_norm = torch.norm(target, p=2)
        relative_l2_error = l2_error / torch.clamp(target_l2_norm, min=1e-6)
        total_relative_l2_error += relative_l2_error.item()

        total_samples += 1

    mean_mse = total_mse / total_samples
    mean_mre = total_mre / total_samples
    mean_relative_l2_error = total_relative_l2_error / total_samples

    average_test_time = total_test_time / total_samples
    print(f"Average test time per sample: {average_test_time:.4f} seconds")

    return mean_mre, mean_mse, mean_relative_l2_error

def post_training_analysis_and_plotting() -> None:
    """Conducts post-training analysis and generates plots for test data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_data_path_HF = os.path.join(C.data_dir, 'MF_test_data_HF.pkl')
    with open(test_data_path_HF, 'rb') as f:
        test_data_HF = pickle.load(f)

    test_data_path_LF = os.path.join(C.data_dir, 'MF_test_data_LF.pkl')
    with open(test_data_path_LF, 'rb') as f:
        test_data_LF = pickle.load(f)

    model_checkpoint_path = os.path.join(C.ckpt_path, C.ckpt_name_lf, "best_model_LF", "MeshGraphNet.0.0.mdlus")
    low_model = load_model(model_checkpoint_path, device, 'MeshGraphNet')

    model_checkpoint_path = os.path.join(C.ckpt_path, C.ckpt_name_hf, "best_model_HF", "GNN.0.0.pt")
    high_model = load_model(model_checkpoint_path, device, 'GNN')

    num_trainable_params = sum(p.numel() for p in low_model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters for LF Model: {num_trainable_params}')

    num_trainable_params = sum(p.numel() for p in high_model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters for HF Model: {num_trainable_params}')

    processed_test_data_HF = preprocess_test_data(test_data_HF, 'HF')
    processed_test_data_LF = preprocess_test_data(test_data_LF, 'LF')

    processed_test_data = update_data(processed_test_data_HF, processed_test_data_LF, low_model, device, k_neighbors=5)

    mean_mre, mean_mse, mean_relative_l2_error = compute_metrics_on_test_set(high_model, processed_test_data, device)
    print(f"Mean Relative Error on Test Set: {mean_mre:.2e}")
    print(f"Mean Squared Error on Test Set: {mean_mse:.2e}")
    print(f"Mean Relative L2 Error on Test Set: {mean_relative_l2_error:.2e}")

    sample_index = 100
    sample_data_HF = test_data_HF[sample_index]
    sample_data_LF = test_data_LF[sample_index]
    geometry, mesh, input_features, target_labels = sample_data_HF
    von_mises_stress = target_labels['von_mises_stress']
    nodes = mesh['nodes']
    element_nodes = mesh['element_nodes']

    processed_test_data_HF = preprocess_test_data([sample_data_HF], 'HF')
    processed_test_data_LF = preprocess_test_data([sample_data_LF], 'LF')

    processed_test_data = update_data(processed_test_data_HF, processed_test_data_LF, low_model, device, k_neighbors=5)
    sample_graph = processed_test_data[0]

    stat_path = os.path.join(C.ckpt_path, C.ckpt_name_hf, 'normalization_params_HF.json')
    with open(stat_path, 'r') as f:
        normalization_params = json.load(f)

    label_min = torch.tensor(normalization_params['label_min'])
    label_max = torch.tensor(normalization_params['label_max'])

    with torch.no_grad():
        sample_graph.to(device)
        prediction = high_model(sample_graph, sample_graph.ndata['x'])
        prediction = prediction * (label_max - label_min) + label_min

    prediction = prediction.cpu().numpy().flatten()
    fig_gt, ax_gt = plt.subplots(figsize=(13, 8))
    plot_mesh(ax_gt, element_nodes, nodes, von_mises_stress, 'jet', '', 'X', 'Y', 'von Mises Stress')
    plt.tight_layout()
    plt.savefig(os.path.join(C.ckpt_path, C.ckpt_name_hf, 'Ground_Truth.png'), dpi=300, bbox_inches='tight')
    plt.show()

    fig_pred, ax_pred = plt.subplots(figsize=(13, 8))
    plot_mesh(ax_pred, element_nodes, nodes, prediction, 'jet', 'Prediction', 'X', 'Y', 'von Mises Stress')
    plt.tight_layout()
    plt.savefig(os.path.join(C.ckpt_path, C.ckpt_name_hf, 'Prediction.png'), dpi=300, bbox_inches='tight')
    plt.show()

    error = np.abs(prediction - von_mises_stress) / np.maximum(von_mises_stress, 1e-8)
    fig_error, ax_error = plt.subplots(figsize=(13, 8))
    plot_mesh(ax_error, element_nodes, nodes, error, 'jet', 'Error', 'X', 'Y', 'Relative Error')
    plt.tight_layout()
    plt.savefig(os.path.join(C.ckpt_path, C.ckpt_name_hf, 'Error.png'), dpi=300, bbox_inches='tight')
    plt.show()

    mean_relative_error = np.mean(error)
    print(f"Mean Relative Error: {mean_relative_error:.2e}")

def train_and_validate(model: MeshGraphNet, train_loader: GraphDataLoader, val_loader: GraphDataLoader, optimizer: Optimizer, scheduler: StepLR, rank_zero_logger: PythonLogger, epochs: int = C.epochs_HF) -> Dict[str, List[float]]:
    """Trains and validates the high-fidelity model.

    Args:
        model (MeshGraphNet): The high-fidelity model.
        train_loader (GraphDataLoader): DataLoader for the training data.
        val_loader (GraphDataLoader): DataLoader for the validation data.
        optimizer (Optimizer): Optimizer for training the model.
        scheduler (StepLR): Learning rate scheduler.
        rank_zero_logger (PythonLogger): Logger for rank zero processes.
        epochs (int): Number of epochs to train the model.

    Returns:
        dict: Dictionary containing training and validation losses.
    """
    stat_path = os.path.join(C.ckpt_path, C.ckpt_name_hf, 'normalization_params_HF.json')
    with open(stat_path, 'r') as f:
        normalization_params = json.load(f)
    label_min = torch.tensor(normalization_params['label_min'])
    label_max = torch.tensor(normalization_params['label_max'])

    best_val_loss = float('inf')
    high_fidelity_losses = {'train': [], 'val': []}
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        for batched_graph in train_loader:
            features = batched_graph.ndata['x']
            labels = batched_graph.ndata['y'].to(features.device)
            preds = model(batched_graph, features)
            diff_norm = torch.norm(
                torch.flatten(preds) - torch.flatten(labels), p=2
            )
            y_norm = torch.norm(torch.flatten(labels), p=2)
            loss = diff_norm / y_norm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batched_graph.batch_num_nodes())
        end_time = time.time()
        total_loss /= len(train_loader.dataset)
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batched_graph in val_loader:
                features = batched_graph.ndata['x']
                labels = batched_graph.ndata['y'].to(features.device)
                labels = labels * (label_max - label_min) + label_min
                preds = model(batched_graph, features)
                preds = preds * (label_max - label_min) + label_min
                diff_norm = torch.norm(
                    torch.flatten(preds) - torch.flatten(labels), p=2
                )
                y_norm = torch.norm(torch.flatten(labels), p=2)
                loss = diff_norm / y_norm
                total_val_loss += loss.item() * len(batched_graph.batch_num_nodes())
        total_val_loss /= len(val_loader.dataset)
        rank_zero_logger.info(f'Epoch {epoch} | Train Loss: {total_loss:.2e} | Val Loss: {total_val_loss:.2e}| Time: {end_time - start_time:.2f}s')
        scheduler.step()
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            save_checkpoint(os.path.join(C.ckpt_path, C.ckpt_name_hf, 'best_model_HF'), model, optimizer, scheduler,
                            None, 0)
            rank_zero_logger.info(f"New best high-fidelity model saved with val_loss: {total_val_loss:.2e}")

        high_fidelity_losses['train'].append(total_loss)
        high_fidelity_losses['val'].append(total_val_loss)
        rank_zero_logger.info(f"Epoch {epoch} | Training Loss: {total_loss:.2e} | Validation Loss: {total_val_loss:.2e}")

    return high_fidelity_losses

def main() -> None:
    """Main function to initialize distributed training, prepare directories, setup logging, and train the model."""
    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.rank == 0:
        os.makedirs(C.ckpt_path, exist_ok=True)
        with open(os.path.join(C.ckpt_path, C.ckpt_name_hf + ".json"), "w") as json_file:
            json_file.write(C.model_dump_json(indent=4))
    initialize_wandb(project="Aero", entity="Modulus", name="Aero-Training", group="Aero-DDP-Group", mode=C.wandb_mode)
    logger, rank_zero_logger = PythonLogger("main"), RankZeroLoggingWrapper(PythonLogger("main"), dist)
    trainer_LF = MGNTrainer(wb, dist, rank_zero_logger)

    for epoch in range(trainer_LF.epoch_init, C.epochs_LF):
        start_time = time.time()
        train_loss = trainer_LF.train_epoch()
        end_time = time.time()
        val_loss = trainer_LF.validation()
        trainer_LF.log_losses(train_loss, val_loss)
        if dist.rank == 0:
            wb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            rank_zero_logger.info(
                f"Epoch {epoch} | Train Loss: {train_loss:.2e} | Val Loss: {val_loss:.2e} | LR: {trainer_LF.get_lr():.3e}| Time: {end_time - start_time:.2f}s")
            save_checkpoint(os.path.join(C.ckpt_path, C.ckpt_name_lf), trainer_LF.model, trainer_LF.optimizer, trainer_LF.scheduler,
                            trainer_LF.scaler, epoch)

    model_checkpoint_path = os.path.join(C.ckpt_path, C.ckpt_name_lf, "best_model_LF", "MeshGraphNet.0.0.mdlus")
    low_model = load_model(model_checkpoint_path, dist.device, 'MeshGraphNet')

    with open(f'{C.data_dir}/MF_train_data_LF.pkl', 'rb') as f:
        low_fidelity_data = pickle.load(f)

    with open(f'{C.data_dir}/MF_train_data_HF.pkl', 'rb') as f:
        high_fidelity_data = pickle.load(f)

    indices = range(C.N_HF)
    sampled_low_fidelity_data = [low_fidelity_data[i] for i in indices]
    sampled_high_fidelity_data = [high_fidelity_data[i] for i in indices]

    low_fidelity_data = preprocess_test_data(sampled_low_fidelity_data, 'LF')
    high_fidelity_data, (feature_min, feature_max), (label_min, label_max) = preprocess_data(sampled_high_fidelity_data)

    normalization_params = {
        'label_min': label_min.item(),
        'label_max': label_max.item(),
        'feature_min': feature_min.tolist(),
        'feature_max': feature_max.tolist()
    }

    stat_path = os.path.join(C.ckpt_path, C.ckpt_name_hf, 'normalization_params_HF.json')
    with open(stat_path, 'w') as f:
        json.dump(normalization_params, f)

    updated_high_fidelity_data = update_data(high_fidelity_data, low_fidelity_data, low_model, dist.device)
    updated_train_data, updated_val_data = train_test_split(updated_high_fidelity_data, test_size=0.1, random_state=C.random_seed)
    train_loader = GraphDataLoader(updated_train_data, batch_size=64, shuffle=True)
    val_loader = GraphDataLoader(updated_val_data, batch_size=64, shuffle=False)

    input_dim = 10
    hidden_dim = 64
    output_dim = 1
    num_gnn_layers = 3
    use_residual = True

    hf_model = GNN(input_dim, hidden_dim, output_dim, num_gnn_layers, residual=use_residual).to(dist.device)
    optimizer = torch.optim.Adam(hf_model.parameters(), lr=0.0005, weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99985)
    hf_losses = train_and_validate(hf_model, train_loader, val_loader, optimizer, scheduler, rank_zero_logger)

    model_checkpoint_path = os.path.join(C.ckpt_path, C.ckpt_name_hf, "best_model_HF", "GNN.0.0.pt")
    hf_model = load_model(model_checkpoint_path, dist.device, 'GNN')

    with open(os.path.join(C.ckpt_path, 'low_fidelity_losses.pkl'), 'wb') as lf_f:
        pickle.dump(trainer_LF.low_fidelity_losses, lf_f)
    with open(os.path.join(C.ckpt_path, 'high_fidelity_losses.pkl'), 'wb') as hf_f:
        pickle.dump(hf_losses, hf_f)

if __name__ == "__main__":
    main()
    post_training_analysis_and_plotting()
