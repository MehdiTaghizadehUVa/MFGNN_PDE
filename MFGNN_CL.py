import os
import time
import random
import pickle
import torch
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import wandb as wb
import dgl
from dgl.dataloading import GraphDataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.nn.functional import mse_loss
from sklearn.model_selection import train_test_split
from dgl import graph as dgl_graph
from typing import List, Tuple, Dict
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import StepLR

from constants_MFGNN_CL import Constants
from modulus.models.meshgraphnet import MeshGraphNet
from modulus.distributed.manager import DistributedManager
from modulus.launch.logging import PythonLogger, initialize_wandb, RankZeroLoggingWrapper
from modulus.launch.utils import load_checkpoint, save_checkpoint

# Optional import for apex
try:
    import apex
except ImportError:
    apex = None

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Instantiate constants
C = Constants()

random.seed(10)
np.random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)  # For all GPUs
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

fidelity_params = {
    "LF": {"lr": C.lr_lf, "weight_decay": C.weight_decay_lf, "epochs": C.epochs_lf, "batch_size": C.batch_size_lf},
    "HF": {"lr": C.lr_hf, "weight_decay": C.weight_decay_hf, "epochs": C.epochs_hf, "batch_size": C.batch_size_hf}
}

def preprocess_data(training_data: List[tuple]) -> Tuple[List[dgl.DGLGraph], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Preprocess the training data to create DGL graphs and normalize the features and labels.

    Args:
        training_data (List[tuple]): List of training data tuples containing geometry, mesh, input features, and target labels.

    Returns:
        Tuple[List[dgl.DGLGraph], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        - List of DGL graphs.
        - Tuple of tensors representing feature min and max values.
        - Tuple of tensors representing label min and max values.
    """
    graphs = []
    all_labels = []
    all_features = []

    for geometry, mesh, input_features, target_labels in training_data:
        element_nodes = mesh['element_nodes'] - 1
        nodes = torch.tensor(mesh['nodes'], dtype=torch.float)
        node_coordinates = nodes.clone()

        edges = [(elements[i], elements[(i + 1) % len(elements)]) for elements in element_nodes for i in range(len(elements))]
        edges = np.array(edges).T
        src, dst = edges[0], edges[1]
        g = dgl.graph((src, dst))

        rel_distances = nodes[dst] - nodes[src]
        rel_distances_norm = torch.norm(rel_distances, dim=1, keepdim=True)
        edge_features = torch.cat([rel_distances, rel_distances_norm], dim=1)

        boundary_conditions = torch.tensor(input_features['boundary_conditions'], dtype=torch.float)
        node_features = torch.cat((node_coordinates, boundary_conditions), dim=1)

        target = torch.tensor(target_labels['von_mises_stress'], dtype=torch.float)

        all_labels.append(target)
        all_features.append(node_features)

        g.ndata['x'] = node_features
        g.edata['x'] = edge_features
        g.ndata['y'] = target.unsqueeze(1)
        graphs.append(g)

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    feature_min = all_features.min(dim=0)[0]
    feature_max = all_features.max(dim=0)[0]
    label_min = all_labels.min()
    label_max = all_labels.max()

    for g in graphs:
        g.ndata['x'] = (g.ndata['x'] - feature_min) / (feature_max - feature_min)
        g.ndata['y'] = (g.ndata['y'] - label_min) / (label_max - label_min)

    return graphs, (feature_min, feature_max), (label_min, label_max)

def preprocess_test_data(test_data: List[tuple]) -> List[dgl.DGLGraph]:
    """
    Preprocess the test data to create DGL graphs and normalize the features and labels.

    Args:
        test_data (List[tuple]): List of test data tuples containing geometry, mesh, input features, and target labels.

    Returns:
        List[dgl.DGLGraph]: List of DGL graphs.
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

    stat_path = os.path.join(C.ckpt_path, C.ckpt_name, 'normalization_params.json')
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
    """
    Trainer class for MeshGraphNet model.

    Attributes:
        best_val_loss (float): Best validation loss.
        dist (DistributedManager): Distributed manager for handling multi-GPU training.
        wb (object): Weights and Biases object for logging.
        rank_zero_logger (PythonLogger): Logger for rank zero processes.
        dataset_path (str): Path to the dataset.
        N_data_LF (int): Number of low-fidelity data samples.
        N_data_HF (int): Number of high-fidelity data samples.
        switch_epoch (int): Epoch at which to switch from low-fidelity to multi-fidelity training.
        current_fidelity (str): Current fidelity level ('LF' or 'HF').
        normalization_params (dict): Normalization parameters for features and labels.
        graphs_LF (List[dgl.DGLGraph]): List of low-fidelity graphs.
        graphs_HF (List[dgl.DGLGraph]): List of high-fidelity graphs.
        dataloader (GraphDataLoader): Data loader for training data.
        validation_dataloader (GraphDataLoader): Data loader for validation data.
        model (MeshGraphNet): MeshGraphNet model.
        optimizer (Optimizer): Optimizer for training.
        scheduler (StepLR): Learning rate scheduler.
        scaler (GradScaler): Gradient scaler for mixed precision training.
    """

    def __init__(self, wb: object, dist: DistributedManager, rank_zero_logger: PythonLogger, dataset_path: str, N_data_LF: int = 2500, N_data_HF: int = 2500, switch_epoch: int = 10):
        self.best_val_loss = float('inf')
        self.dist = dist
        self.wb = wb
        self.rank_zero_logger = rank_zero_logger
        self.dataset_path = dataset_path
        self.N_data_LF = N_data_LF
        self.N_data_HF = N_data_HF
        self.switch_epoch = switch_epoch
        self.current_fidelity = 'LF'

        self.load_and_preprocess_data()
        self.initialize_model()
        self.initialize_optimizer_and_scheduler()

    def load_and_preprocess_data(self) -> None:
        """
        Load and preprocess training data, then split it into low-fidelity and high-fidelity datasets.
        """
        self.rank_zero_logger.info("Loading and preprocessing LF and HF datasets...")
        with open(f'{self.dataset_path}/MF_train_data_LF.pkl', 'rb') as f:
            training_data_LF = pickle.load(f)
        training_data_LF = training_data_LF[:self.N_data_LF]

        with open(f'{self.dataset_path}/MF_train_data_HF.pkl', 'rb') as f:
            training_data_HF = pickle.load(f)
        training_data_HF = training_data_HF[:self.N_data_HF]

        combined_training_data = training_data_LF + training_data_HF
        graphs, (feature_min, feature_max), (label_min, label_max) = preprocess_data(combined_training_data)

        self.normalization_params = {
            'label_min': label_min.item(),
            'label_max': label_max.item(),
            'feature_min': feature_min.tolist(),
            'feature_max': feature_max.tolist()
        }

        stat_path = os.path.join(C.ckpt_path, C.ckpt_name)
        if not Path(stat_path).is_dir():
            Path(stat_path).mkdir(parents=True, exist_ok=True)
        stat_path = os.path.join(C.ckpt_path, C.ckpt_name, 'normalization_params.json')
        with open(stat_path, 'w') as f:
            json.dump(self.normalization_params, f)

        self.graphs_LF = graphs[:len(training_data_LF)]
        self.graphs_HF = graphs[len(training_data_LF):]

        self.update_dataloader(self.graphs_LF)

    def denormalize(self, normalized_labels: torch.Tensor) -> torch.Tensor:
        """
        Denormalize the normalized labels.

        Args:
            normalized_labels (torch.Tensor): Normalized labels.

        Returns:
            torch.Tensor: Denormalized labels.
        """
        label_min = self.normalization_params['label_min']
        label_max = self.normalization_params['label_max']
        denormalized_labels = normalized_labels * (label_max - label_min) + label_min
        return denormalized_labels

    def update_dataloader(self, graphs: List[dgl.DGLGraph]) -> None:
        """
        Update the data loader with the given graphs.

        Args:
            graphs (List[dgl.DGLGraph]): List of graphs.
        """
        batch_size = fidelity_params['LF' if self.current_fidelity == 'LF' else 'HF']["batch_size"]
        dataset, validation_dataset = train_test_split(graphs, test_size=0.1, random_state=42)
        self.dataloader = GraphDataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.validation_dataloader = GraphDataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    def initialize_model(self) -> None:
        """
        Initialize the MeshGraphNet model.
        """
        self.model = MeshGraphNet(
            C.input_dim_nodes, C.input_dim_edges, C.output_dim, aggregation=C.aggregation, processor_size= C.processor_size,
            hidden_dim_node_encoder=C.hidden_dim_node_encoder, hidden_dim_edge_encoder=C.hidden_dim_edge_encoder,
            hidden_dim_node_decoder=C.hidden_dim_node_decoder, hidden_dim_processor = C.hidden_dim_processor
        ).to(self.dist.device)

        if C.jit:
            self.model = torch.jit.script(self.model)

        if self.dist.world_size > 1:
            self.model = DistributedDataParallel(self.model, device_ids=[self.dist.local_rank], output_device=self.dist.device)

    def initialize_optimizer_and_scheduler(self) -> None:
        """
        Initialize the optimizer and learning rate scheduler.
        """
        lr = fidelity_params[self.current_fidelity]["lr"]
        weight_decay = fidelity_params[self.current_fidelity]["weight_decay"]

        if apex:
            self.optimizer = apex.optimizers.FusedAdam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10 * len(self.dataloader), gamma=0.99)
        self.scaler = GradScaler()

    def train(self, graph: dgl.DGLGraph) -> float:
        """
        Perform a training step on the given graph.

        Args:
            graph (dgl.DGLGraph): Input graph.

        Returns:
            float: Training loss.
        """
        self.optimizer.zero_grad()

        with autocast(enabled=C.amp):
            pred = self.model(graph.ndata["x"], graph.edata["x"], graph)
            diff_norm = torch.norm(
                torch.flatten(pred) - torch.flatten(graph.ndata["y"]), p=2
            )
            y_norm = torch.norm(torch.flatten(graph.ndata["y"]), p=2)
            loss = diff_norm / y_norm

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.scheduler.step()

        return loss.item()

    def get_lr(self) -> float:
        """
        Get the current learning rate.

        Returns:
            float: Current learning rate.
        """
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    @torch.no_grad()
    def validation(self) -> float:
        """
        Perform validation on the validation dataset.

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
        """
        Save the best model checkpoint.
        """
        best_model_path = os.path.join(C.ckpt_path, C.ckpt_name, f"best_model_{self.current_fidelity}")
        save_checkpoint(
            best_model_path,
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=0,
        )
        self.rank_zero_logger.info("Best model saved with validation loss: {:.3e}".format(self.best_val_loss))

    def run_epoch(self, epoch: int) -> None:
        """
        Run one epoch of training.

        Args:
            epoch (int): Current epoch number.
        """
        if epoch == self.switch_epoch:
            self.current_fidelity = 'MF'
            self.best_val_loss = float('inf')
            combined_graphs = self.graphs_LF + self.graphs_HF
            self.update_dataloader(combined_graphs)
            self.rank_zero_logger.info("Switched to combined LF+HF training.")

        epoch_loss = 0
        epoch_start_time = time.time()
        for batch, graph in enumerate(self.dataloader):
            graph = graph.to(self.dist.device)

            loss = self.train(graph)
            epoch_loss += loss
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / len(self.dataloader)
        val_loss = self.validation()

        self.rank_zero_logger.info(f"Epoch: {epoch}, Train Loss: {avg_loss:.2e}, Val Loss: {val_loss:.2e}, Time/Epoch: {epoch_time:.3e} sec")
        if self.wb:
            self.wb.log({"epoch": epoch, f"{self.current_fidelity.lower()}_train_loss": avg_loss,
                         f"{self.current_fidelity.lower()}_val_loss": val_loss})

def prepare_directories_and_files(ckpt_path: str) -> None:
    """
    Prepare directories and files for saving checkpoints.

    Args:
        ckpt_path (str): Path to the checkpoint directory.
    """
    os.makedirs(ckpt_path, exist_ok=True)

def setup_logging_and_wandb(dist: DistributedManager) -> Tuple[PythonLogger, RankZeroLoggingWrapper]:
    """
    Set up logging and Weights and Biases.

    Args:
        dist (DistributedManager): Distributed manager for handling multi-GPU training.

    Returns:
        Tuple[PythonLogger, RankZeroLoggingWrapper]: Logger and rank zero logger.
    """
    logger = PythonLogger("main")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)

    if dist.rank == 0:
        initialize_wandb(project="Aero", entity="Modulus", name="Aero-Training", group="Aero-DDP-Group", mode=C.wandb_mode)

    return logger, rank_zero_logger

def load_model(model_path: str, device: torch.device) -> MeshGraphNet:
    """
    Load a pretrained MeshGraphNet model.

    Args:
        model_path (str): Path to the model checkpoint.
        device (torch.device): Device to load the model on.

    Returns:
        MeshGraphNet: Loaded model.
    """
    model = MeshGraphNet(
        C.input_dim_nodes, C.input_dim_edges, C.output_dim, aggregation=C.aggregation, processor_size= C.processor_size,
        hidden_dim_node_encoder=C.hidden_dim_node_encoder, hidden_dim_edge_encoder=C.hidden_dim_edge_encoder,
        hidden_dim_node_decoder=C.hidden_dim_node_decoder, hidden_dim_processor = C.hidden_dim_processor
    ).to(device)

    model.eval()
    model.load(model_path)

    return model

def make_prediction(model: MeshGraphNet, graph: dgl.DGLGraph, device: torch.device) -> torch.Tensor:
    """
    Make a prediction using the trained model.

    Args:
        model (MeshGraphNet): Trained model.
        graph (dgl.DGLGraph): Input graph.
        device (torch.device): Device to perform the prediction on.

    Returns:
        torch.Tensor: Model prediction.
    """
    stat_path = os.path.join(C.ckpt_path, C.ckpt_name, 'normalization_params.json')
    with open(stat_path, 'r') as f:
        normalization_params = json.load(f)

    label_min = torch.tensor(normalization_params['label_min'])
    label_max = torch.tensor(normalization_params['label_max'])

    graph = graph.to(device)
    with torch.no_grad():
        prediction = model(graph.ndata['x'], graph.edata['x'], graph)
    return prediction * (label_max - label_min) + label_min

def plot_mesh(ax: plt.Axes, element_nodes: np.ndarray, nodes: np.ndarray, values: np.ndarray, cmap: str, title: str, xlabel: str, ylabel: str, cbar_label: str) -> None:
    """
    Plot the mesh with the given values.

    Args:
        ax (plt.Axes): Matplotlib axis to plot on.
        element_nodes (np.ndarray): Array of element nodes.
        nodes (np.ndarray): Array of node coordinates.
        values (np.ndarray): Array of values to plot.
        cmap (str): Colormap for the plot.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        cbar_label (str): Label for the color bar.
    """
    for element in element_nodes:
        element = element - 1
        vertices = nodes[element, :2]
        polygon = Polygon(vertices, edgecolor='k', lw=0.5, fill=False)
        ax.add_patch(polygon)
    scatter = ax.scatter(nodes[:, 0], nodes[:, 1], c=values, cmap=cmap, s=8)

    ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max())
    ax.set_ylim(nodes[:, 1].min(), nodes[:, 1].max())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', location='bottom')
    cbar.set_label(cbar_label)
    ax.set_aspect('equal')
    ax.set_title(title)

def compute_metrics_on_test_set(model: MeshGraphNet, test_data: List[tuple], device: torch.device) -> Tuple[float, float, float]:
    """
    Compute metrics on the test set.

    Args:
        model (MeshGraphNet): Trained model.
        test_data (List[tuple]): List of test data tuples.
        device (torch.device): Device to perform computations on.

    Returns:
        Tuple[float, float, float]: Mean relative error, mean squared error, and mean relative L2 error.
    """
    stat_path = os.path.join(C.ckpt_path, C.ckpt_name, 'normalization_params.json')
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
        geometry, mesh, input_features, target_labels = sample_data
        processed_data = preprocess_test_data([sample_data])
        graph = processed_data[0].to(device)

        prediction = make_prediction(model, graph, device)
        end_time = time.time()
        total_test_time += (end_time - start_time)
        target = graph.ndata['y']

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
    """
    Perform post-training analysis and plot the results.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_data_path = os.path.join(C.data_dir, 'MF_test_data_HF.pkl')
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)

    model_checkpoint_path = os.path.join(C.ckpt_path, C.ckpt_name, "best_model_HF")
    model = load_model(model_checkpoint_path, device)

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {num_trainable_params}')

    mean_mre, mean_mse, mean_relative_l2_error = compute_metrics_on_test_set(model, test_data, device)
    print(f"Mean Relative Error on Test Set: {mean_mre:.2e}")
    print(f"Mean Squared Error on Test Set: {mean_mse:.2e}")
    print(f"Mean Relative L2 Error on Test Set: {mean_relative_l2_error:.2e}")

    sample_index = 25
    sample_data = test_data[sample_index]
    geometry, mesh, input_features, target_labels = sample_data
    von_mises_stress = target_labels['von_mises_stress']
    nodes = mesh['nodes']
    element_nodes = mesh['element_nodes']

    processed_test_data = preprocess_test_data([sample_data])
    sample_graph = processed_test_data[0]

    prediction = make_prediction(model, sample_graph, device).cpu().numpy().flatten()

    fig_gt, ax_gt = plt.subplots(figsize=(13, 8))
    plot_mesh(ax_gt, element_nodes, nodes, von_mises_stress, 'jet', 'Ground Truth', 'X', 'Y', 'von Mises Stress')
    plt.tight_layout()
    plt.savefig(os.path.join(C.ckpt_path, C.ckpt_name, 'Ground_Truth.png'), dpi=300, bbox_inches='tight')
    plt.show()

    fig_pred, ax_pred = plt.subplots(figsize=(13, 8))
    plot_mesh(ax_pred, element_nodes, nodes, prediction, 'jet', 'Prediction', 'X', 'Y', 'von Mises Stress')
    plt.tight_layout()
    plt.savefig(os.path.join(C.ckpt_path, C.ckpt_name, 'Prediction.png'), dpi=300, bbox_inches='tight')
    plt.show()

    error = np.abs(prediction - von_mises_stress) / np.maximum(von_mises_stress, 1e-8)
    fig_error, ax_error = plt.subplots(figsize=(13, 8))
    plot_mesh(ax_error, element_nodes, nodes, error, 'jet', 'Error', 'X', 'Y', 'Relative Error')
    plt.tight_layout()
    plt.savefig(os.path.join(C.ckpt_path, C.ckpt_name, 'Error.png'), dpi=300, bbox_inches='tight')
    plt.show()

    mean_relative_error = np.mean(error)
    print(f"Mean Relative Error: {mean_relative_error:.2e}")

def main() -> None:
    """
    Main function to initialize distributed training, prepare directories, setup logging, and train the model.
    """
    DistributedManager.initialize()
    dist = DistributedManager()

    prepare_directories_and_files(C.ckpt_path)
    logger, rank_zero_logger = setup_logging_and_wandb(dist)

    trainer = MGNTrainer(
        wb, dist, rank_zero_logger, C.data_dir, N_data_LF=C.N_LF, N_data_HF=C.N_HF, switch_epoch=C.switch_epoch
    )

    total_start_time = time.time()
    rank_zero_logger.info("Training started...")

    for epoch in range(C.epochs):
        trainer.run_epoch(epoch)
        if dist.rank == 0:
            save_checkpoint(
                os.path.join(C.ckpt_path, C.ckpt_name),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )

    total_training_time = time.time() - total_start_time
    rank_zero_logger.info(f"Training completed in {total_training_time:.2f} seconds!")

if __name__ == "__main__":
    main()
    post_training_analysis_and_plotting()
