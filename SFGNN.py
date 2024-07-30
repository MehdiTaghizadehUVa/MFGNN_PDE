import os
import time
import random
import pickle
import warnings
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from sklearn.model_selection import train_test_split
from dgl import graph as dgl_graph
from dgl.dataloading import GraphDataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.nn.functional import mse_loss
from typing import List, Tuple, Dict
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import StepLR

from constants_SFGNN import Constants
from modulus.models.meshgraphnet import MeshGraphNet
from modulus.distributed.manager import DistributedManager
from modulus.launch.logging import initialize_wandb, PythonLogger, RankZeroLoggingWrapper
from modulus.launch.utils import load_checkpoint, save_checkpoint
import json

# Optional imports
try:
    import apex
except ImportError:
    apex = None

try:
    import wandb as wb
except ImportError:
    wb = None

# Suppress specific warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

# Instantiate constants
C = Constants()

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)  # For all GPUs
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def preprocess_data(training_data: List[tuple]) -> Tuple[List[dgl.DGLGraph], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
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

def preprocess_test_data(test_data: List[tuple]) -> List[dgl.DGLGraph]:
    """Preprocesses the test data by creating graphs and normalizing features and labels using min-max normalization.

    Args:
        test_data (list): A list containing test data tuples.

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
    """Trainer class for MeshGraphNet model.

    Attributes:
        best_val_loss (float): Best validation loss.
        dist (DistributedManager): Distributed manager for handling multi-GPU training.
        wb (object): Weights and Biases object for logging.
        rank_zero_logger (RankZeroLoggingWrapper): Logger for rank zero processes.
    """

    def __init__(self, wb: object, dist: DistributedManager, rank_zero_logger: RankZeroLoggingWrapper):
        self.best_val_loss = float('inf')
        self.dist = dist
        self.wb = wb
        self.rank_zero_logger = rank_zero_logger

        self.rank_zero_logger.info("Loading the training dataset...")
        dataset_path = C.data_dir
        training_data_path = os.path.join(dataset_path, 'MF_train_data_HF.pkl')
        with open(training_data_path, 'rb') as f:
            training_data = pickle.load(f)

        if len(training_data) >= C.N_HF:
            training_data_HF = training_data[:C.N_HF]
        else:
            raise ValueError(f"Dataset contains less than {C.N_HF} samples.")

        high_fidelity_data, (feature_min, feature_max), (label_min, label_max) = preprocess_data(training_data_HF)
        dataset, validation_dataset = train_test_split(high_fidelity_data, test_size=0.1, random_state=1)

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

        self.setup_dataloaders(dataset, validation_dataset)
        self.setup_model()
        self.setup_optimizer_and_scheduler()

        if dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            os.path.join(C.ckpt_path, C.ckpt_name),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=dist.device,
        )

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

    def setup_dataloaders(self, dataset: List[dgl.DGLGraph], validation_dataset: List[dgl.DGLGraph]) -> None:
        """Sets up the dataloaders for training and validation datasets.

        Args:
            dataset (list): Training dataset.
            validation_dataset (list): Validation dataset.
        """
        self.dataloader = GraphDataLoader(
            dataset, batch_size=C.batch_size, shuffle=True, drop_last=False, pin_memory=True,
            use_ddp=self.dist.world_size > 1
        )
        self.validation_dataloader = GraphDataLoader(
            validation_dataset, batch_size=C.batch_size, shuffle=False, drop_last=False, pin_memory=True, use_ddp=False
        )

    def setup_model(self) -> None:
        """Sets up the MeshGraphNet model."""
        self.model = MeshGraphNet(
            C.input_dim_nodes, C.input_dim_edges, C.output_dim, aggregation=C.aggregation, processor_size= C.processor_size,
            hidden_dim_node_encoder=C.hidden_dim_node_encoder, hidden_dim_edge_encoder=C.hidden_dim_edge_encoder,
            hidden_dim_node_decoder=C.hidden_dim_node_decoder, hidden_dim_processor = C.hidden_dim_processor
        ).to(self.dist.device)

        if C.jit:
            self.model = torch.jit.script(self.model)

        if self.dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.dist.local_rank], output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers, find_unused_parameters=self.dist.find_unused_parameters
            )

        self.model.train()

    def setup_optimizer_and_scheduler(self) -> None:
        """Sets up the optimizer and learning rate scheduler."""
        if apex:
            self.optimizer = apex.optimizers.FusedAdam(self.model.parameters(), lr=C.lr)
            self.rank_zero_logger.info("Using FusedAdam optimizer")
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=C.lr, weight_decay=C.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: C.lr_decay_rate**(epoch/len(self.dataloader))
        )
        self.scaler = GradScaler()

    def train(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """Trains the model for one batch.

        Args:
            graph (dgl.DGLGraph): Input graph.

        Returns:
            torch.Tensor: Training loss.
        """
        self.optimizer.zero_grad()
        loss = self.forward(graph)
        self.backward(loss)
        self.scheduler.step()
        return loss

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            graph (dgl.DGLGraph): Input graph.

        Returns:
            torch.Tensor: Computed loss.
        """
        with autocast(enabled=C.amp):
            pred = self.model(graph.ndata["x"], graph.edata["x"], graph)
            diff_norm = torch.norm(
                torch.flatten(pred) - torch.flatten(graph.ndata["y"]), p=2
            )
            y_norm = torch.norm(torch.flatten(graph.ndata["y"]), p=2)
            loss = diff_norm / y_norm
            return loss

    def backward(self, loss: torch.Tensor) -> None:
        """Performs backpropagation and updates the model parameters.

        Args:
            loss (torch.Tensor): Computed loss.
        """
        if C.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        lr = self.get_lr()
        self.wb.log({"lr": lr}) if self.wb else None

    def get_lr(self) -> float:
        """Returns the current learning rate.

        Returns:
            float: Current learning rate.
        """
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

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
        best_model_path = os.path.join(C.ckpt_path, C.ckpt_name, "best_model")
        save_checkpoint(
            best_model_path,
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=self.epoch_init,
        )
        self.rank_zero_logger.info("Best model saved with validation loss: {:.3e}".format(self.best_val_loss))

def load_model(model_path: str, device: torch.device) -> MeshGraphNet:
    """Loads a trained MeshGraphNet model from a specified path.

    Args:
        model_path (str): Path to the saved model file.
        device (torch.device): Device to load the model onto.

    Returns:
        MeshGraphNet: The loaded MeshGraphNet model.
    """
    model = MeshGraphNet(
        C.input_dim_nodes, C.input_dim_edges, C.output_dim, aggregation=C.aggregation, processor_size=C.processor_size,
        hidden_dim_node_encoder=C.hidden_dim_node_encoder, hidden_dim_edge_encoder=C.hidden_dim_edge_encoder,
        hidden_dim_node_decoder=C.hidden_dim_node_decoder, hidden_dim_processor=C.hidden_dim_processor
    ).to(device)

    model.eval()
    model.load(model_path)

    return model

def make_prediction(model: MeshGraphNet, graph: dgl.DGLGraph, device: torch.device) -> torch.Tensor:
    """Makes a prediction using a trained MeshGraphNet model and denormalizes it.

    Args:
        model (MeshGraphNet): The trained MeshGraphNet model.
        graph (dgl.DGLGraph): The graph data for prediction.
        device (torch.device): The device to perform the prediction on.

    Returns:
        torch.Tensor: The denormalized prediction tensor.
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
    """Plots a mesh with the given parameters.

    Args:
        ax (plt.Axes): The matplotlib axis to plot on.
        element_nodes (np.ndarray): Element nodes data.
        nodes (np.ndarray): Nodes data.
        values (np.ndarray): Values to color the nodes with.
        cmap (str): Colormap for node coloring.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        cbar_label (str): Label for the colorbar.
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
    """Computes the Mean Relative Error (MRE), Mean Squared Error (MSE), and Mean Relative L2 Error on the test set.

    Args:
        model (MeshGraphNet): The trained MeshGraphNet model.
        test_data (list): The test dataset.
        device (torch.device): The device to perform computations on.

    Returns:
        tuple: A tuple containing the Mean Relative Error, Mean Squared Error, and Relative L2 Norm Error.
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
    """Performs post-training analysis and generates plots for test data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_data_path = os.path.join(C.data_dir, 'MF_test_data_HF.pkl')
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)

    model_checkpoint_path = os.path.join(C.ckpt_path, C.ckpt_name, "best_model", "MeshGraphNet.0.0.mdlus")
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
    """Main function to initialize distributed training, prepare directories, setup logging, and train the model."""
    DistributedManager.initialize()
    dist = DistributedManager()

    if dist.rank == 0:
        os.makedirs(C.ckpt_path, exist_ok=True)
        with open(os.path.join(C.ckpt_path, C.ckpt_name + ".json"), "w") as json_file:
            json_file.write(C.model_dump_json(indent=4))

    if wb:
        initialize_wandb(
            project="Aero",
            entity="Modulus",
            name="Aero-Training",
            group="Aero-DDP-Group",
            mode=C.wandb_mode,
        )
    logger = PythonLogger("main")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging()

    trainer = MGNTrainer(wb, dist, rank_zero_logger)
    total_start_time = time.time()
    rank_zero_logger.info("Training started...")

    training_losses = []
    validation_losses = []

    for epoch in range(trainer.epoch_init, C.epochs):
        loss_agg = 0
        epoch_start_time = time.time()
        for graph in trainer.dataloader:
            graph = graph.to(dist.device)
            loss = trainer.train(graph)
            loss_agg += loss.detach().cpu().numpy()
        loss_agg /= len(trainer.dataloader)

        epoch_time = time.time() - epoch_start_time
        rank_zero_logger.info(
            f"Epoch: {epoch}, Train Loss: {loss_agg:.3e}, LR: {trainer.get_lr():.3e}, Time/Epoch: {epoch_time:.3e} sec")
        wb.log({"train_loss": loss_agg}) if wb else None

        if dist.rank == 0:
            val_loss_agg = trainer.validation()
            validation_losses.append(val_loss_agg)
            wb.log({"val_loss": val_loss_agg}) if wb else None
            rank_zero_logger.info(f"Epoch: {epoch}, Validation Loss: {val_loss_agg:.3e}")

        if dist.rank == 0:
            with open(os.path.join(C.ckpt_path, "losses.pkl"), "wb") as f:
                pickle.dump({"training_losses": training_losses, "validation_losses": validation_losses}, f)
            save_checkpoint(
                os.path.join(C.ckpt_path, C.ckpt_name),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )
            rank_zero_logger.info(f"Checkpoint saved after epoch {epoch}.")

    total_training_time = time.time() - total_start_time
    rank_zero_logger.info(f"Training completed in {total_training_time:.2f} seconds!")

if __name__ == "__main__":
    post_training_analysis_and_plotting()
