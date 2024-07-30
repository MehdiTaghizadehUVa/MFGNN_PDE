import os
from pathlib import Path
from pydantic import BaseModel
from typing import Tuple, Optional

class Constants(BaseModel):
    ckpt_path: str = "./checkpoints_exp03_time"
    ckpt_name: str = "./hf_models_2000"
    data_dir: str = "dataset_exp03"
    results_dir: str = "./results"

    input_dim_nodes: int = 9
    input_dim_edges: int = 3
    output_dim: int = 1
    aggregation: int = "sum"
    processor_size: int = 6
    hidden_dim_processor: int = 64
    hidden_dim_node_encoder: int = 64
    hidden_dim_edge_encoder: int = 64
    hidden_dim_node_decoder: int = 64

    batch_size_lf: int = 64
    batch_size_hf: int = 64
    epochs_lf: int = 5
    epochs_hf: int = 5
    N_LF: int = 5000
    N_HF: int = 2000

    lr_lf: float = 0.001
    lr_hf: float = 0.0005

    weight_decay_lf: float = 0.0001
    weight_decay_hf: float = 0.0001

    amp: bool = False
    jit: bool = False

    wandb_mode: str = "disabled"
