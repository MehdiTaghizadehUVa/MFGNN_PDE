import os
from pathlib import Path
from pydantic import BaseModel
from typing import Tuple, Optional

class Constants(BaseModel):
    ckpt_path: str = "./checkpoints_exp03"
    ckpt_name: str = "./hf_models_2000"
    data_dir: str = "dataset_exp03"
    results_dir: str = "./results_hf"

    input_dim_nodes: int = 9
    input_dim_edges: int = 3
    output_dim: int = 1
    aggregation: int = "sum"
    processor_size: int = 6
    hidden_dim_processor: int = 64
    hidden_dim_node_encoder: int = 64
    hidden_dim_edge_encoder: int = 64
    hidden_dim_node_decoder: int = 64

    batch_size: int = 32
    epochs: int = 500
    N_HF: int = 2000

    lr: float = 0.0005
    lr_decay_rate: float = 0.99985

    weight_decay: float = 0.0001

    amp: bool = False
    jit: bool = False

    wandb_mode: str = "disabled"