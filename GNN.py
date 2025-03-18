#!/usr/bin/env python3
"""
Graph Neural Network for Structural Wear Prediction

This model uses a Graph Neural Network to predict wear at nodes in a structural mesh
based on material properties, geometry, and loading conditions.
"""

import os
import time
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_max_pool
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T
import networkx as nx
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("wear_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


class EdgeFeatureExtractor:
    """
    Extracts edge features based on node positions and element properties.
    """
    
    def __init__(self):
        pass
        
    def compute_edge_features(self, node_positions, elements, thickness, youngs_modulus):
        """
        Compute edge features for the graph.
        
        Args:
            node_positions: Tensor of node coordinates (n_nodes, 3)
            elements: List of element connectivity
            thickness: Tensor or float of thickness values
            youngs_modulus: Tensor or float of Young's modulus values
            
        Returns:
            edge_index: Tensor of shape (2, n_edges) containing edge connectivity
            edge_attr: Tensor of shape (n_edges, n_features) containing edge features
        """
        # Convert elements to edge indices
        edge_list = []
        
        try:
            # Handle different element formats (list or np.ndarray)
            if isinstance(elements, np.ndarray):
                elements = elements.tolist()
                
            # Create edges from elements
            for elem in elements:
                # Make sure each element is a list or similar iterable
                if not hasattr(elem, '__iter__'):
                    continue
                    
                # For each element, create edges between all pairs of nodes
                for i in range(len(elem)):
                    for j in range(i+1, len(elem)):
                        # Check that node indices are valid
                        if elem[i] < node_positions.shape[0] and elem[j] < node_positions.shape[0]:
                            edge_list.append([elem[i], elem[j]])
                            edge_list.append([elem[j], elem[i]])  # Add both directions
        except Exception as e:
            logger.error(f"Error creating edge list: {e}")
            # If there's an error, create a minimal edge list to prevent failure
            for i in range(node_positions.shape[0] - 1):
                edge_list.append([i, i+1])
                edge_list.append([i+1, i])
        
        # Remove duplicates and convert to tensor
        edge_list = np.array(edge_list)
        edge_index = torch.tensor(edge_list.T, dtype=torch.long)
        
        # Compute edge features
        start_nodes = edge_index[0]
        end_nodes = edge_index[1]
        
        # Distance between connected nodes
        start_pos = node_positions[start_nodes]
        end_pos = node_positions[end_nodes]
        distances = torch.norm(end_pos - start_pos, dim=1, keepdim=True)
        
        # If thickness and Young's modulus are provided per element, map them to edges
        if isinstance(thickness, torch.Tensor) and thickness.dim() > 0:
            # Find the element each edge belongs to
            edge_thickness = []
            for i in range(edge_index.shape[1]):
                edge_thickness.append(self._find_edge_property(edge_index[:, i], elements, thickness))
            edge_thickness = torch.tensor(edge_thickness, dtype=torch.float).view(-1, 1)
        else:
            # Use constant thickness for all edges
            edge_thickness = torch.ones(edge_index.shape[1], 1) * thickness
            
        if isinstance(youngs_modulus, torch.Tensor) and youngs_modulus.dim() > 0:
            edge_youngs = []
            for i in range(edge_index.shape[1]):
                edge_youngs.append(self._find_edge_property(edge_index[:, i], elements, youngs_modulus))
            edge_youngs = torch.tensor(edge_youngs, dtype=torch.float).view(-1, 1)
        else:
            edge_youngs = torch.ones(edge_index.shape[1], 1) * youngs_modulus
            
        # Combine features
        edge_attr = torch.cat([distances, edge_thickness, edge_youngs], dim=1)
        
        return edge_index, edge_attr
        
    def _find_edge_property(self, edge, elements, property_values):
        """
        Find the property value for an edge based on which element it belongs to.
        
        Args:
            edge: Edge indices [u, v]
            elements: List of element connectivity
            property_values: Tensor of property values per element
            
        Returns:
            float: Property value for the edge
        """
        try:
            # Find which element this edge belongs to
            for i, elem in enumerate(elements):
                if edge[0].item() in elem and edge[1].item() in elem:
                    if i < len(property_values):
                        return property_values[i].item()
            
            # If not found, return average
            return property_values.mean().item()
        except Exception as e:
            # In case of error, return a default value
            logger.warning(f"Error finding element properties: {e}")
            return property_values.mean().item() if property_values.numel() > 0 else 1.0


class MeshGraphDataset:
    """
    Creates a graph dataset from mesh data for wear prediction.
    """
    
    def __init__(self, scaler=None):
        """
        Initialize the dataset.
        
        Args:
            scaler: Optional StandardScaler for feature normalization
        """
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.edge_extractor = EdgeFeatureExtractor()
        
    def create_graph(self, nodes, elements, thickness, youngs_modulus, density, 
                     roller_paths, node_features=None, wear_values=None):
        """
        Create a graph from mesh data.
        
        Args:
            nodes: Node coordinates (n_nodes, 3)
            elements: Element connectivity list
            thickness: Element thickness (scalar or array)
            youngs_modulus: Young's modulus (scalar or array)
            density: Material density (scalar or array)
            roller_paths: Number of roller paths
            node_features: Optional additional node features
            wear_values: Optional known wear values for training
            
        Returns:
            torch_geometric.data.Data: Graph data object
        """
        try:
            # Convert inputs to tensors
            nodes_tensor = torch.tensor(nodes, dtype=torch.float)
            
            # If thickness, youngs_modulus, and density are scalars, convert to tensors
            if not isinstance(thickness, torch.Tensor):
                thickness = torch.tensor(thickness, dtype=torch.float)
            if not isinstance(youngs_modulus, torch.Tensor):
                youngs_modulus = torch.tensor(youngs_modulus, dtype=torch.float)
            if not isinstance(density, torch.Tensor):
                density = torch.tensor(density, dtype=torch.float)
            
            # Generate edge_index and edge_attr
            edge_index, edge_attr = self.edge_extractor.compute_edge_features(
                nodes_tensor, elements, thickness, youngs_modulus
            )
            
            # Create node features
            if node_features is None:
                # Calculate node degree
                edge_index_with_self_loops, _ = add_self_loops(edge_index)
                node_degree = degree(edge_index_with_self_loops[0], num_nodes=len(nodes))
                
                # Basic node features: 3D position, degree, and global properties
                x_features = []
                
                # Add node positions
                x_features.append(nodes_tensor)
                
                # Add node degree as a feature
                x_features.append(node_degree.view(-1, 1))
                
                # Add material properties
                if isinstance(density, torch.Tensor) and density.dim() > 0 and density.shape[0] == len(nodes):
                    # Density per node
                    x_features.append(density.view(-1, 1))
                else:
                    # Same density for all nodes
                    x_features.append(torch.ones(len(nodes), 1) * density)
                
                # Add roller paths as a global feature
                x_features.append(torch.ones(len(nodes), 1) * roller_paths)
                
                # Concatenate all features
                x = torch.cat(x_features, dim=1)
            else:
                # Use provided node features
                x = torch.tensor(node_features, dtype=torch.float)
            
            # Normalize node features
            x_numpy = x.numpy()
            x_scaled = self.scaler.fit_transform(x_numpy)
            x = torch.tensor(x_scaled, dtype=torch.float)
            
            # Create graph data object
            if wear_values is not None:
                # For training data with known wear values
                y = torch.tensor(wear_values, dtype=torch.float).view(-1, 1)
                graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            else:
                # For prediction without known wear values
                graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
            # Add metadata
            graph_data.num_nodes = len(nodes)
            
            return graph_data
            
        except Exception as e:
            logger.error(f"Error creating graph: {e}")
            # Create a minimal valid graph to prevent crashing
            x = torch.tensor(nodes, dtype=torch.float)
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            edge_attr = torch.ones((2, 3), dtype=torch.float)
            
            if wear_values is not None:
                y = torch.tensor(wear_values, dtype=torch.float).view(-1, 1)
                graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            else:
                graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                
            graph_data.num_nodes = len(nodes)
            return graph_data


class EdgeAttentionLayer(MessagePassing):
    """
    Custom edge attention layer for wear prediction.
    """
    
    def __init__(self, in_channels, out_channels):
        super(EdgeAttentionLayer, self).__init__(aggr='add')
        
        self.lin = nn.Linear(in_channels, out_channels)
        self.att = nn.Linear(2*out_channels + 3, 1)  # Combined attention
        self.out_channels = out_channels
        
    def forward(self, x, edge_index, edge_attr):
        # Transform node features
        x = self.lin(x)
        
        # Propagate messages
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
    def message(self, x_i, x_j, edge_attr):
        # Concatenate source, destination and edge features
        alpha_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        
        # Single attention mechanism
        alpha = self.att(alpha_input)
        alpha = F.leaky_relu(alpha)
        
        # Apply sigmoid instead of softmax (more stable)
        alpha = torch.sigmoid(alpha)
        
        # Ensure proper broadcasting
        return x_j * alpha
        
    def update(self, aggr_out):
        return aggr_out


class WearPredictionGNN(nn.Module):
    """
    Graph Neural Network model for predicting wear at each node.
    """
    
    def __init__(self, node_features, edge_features, hidden_channels, num_conv_layers=3, dropout=0.3):
        super(WearPredictionGNN, self).__init__()
        
        self.num_conv_layers = num_conv_layers
        
        # Input feature embedding
        self.node_encoder = nn.Linear(node_features, hidden_channels)
        self.edge_encoder = nn.Linear(edge_features, hidden_channels)
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer: custom edge attention
        self.convs.append(EdgeAttentionLayer(hidden_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Additional GNN layers with different types
        for i in range(1, num_conv_layers):
            if i % 3 == 0:
                # Graph Attention layer
                self.convs.append(GATv2Conv(hidden_channels, hidden_channels, heads=2, concat=False, edge_dim=edge_features))
            elif i % 3 == 1:
                # GraphSAGE layer
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            else:
                # Graph Convolutional layer
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layers
        self.dropout = dropout
        
        # Multi-layer regressor for final prediction
        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )
        
    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        Forward pass through the network.
        
        Args:
            x: Node features [num_nodes, num_node_features]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, num_edge_features]
            batch: Batch assignment for multiple graphs [num_nodes]
            
        Returns:
            torch.Tensor: Predicted wear values for each node [num_nodes, 1]
        """
        # Initial feature encoding
        x = self.node_encoder(x)
        edge_attr_encoded = self.edge_encoder(edge_attr)
        
        # Apply GNN layers with residual connections
        for i in range(self.num_conv_layers):
            identity = x
            
            if i == 0:
                # Custom edge attention layer
                x = self.convs[i](x, edge_index, edge_attr)
            elif isinstance(self.convs[i], GATv2Conv):
                # GAT layer with edge features
                x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            else:
                # Other GNN layers
                x = self.convs[i](x, edge_index)
                
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Add residual connection if shapes match
            if x.shape == identity.shape:
                x = x + identity
        
        # Apply regressor to get wear predictions for each node
        wear_predictions = self.regressor(x)
        
        return wear_predictions


class WearPredictionTrainer:
    """
    Trainer class for the Wear Prediction GNN model.
    """
    
    def __init__(self, model, device, lr=0.001, weight_decay=5e-4):
        """
        Initialize the trainer.
        
        Args:
            model: The GNN model to train
            device: Device to use (cpu/cuda)
            lr: Learning rate
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        self.criterion = nn.MSELoss()
        self.best_val_loss = float('inf')
        
    def train_epoch(self, train_loader):
        """
        Train the model for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            float: Training loss
        """
        self.model.train()
        total_loss = 0
        
        for data in train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = self.criterion(out, data.y)
            
            # Backward and optimize
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * data.num_graphs
            
        return total_loss / len(train_loader.dataset)
        
    def validate(self, val_loader):
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            float: Validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
                loss = self.criterion(out, data.y)
                total_loss += loss.item() * data.num_graphs
                
        val_loss = total_loss / len(val_loader.dataset)
        self.scheduler.step(val_loss)
        
        return val_loss
        
    def train(self, train_loader, val_loader, num_epochs=100, patience=15, 
              save_checkpoints=True, checkpoint_dir='checkpoints', visualize_progress=True):
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            save_checkpoints: Whether to save intermediate checkpoints
            checkpoint_dir: Directory to save checkpoints
            visualize_progress: Whether to visualize training progress
            
        Returns:
            dict: Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        # Create checkpoint directory if needed
        if save_checkpoints:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'best_epoch': 0
        }
        
        best_model_weights = None
        no_improve_epochs = 0
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Train and validate
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            # Track history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Check if this is the best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_weights = self.model.state_dict().copy()
                history['best_epoch'] = epoch
                no_improve_epochs = 0
                logger.info(f"New best model at epoch {epoch}")
                
                # Save the best model
                if save_checkpoints:
                    self.save_model(os.path.join(checkpoint_dir, 'best_model.pt'))
                
            else:
                no_improve_epochs += 1
            
            # Log progress
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch}/{num_epochs} | "
                        f"Train Loss: {train_loss:.6f} | "
                        f"Val Loss: {val_loss:.6f} | "
                        f"Time: {epoch_time:.2f}s | "
                        f"LR: {self.optimizer.param_groups[0]['lr']:.8f}")
            
            # Save checkpoint periodically
            if save_checkpoints and epoch % 20 == 0:
                self.save_model(os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pt'))
            
            # Visualize progress
            if visualize_progress and epoch % 10 == 0:
                self.visualize_progress(val_loader, epoch)
                
            # Early stopping
            if no_improve_epochs >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Restore best model
        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)
            logger.info(f"Restored best model from epoch {history['best_epoch']}")
        
        return history
        
    def predict(self, data_loader):
        """
        Make predictions with the trained model.
        
        Args:
            data_loader: DataLoader for test data
            
        Returns:
            torch.Tensor: Predicted wear values
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
                predictions.append(out.cpu())
                
        return torch.cat(predictions, dim=0)
        
    def evaluate(self, test_loader, return_predictions=False):
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: DataLoader for test data
            return_predictions: Whether to return predictions
            
        Returns:
            dict: Dictionary of evaluation metrics
            torch.Tensor (optional): Predicted values
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
                loss = self.criterion(out, data.y)
                total_loss += loss.item() * data.num_graphs
                
                predictions.append(out.cpu())
                targets.append(data.y.cpu())
                
        test_loss = total_loss / len(test_loader.dataset)
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
        
        # Calculate metrics
        mse = F.mse_loss(predictions, targets).item()
        mae = F.l1_loss(predictions, targets).item()
        rmse = torch.sqrt(torch.tensor(mse)).item()
        
        metrics = {
            'test_loss': test_loss,
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        }
        
        logger.info(f"Test Metrics: MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")
        
        if return_predictions:
            return metrics, predictions.numpy()
        else:
            return metrics
    
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss
        }, filepath)
        logger.info(f"Model saved to {filepath}")
        
    def load_model(self, filepath):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Model loaded from {filepath}")
        
    def visualize_progress(self, val_loader, epoch):
        """
        Visualize current model predictions on validation data.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
        """
        try:
            # Get a sample from validation set
            sample = next(iter(val_loader)).to(self.device)
            
            # Get predictions
            self.model.eval()
            with torch.no_grad():
                pred = self.model(sample.x, sample.edge_index, sample.edge_attr, sample.batch)
            
            # Create 2D visualization (compatible with all matplotlib versions)
            plt.figure(figsize=(12, 8))
            
            # Extract data for plotting
            node_pos = sample.x[:, :2].cpu().numpy()
            actual_wear = sample.y.cpu().numpy()
            predicted_wear = pred.cpu().numpy()
            
            # Create comparison plot
            plt.subplot(1, 2, 1)
            plt.scatter(node_pos[:, 0], node_pos[:, 1], c=actual_wear, cmap='viridis')
            plt.colorbar(label='Actual Wear')
            plt.title('Actual Wear')
            
            plt.subplot(1, 2, 2)
            plt.scatter(node_pos[:, 0], node_pos[:, 1], c=predicted_wear, cmap='viridis')
            plt.colorbar(label='Predicted Wear')
            plt.title('Predicted Wear')
            
            plt.suptitle(f'Epoch {epoch} - Model Prediction')
            
            # Save figure
            plt.tight_layout()
            plt.savefig(f'progress_epoch_{epoch}.png')
            plt.close()
            
            logger.info(f"Saved prediction visualization for epoch {epoch}")
            
        except Exception as e:
            logger.error(f"Error visualizing progress for epoch {epoch}: {e}")


def softmax(src, index, ptr, size_i):
    """
    Custom softmax function for sparse attention.
    
    Args:
        src: Source values
        index: Edge indices
        ptr: Pointers
        size_i: Size of the target dimension
        
    Returns:
        torch.Tensor: Softmax values
    """
    if ptr is not None:
        # If we have multiple graphs, we need to use ptr
        return src - torch.gather(scatter_max(src, ptr, dim=0, dim_size=size_i)[0], 0,
                                 torch.repeat_interleave(ptr[1:] - ptr[:-1], ptr[1:] - ptr[:-1]))
    else:
        # If we have a single graph, we can use index
        return src - scatter_max(src, index, dim=0, dim_size=size_i)[0][index]


def scatter_max(src, index, dim=-1, dim_size=None):
    """
    Custom scatter max function for sparse attention.
    
    Args:
        src: Source values
        index: Indices
        dim: Dimension along which to scatter
        dim_size: Size of the output dimension
        
    Returns:
        tuple: (Output tensor, indices of maximum values)
    """
    if dim_size is None:
        dim_size = index.max().item() + 1
        
    output = torch.zeros(dim_size, device=src.device)
    indices = torch.zeros(dim_size, dtype=torch.long, device=src.device)
    
    for i in range(len(index)):
        idx = index[i].item()
        if src[i].item() > output[idx].item():
            output[idx] = src[i]
            indices[idx] = i
            
    return output, indices


def visualize_graph(graph, predicted_wear=None, actual_wear=None, node_size=50, ax=None):
    """
    Visualize a graph and its wear predictions.
    
    Args:
        graph: PyG Data object representing the graph
        predicted_wear: Optional tensor of predicted wear values
        actual_wear: Optional tensor of actual wear values
        node_size: Size of nodes in the visualization
        ax: Matplotlib axes for plotting
        
    Returns:
        matplotlib.axes.Axes: The plot axes
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Convert PyG graph to NetworkX for visualization
    G = nx.Graph()
    
    # Add nodes with 3D positions
    node_pos = graph.x[:, :3].numpy()
    for i in range(graph.num_nodes):
        G.add_node(i, pos=node_pos[i])
    
    # Add edges
    edge_index = graph.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i], edge_index[1, i])
    
    # Get node positions
    pos_3d = nx.get_node_attributes(G, 'pos')
    
    # Prepare for 3D visualization
    xs = [pos[0] for pos in pos_3d.values()]
    ys = [pos[1] for pos in pos_3d.values()]
    zs = [pos[2] for pos in pos_3d.values()]
    
    # Plot edges
    for edge in G.edges():
        x = [pos_3d[edge[0]][0], pos_3d[edge[1]][0]]
        y = [pos_3d[edge[0]][1], pos_3d[edge[1]][1]]
        z = [pos_3d[edge[0]][2], pos_3d[edge[1]][2]]
        ax.plot(x, y, z, 'gray', alpha=0.5, linewidth=0.5)
    
    # Color nodes based on wear
    if predicted_wear is not None:
        colors = predicted_wear.flatten().numpy()
        scatter = ax.scatter(xs, ys, zs, c=colors, s=node_size, cmap='viridis', 
                            vmin=min(colors), vmax=max(colors))
        plt.colorbar(scatter, ax=ax, label='Predicted Wear')
        
        if actual_wear is not None:
            # Add a text annotation for model error
            mse = np.mean((colors - actual_wear.numpy().flatten())**2)
            rmse = np.sqrt(mse)
            ax.text2D(0.05, 0.95, f'RMSE: {rmse:.4f}', transform=ax.transAxes)
    else:
        ax.scatter(xs, ys, zs, c='blue', s=node_size)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Mesh Structure with Node Wear Prediction')
    
    return ax


def visualize_sample(graph_data, sample_name, output_dir='sample_visualizations'):
    """
    Visualize a single graph sample.
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        from mpl_toolkits.mplot3d import Axes3D  # Import this explicitly
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract node positions and wear values
        node_pos = graph_data.x[:, :3].cpu().numpy()
        wear = graph_data.y.cpu().numpy()
        
        # Create scatter plot
        scatter = ax.scatter(node_pos[:, 0], node_pos[:, 1], node_pos[:, 2], 
                            c=wear, cmap='viridis', s=50)
        
        plt.colorbar(scatter, label='Wear')
        ax.set_title(f'Sample: {sample_name}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.savefig(os.path.join(output_dir, f'{sample_name}.png'))
        plt.close()
    except Exception as e:
        # Fallback to 2D plotting if 3D fails
        logger.error(f"Error with 3D visualization, falling back to 2D: {e}")
        plt.figure(figsize=(10, 8))
        node_pos = graph_data.x[:, :2].cpu().numpy()
        plt.scatter(node_pos[:, 0], node_pos[:, 1], c=graph_data.y.cpu().numpy(), cmap='viridis')
        plt.colorbar(label='Wear')
        plt.title(f'Sample: {sample_name}')
        plt.savefig(os.path.join(output_dir, f'{sample_name}.png'))
        plt.close()


def generate_synthetic_data(num_samples=100, noise_level=0.1):
    """
    Generate synthetic data for testing the GNN model.
    
    Args:
        num_samples: Number of samples to generate
        noise_level: Level of noise to add to the target
        
    Returns:
        list: List of graph data objects
    """
    logger.info(f"Generating {num_samples} synthetic samples")
    
    graphs = []
    dataset = MeshGraphDataset()
    
    for i in range(num_samples):
        # Generate random number of nodes
        n_nodes = np.random.randint(50, 200)
        
        # Generate random 3D node positions
        nodes = np.random.rand(n_nodes, 3) * 10
        
        # Generate elements (triangles for simplicity)
        elements = []
        for j in range(n_nodes - 2):
            if j % 3 == 0 and j + 2 < n_nodes:
                elements.append([j, j+1, j+2])
        
        # Random material properties
        thickness = np.random.uniform(0.5, 2.0, len(elements))
        youngs_modulus = np.random.uniform(1e5, 5e5, len(elements))
        density = np.random.uniform(2000, 8000)
        roller_paths = np.random.randint(1, 10)
        
        # Generate synthetic wear values based on a simplified physical model
        # Higher thickness -> Lower wear
        # Higher Young's modulus -> Lower wear
        # Higher density -> Higher wear
        # More roller paths -> Higher wear
        
        # Base wear calculation
        base_wear = np.zeros(n_nodes)
        
        # Effect of node position (higher wear at extremes)
        position_factor = np.sum(np.abs(nodes - np.mean(nodes, axis=0)), axis=1) / 10
        base_wear += position_factor
        
        # Effect of roller paths (more paths -> more wear)
        base_wear *= (1 + 0.2 * roller_paths)
        
        # Add element-based effects to connected nodes
        for j, elem in enumerate(elements):
            # Inverse relationship with thickness and Young's modulus
            wear_factor = 1 / (thickness[j] * youngs_modulus[j] * 1e-6)
            
            # Distribute wear factor to nodes in this element
            for node_idx in elem:
                base_wear[node_idx] += wear_factor * 0.1
        
        # Effect of density (higher density -> higher wear)
        base_wear *= (density / 5000)
        
        # Add noise
        noise = np.random.normal(0, noise_level * np.mean(base_wear), size=n_nodes)
        wear = base_wear + noise
        
        # Ensure wear is positive
        wear = np.maximum(wear, 0.01)
        
        # Convert to tensors
        thickness_tensor = torch.tensor(thickness, dtype=torch.float)
        youngs_modulus_tensor = torch.tensor(youngs_modulus, dtype=torch.float)
        
        # Create graph
        graph = dataset.create_graph(
            nodes=nodes,
            elements=elements,
            thickness=thickness_tensor,
            youngs_modulus=youngs_modulus_tensor,
            density=density,
            roller_paths=roller_paths,
            wear_values=wear
        )
        
        graphs.append(graph)
        
    return graphs


def load_wear_dataset(dataset_dir):
    """
    Load dataset from the wear_dataset directory structure.
    
    Args:
        dataset_dir: Path to the dataset directory containing sample folders
        
    Returns:
        list: List of PyG Data objects
    """
    logger.info(f"Loading dataset from {dataset_dir}")
    
    # Create dataset object
    dataset_builder = MeshGraphDataset()
    graphs = []
    
    # List all sample directories (sample_01, sample_02, etc.)
    try:
        sample_dirs = sorted([d for d in os.listdir(dataset_dir) 
                             if os.path.isdir(os.path.join(dataset_dir, d))])
        
        for sample_dir in sample_dirs:
            sample_path = os.path.join(dataset_dir, sample_dir)
            try:
                # Load nodes
                nodes = np.loadtxt(os.path.join(sample_path, 'nodes.csv'), delimiter=',')
                
                # Load elements
                elements = np.loadtxt(os.path.join(sample_path, 'elements.csv'), 
                                     delimiter=',', dtype=np.int64).tolist()
                
                # Load properties (thickness, youngs_modulus, density, num_roller_paths)
                properties = np.loadtxt(os.path.join(sample_path, 'properties.csv'), delimiter=',')
                thickness = properties[0]
                youngs_modulus = properties[1]
                density = properties[2]
                num_roller_paths = int(properties[3])
                
                # Load wear values (target)
                wear = np.loadtxt(os.path.join(sample_path, 'wear.csv'), delimiter=',')
                
                # Create graph data object
                graph = dataset_builder.create_graph(
                    nodes=nodes,
                    elements=elements,
                    thickness=thickness,
                    youngs_modulus=youngs_modulus,
                    density=density,
                    roller_paths=num_roller_paths,
                    wear_values=wear
                )
                
                graphs.append(graph)
                logger.info(f"Loaded {sample_dir}: {len(nodes)} nodes, {len(elements)} elements")
                
                # Visualize some samples for verification
                if len(graphs) <= 3:  # Only visualize first 3 samples
                    visualize_sample(graph, sample_dir)
                
            except Exception as e:
                logger.error(f"Error loading sample {sample_dir}: {e}")
        
        logger.info(f"Successfully loaded {len(graphs)} samples")
    except Exception as e:
        logger.error(f"Error accessing dataset directory: {e}")
    
    return graphs


def load_real_data(node_file, element_file, material_file, wear_file=None):
    """
    Load real data from files.
    
    Args:
        node_file: Path to file containing node coordinates
        element_file: Path to file containing element connectivity
        material_file: Path to file containing material properties
        wear_file: Optional path to file containing wear values
        
    Returns:
        dict: Dictionary containing loaded data
    """
    logger.info("Loading real data from files")
    
    # Load nodes
    nodes_df = pd.read_csv(node_file)
    nodes = nodes_df[['x', 'y', 'z']].values
    
    # Load elements
    elements_df = pd.read_csv(element_file)
    elements = []
    for _, row in elements_df.iterrows():
        elem = [int(row[f'node{i}']) for i in range(1, len(row)+1) if f'node{i}' in row]
        elements.append(elem)
    
    # Load material properties
    material_df = pd.read_csv(material_file)
    thickness = material_df['thickness'].values
    youngs_modulus = material_df['youngs_modulus'].values
    density = material_df['density'].values
    roller_paths = material_df['roller_paths'].values[0] if 'roller_paths' in material_df else 1
    
    # Load wear values if available
    wear = None
    if wear_file is not None and os.path.exists(wear_file):
        wear_df = pd.read_csv(wear_file)
        wear = wear_df['wear'].values
    
    return {
        'nodes': nodes,
        'elements': elements,
        'thickness': thickness,
        'youngs_modulus': youngs_modulus,
        'density': density,
        'roller_paths': roller_paths,
        'wear': wear
    }


def prepare_datasets(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=32):
    """
    Prepare train, validation and test datasets.
    
    Args:
        data: List of graph data objects or dictionary of loaded data
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        batch_size: Batch size for data loaders
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    logger.info("Preparing datasets")
    
    # If data is a dictionary (loaded from files), create a single graph
    if isinstance(data, dict):
        dataset = MeshGraphDataset()
        graph = dataset.create_graph(
            nodes=data['nodes'],
            elements=data['elements'],
            thickness=torch.tensor(data['thickness'], dtype=torch.float),
            youngs_modulus=torch.tensor(data['youngs_modulus'], dtype=torch.float),
            density=data['density'],
            roller_paths=data['roller_paths'],
            wear_values=data['wear']
        )
        data = [graph]
    
    # Split data into train, validation and test sets
    n_samples = len(data)
    indices = list(range(n_samples))
    np.random.shuffle(indices)
    
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_dataset = [data[i] for i in train_indices]
    val_dataset = [data[i] for i in val_indices]
    test_dataset = [data[i] for i in test_indices]
    
    # Create data loaders with specified batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    logger.info(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test")
    
    return train_loader, val_loader, test_loader


def plot_training_history(history):
    """
    Plot training history.
    
    Args:
        history: Dictionary containing training and validation losses
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.axvline(x=history['best_epoch'], color='r', linestyle='--', label=f'Best Model (Epoch {history["best_epoch"]})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    plt.close()


def main(use_synthetic_data=True, node_file=None, element_file=None, material_file=None, wear_file=None):
    """
    Main function to run the GNN wear prediction model.
    
    Args:
        use_synthetic_data: Whether to use synthetic data
        node_file: Path to node file (if use_synthetic_data is False)
        element_file: Path to element file (if use_synthetic_data is False)
        material_file: Path to material file (if use_synthetic_data is False)
        wear_file: Path to wear file (if use_synthetic_data is False)
    """
    logger.info("Starting Wear Prediction GNN")
    
    # Set up the data
    if use_synthetic_data:
        logger.info("Using synthetic data")
        data = generate_synthetic_data(num_samples=200, noise_level=0.1)
    else:
        logger.info("Using real data from files")
        data = load_real_data(node_file, element_file, material_file, wear_file)
    
    # Prepare datasets
    train_loader, val_loader, test_loader = prepare_datasets(data)
    
    # Get feature dimensions from the data
    sample_data = next(iter(train_loader))
    node_features = sample_data.x.size(1)
    edge_features = sample_data.edge_attr.size(1)
    
    # Create model
    hidden_channels = 64
    model = WearPredictionGNN(
        node_features=node_features,
        edge_features=edge_features,
        hidden_channels=hidden_channels,
        num_conv_layers=4,
        dropout=0.2
    )
    
    logger.info(f"Model created with {node_features} node features and {edge_features} edge features")
    
    # Train model
    trainer = WearPredictionTrainer(model, device, lr=0.001, weight_decay=5e-4)
    history = trainer.train(train_loader, val_loader, num_epochs=150, patience=15)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    metrics = trainer.evaluate(test_loader)
    logger.info(f"Test evaluation: {metrics}")
    
    # Save model
    trainer.save_model('wear_prediction_model.pt')
    
    # Visualize results on a sample
    sample = next(iter(test_loader)).to(device)
    sample_pred = trainer.model(sample.x, sample.edge_index, sample.edge_attr)
    
    plt.figure(figsize=(12, 10))
    ax = plt.gca(projection='3d')
    visualize_graph(sample, predicted_wear=sample_pred.cpu(), actual_wear=sample.y.cpu(), ax=ax)
    plt.savefig('wear_prediction_visualization.png')
    plt.close()
    
    logger.info("Wear prediction model training and evaluation completed")


def main_wear_dataset(dataset_dir='wear_dataset', batch_size=8):
    """
    Main function to run the GNN wear prediction model with your dataset.
    
    Args:
        dataset_dir: Path to the dataset directory
        batch_size: Batch size for training (smaller batch for memory efficiency)
    """
    logger.info("Starting Wear Prediction GNN with custom dataset")
    
    # Load the dataset
    data = load_wear_dataset(dataset_dir)
    
    if not data:
        logger.error("No data loaded from the dataset directory!")
        return
    
    # Prepare datasets with smaller batch size for memory efficiency
    train_loader, val_loader, test_loader = prepare_datasets(data, batch_size=batch_size)
    
    # Get feature dimensions from the data
    sample_data = next(iter(train_loader))
    node_features = sample_data.x.size(1)
    edge_features = sample_data.edge_attr.size(1)
    
    '''
    # sliglty complex model (smaller hidden channels for memory efficiency)
    hidden_channels = 32
    model = WearPredictionGNN(
        node_features=node_features,
        edge_features=edge_features,
        hidden_channels=hidden_channels,
        num_conv_layers=3,  # Fewer layers for memory efficiency
        dropout=0.2
    )
    '''
    # Create a simpler model
    hidden_channels = 16
    model = WearPredictionGNN(
        node_features=node_features,
        edge_features=edge_features,
        hidden_channels=hidden_channels,
        num_conv_layers=2,  # Reduced from 3
        dropout=0.2
    )
    
    logger.info(f"Model created with {node_features} node features and {edge_features} edge features")
    
    # Train model
    trainer = WearPredictionTrainer(model, device, lr=0.001, weight_decay=5e-4)
    history = trainer.train(
        train_loader, 
        val_loader, 
        num_epochs=150, 
        patience=15,
        save_checkpoints=True,
        checkpoint_dir='wear_checkpoints',
        visualize_progress=True
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    metrics = trainer.evaluate(test_loader)
    logger.info(f"Test evaluation: {metrics}")
    
    # Save model
    trainer.save_model('wear_prediction_model.pt')
    
    # Visualize results on a sample
    sample = next(iter(test_loader)).to(device)
    sample_pred = trainer.model(sample.x, sample.edge_index, sample.edge_attr)
    
    plt.figure(figsize=(12, 10))
    ax = plt.gca(projection='3d')
    visualize_graph(sample, predicted_wear=sample_pred.cpu(), actual_wear=sample.y.cpu(), ax=ax)
    plt.savefig('wear_prediction_visualization.png')
    plt.close()
    
    logger.info("Wear prediction model training and evaluation completed")


def predict_wear(model_path, nodes, elements, thickness, youngs_modulus, density, roller_paths):
    """
    Predict wear on new data using a trained model.
    
    Args:
        model_path: Path to the saved model
        nodes: Node coordinates
        elements: Element connectivity
        thickness: Element thickness
        youngs_modulus: Young's modulus
        density: Material density
        roller_paths: Number of roller paths
        
    Returns:
        numpy.ndarray: Predicted wear values for each node
    """
    # Load the model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get node_features and edge_features from the dataset
    dataset = MeshGraphDataset()
    graph = dataset.create_graph(
        nodes=nodes,
        elements=elements,
        thickness=torch.tensor(thickness, dtype=torch.float),
        youngs_modulus=torch.tensor(youngs_modulus, dtype=torch.float),
        density=density,
        roller_paths=roller_paths
    )
    
    # Get model parameters from the graph
    node_features = graph.x.size(1)
    edge_features = graph.edge_attr.size(1)
    
    # Create model with same architecture (adapted to the input dimensions)
    hidden_channels = 64
    model = WearPredictionGNN(
        node_features=node_features,
        edge_features=edge_features,
        hidden_channels=hidden_channels,
        num_conv_layers=4,
        dropout=0.0  # No dropout during inference
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        edge_attr = graph.edge_attr.to(device)
        
        predictions = model(x, edge_index, edge_attr)
        
    return predictions.cpu().numpy()


if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='GNN for Wear Prediction')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--custom-dataset', action='store_true', help='Use custom dataset from wear_dataset folder')
    parser.add_argument('--dataset-dir', type=str, default='wear_dataset', help='Path to custom dataset directory')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--node-file', type=str, help='Path to node file')
    parser.add_argument('--element-file', type=str, help='Path to element file')
    parser.add_argument('--material-file', type=str, help='Path to material file')
    parser.add_argument('--wear-file', type=str, help='Path to wear file')
    parser.add_argument('--predict', action='store_true', help='Make prediction using trained model')
    parser.add_argument('--model-path', type=str, default='wear_prediction_model.pt', help='Path to trained model')
    parser.add_argument('--visualize-samples', action='store_true', help='Visualize dataset samples before training')
    
    args = parser.parse_args()
    
    if args.predict:
        if not args.node_file or not args.element_file or not args.material_file:
            logger.error("For prediction, node-file, element-file, and material-file are required")
            sys.exit(1)
            
        # Load data for prediction
        data = load_real_data(args.node_file, args.element_file, args.material_file)
        
        # Make prediction
        predictions = predict_wear(
            args.model_path,
            data['nodes'],
            data['elements'],
            data['thickness'],
            data['youngs_modulus'],
            data['density'],
            data['roller_paths']
        )
        
        # Visualize prediction
        dataset = MeshGraphDataset()
        graph = dataset.create_graph(
            nodes=data['nodes'],
            elements=data['elements'],
            thickness=torch.tensor(data['thickness'], dtype=torch.float),
            youngs_modulus=torch.tensor(data['youngs_modulus'], dtype=torch.float),
            density=data['density'],
            roller_paths=data['roller_paths']
        )
        
        plt.figure(figsize=(12, 10))
        ax = plt.gca(projection='3d')
        visualize_graph(graph, predicted_wear=torch.tensor(predictions), ax=ax)
        plt.savefig('wear_prediction.png')
        plt.show()
        
        # Save predictions to CSV
        results_df = pd.DataFrame({
            'node_id': range(len(data['nodes'])),
            'x': data['nodes'][:, 0],
            'y': data['nodes'][:, 1],
            'z': data['nodes'][:, 2],
            'predicted_wear': predictions.flatten()
        })
        results_df.to_csv('wear_predictions.csv', index=False)
        
        logger.info(f"Predictions saved to wear_predictions.csv and wear_prediction.png")
    
    elif args.visualize_samples:
        # Just load and visualize samples without training
        if args.custom_dataset:
            data = load_wear_dataset(args.dataset_dir)
            logger.info("Dataset samples visualized. Check 'sample_visualizations' directory.")
        else:
            logger.error("Please specify --custom-dataset to use this option")
    
    elif args.custom_dataset:
        # Use the custom dataset from the wear_dataset directory
        main_wear_dataset(dataset_dir=args.dataset_dir, batch_size=args.batch_size)
        
    else:
        # Run training with either synthetic or file-based data
        main(
            use_synthetic_data=args.synthetic,
            node_file=args.node_file,
            element_file=args.element_file,
            material_file=args.material_file,
            wear_file=args.wear_file
        )
        