import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, GATConv
from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt

class WearMeshDataset:
    """
    Dataset class for wear prediction on mechanical structures.
    """
    def __init__(self, root_dir):
        """
        Args:
            root_dir (str): Directory with input data
        """
        self.root_dir = root_dir
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """Load all samples from the directory"""
        samples = []
        sample_dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        sample_dirs.sort()  # Ensure consistent ordering
        
        print(f"Found {len(sample_dirs)} sample directories")
        
        for sample_dir in sample_dirs:
            try:
                sample_path = os.path.join(self.root_dir, sample_dir)
                
                # Load node positions
                nodes = np.loadtxt(os.path.join(sample_path, 'nodes.csv'), delimiter=',')
                
                # Load element connectivity
                elements = np.loadtxt(os.path.join(sample_path, 'elements.csv'), 
                                     delimiter=',', dtype=np.int64)
                
                # Load material properties
                properties = np.loadtxt(os.path.join(sample_path, 'properties.csv'), delimiter=',')
                thickness = properties[0]
                youngs_modulus = properties[1]
                density = properties[2]
                num_roller_paths = int(properties[3])
                
                # Load roller paths
                roller_paths = []
                for i in range(num_roller_paths):
                    path_file = os.path.join(sample_path, f'roller_path_{i+1}.csv')
                    if os.path.exists(path_file):
                        path = np.loadtxt(path_file, delimiter=',')
                        roller_paths.append(path)
                
                # Load wear values (target)
                wear = np.loadtxt(os.path.join(sample_path, 'wear.csv'), delimiter=',')
                
                print(f"Sample {sample_dir}: {len(nodes)} nodes, {len(elements)} elements, {len(wear)} wear values")
                
                # Convert to PyG Data object
                data = self._convert_to_pyg_data(nodes, elements, thickness, youngs_modulus, 
                                               density, roller_paths, wear)
                samples.append(data)
                
            except Exception as e:
                print(f"Error loading sample {sample_dir}: {e}")
                
        return samples
    
    def _convert_to_pyg_data(self, nodes, elements, thickness, youngs_modulus, density, roller_paths, wear):
        """Convert sample data to PyTorch Geometric Data object"""
        # Node features: Position (x,y,z) + material properties
        node_features = np.zeros((len(nodes), 6))
        node_features[:, :3] = nodes  # x, y, z coordinates
        node_features[:, 3] = thickness  # same thickness for all nodes
        node_features[:, 4] = youngs_modulus  # same Young's modulus
        node_features[:, 5] = density  # same density
        
        # Create edge index from elements
        edges = []
        for element in elements:
            # For each element, create edges between all pairs of nodes
            for i in range(len(element)):
                for j in range(i+1, len(element)):
                    edges.append([element[i], element[j]])
                    edges.append([element[j], element[i]])  # Bidirectional
        
        # Remove duplicates from edges
        edges = list(set(tuple(edge) for edge in edges))
        edges = [list(edge) for edge in edges]
                    
        # Convert to edge index format
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Generate edge features - distance between nodes
        edge_features = []
        for edge in edges:
            n1, n2 = edge
            coord1 = nodes[n1]
            coord2 = nodes[n2]
            distance = np.linalg.norm(coord1 - coord2)
            edge_features.append([distance])
        
        # Roller path information as additional node features
        if roller_paths:
            roller_distances = np.zeros((len(nodes), 1))
            for i, node_pos in enumerate(nodes):
                min_dist = float('inf')
                for path in roller_paths:
                    for point in path:
                        dist = np.linalg.norm(node_pos[:2] - point[:2])
                        if dist < min_dist:
                            min_dist = dist
                roller_distances[i, 0] = min_dist
            
            # Add roller path information to node features
            node_features = np.hstack((node_features, roller_distances))
        
        # Convert to PyG Data object
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=torch.tensor(edge_features, dtype=torch.float),
            y=torch.tensor(wear, dtype=torch.float)
        )
        
        return data
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

class CustomWearMessagePassingLayer(MessagePassing):
    """
    Custom message passing layer that accounts for physical properties
    """
    def __init__(self, in_channels, out_channels):
        super(CustomWearMessagePassingLayer, self).__init__(aggr='add')  # "Add" aggregation
        
        self.lin_node = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(1, out_channels)  # Edge feature is just distance
        self.lin_message = nn.Linear(in_channels + out_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr):
        # Transform node features
        node_feats = self.lin_node(x)
        
        # Add self-loops to edge_index
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Extend edge_attr for self-loops (use zero distance for self-loops)
        self_loop_attr = torch.zeros(x.size(0), 1, device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        
        # Start propagating messages
        return self.propagate(edge_index, x=x, node_feats=node_feats, edge_attr=edge_attr)
    
    def message(self, x_j, node_feats_j, edge_attr):
        # Transform edge features
        edge_feats = self.lin_edge(edge_attr)
        
        # Combine node and edge features
        message = torch.cat([x_j, edge_feats], dim=-1)
        
        # Transform the combined features
        return self.lin_message(message)
    
    def update(self, aggr_out, node_feats):
        # Combine aggregated messages with the node's transformed features
        return F.relu(aggr_out + node_feats)

class PhysicsInformedGNN(torch.nn.Module):
    """
    GNN with physics-informed components for wear prediction
    """
    def __init__(self, num_node_features, hidden_channels=64):
        super(PhysicsInformedGNN, self).__init__()
        
        # First custom physics-informed message passing layer
        self.conv1 = CustomWearMessagePassingLayer(num_node_features, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        # Second custom physics-informed message passing layer
        self.conv2 = CustomWearMessagePassingLayer(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        # Third layer with attention
        self.conv3 = GATConv(hidden_channels, hidden_channels, heads=4, concat=False)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
        # MLP for final prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, hidden_channels//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels//2, 1)
        )
        
        # Physics-informed regularization parameters
        self.stress_weight = nn.Parameter(torch.tensor(1.0))
        self.thickness_weight = nn.Parameter(torch.tensor(1.0))
        self.distance_weight = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Extract physical properties
        positions = x[:, :3]
        thickness = x[:, 3].view(-1, 1)
        youngs_modulus = x[:, 4].view(-1, 1)
        density = x[:, 5].view(-1, 1)
        
        # First layer
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second layer
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Third layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Final MLP
        wear = self.mlp(x).view(-1)
        
        return wear
    
    def physics_regularization(self, data, predicted_wear):
        """
        Add physics-based regularization to the loss function
        """
        x = data.x
        positions = x[:, :3]
        thickness = x[:, 3].view(-1)
        youngs_modulus = x[:, 4].view(-1)
        
        # 1. Wear should be inversely proportional to thickness
        thickness_reg = torch.mean(predicted_wear * thickness)
        
        # 2. Wear should be inversely proportional to Young's modulus
        modulus_reg = torch.mean(predicted_wear * youngs_modulus / 1e10)
        
        # 3. Wear should decrease with distance from roller paths
        if x.shape[1] > 6:  # If we have roller path distances
            roller_distances = x[:, 6].view(-1)
            distance_reg = -torch.mean(predicted_wear * torch.exp(-roller_distances))
        else:
            distance_reg = 0.0
            
        return self.thickness_weight * thickness_reg + \
               self.stress_weight * modulus_reg + \
               self.distance_weight * distance_reg

def train(model, train_loader, optimizer, device, physics_weight=0.1):
    model.train()
    total_loss = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(data)
        
        # Main loss (MSE)
        mse_loss = F.mse_loss(pred, data.y)
        
        # Add physics-based regularization
        physics_reg = model.physics_regularization(data, pred)
        loss = mse_loss + physics_weight * physics_reg
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_nodes
        
    return total_loss / len(train_loader.dataset)

def test(model, test_loader, device):
    model.eval()
    mse_sum = 0
    mae_sum = 0
    num_nodes = 0
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred = model(data)
            
            # Calculate MSE
            mse = F.mse_loss(pred, data.y, reduction='sum').item()
            mse_sum += mse
            
            # Calculate MAE
            mae = F.l1_loss(pred, data.y, reduction='sum').item()
            mae_sum += mae
            
            num_nodes += data.num_nodes
    
    return mse_sum / num_nodes, mae_sum / num_nodes

def visualize_results(model, data, device, output_file=None):
    """Visualize the predicted wear vs actual wear on nodes"""
    model.eval()
    data = data.to(device)
    
    with torch.no_grad():
        pred = model(data)
    
    # Get node positions and wear values
    node_pos = data.x[:, :3].cpu().numpy()
    actual_wear = data.y.cpu().numpy()
    predicted_wear = pred.cpu().numpy()
    
    # Create a figure with two subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot actual wear
    scatter1 = ax1.scatter(node_pos[:, 0], node_pos[:, 1], c=actual_wear, 
                          cmap='viridis', s=50, alpha=0.8)
    ax1.set_title('Actual Wear')
    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')
    fig.colorbar(scatter1, ax=ax1, label='Wear Magnitude')
    
    # Plot predicted wear
    scatter2 = ax2.scatter(node_pos[:, 0], node_pos[:, 1], c=predicted_wear, 
                          cmap='viridis', s=50, alpha=0.8)
    ax2.set_title('Predicted Wear')
    ax2.set_xlabel('X position')
    ax2.set_ylabel('Y position')
    fig.colorbar(scatter2, ax=ax2, label='Wear Magnitude')
    
    # Plot prediction error
    error = np.abs(actual_wear - predicted_wear)
    scatter3 = ax3.scatter(node_pos[:, 0], node_pos[:, 1], c=error, 
                          cmap='Reds', s=50, alpha=0.8)
    ax3.set_title('Prediction Error')
    ax3.set_xlabel('X position')
    ax3.set_ylabel('Y position')
    fig.colorbar(scatter3, ax=ax3, label='Error Magnitude')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Results visualization saved to {output_file}")
    
    return fig

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Wear Prediction GNN')
    parser.add_argument('--dataset_dir', type=str, default='wear_dataset',
                      help='Directory containing the dataset')
    parser.add_argument('--output_model', type=str, default='wear_gnn_model.pt',
                      help='Path to save the trained model')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--visualize', action='store_true',
                      help='Visualize results')
    
    args = parser.parse_args()
    
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = WearMeshDataset(root_dir=args.dataset_dir)
    print(f"Loaded dataset with {len(dataset)} samples")
    
    # Split into train/validation/test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Determine number of node features from the first sample
    first_data = dataset[0]
    num_node_features = first_data.x.shape[1]
    print(f"Number of node features: {num_node_features}")
    
    # Initialize model
    model = PhysicsInformedGNN(num_node_features=num_node_features, hidden_channels=64).to(device)
    print(model)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=5, 
                                                         verbose=True)
    
    # Training loop
    best_val_mse = float('inf')
    best_epoch = 0
    patience = 15
    counter = 0
    
    for epoch in range(args.epochs):
        # Train
        loss = train(model, train_loader, optimizer, device, physics_weight=0.1)
        
        # Validate
        val_mse, val_mae = test(model, val_loader, device)
        
        # Learning rate scheduler
        scheduler.step(val_mse)
        
        # Print progress
        print(f'Epoch: {epoch+1:03d}, Train Loss: {loss:.4f}, Val MSE: {val_mse:.4f}, Val MAE: {val_mae:.4f}')
        
        # Check for early stopping
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            counter = 0
            # Save the best model
            torch.save(model.state_dict(), args.output_model)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(args.output_model))
    
    # Test on test set
    test_mse, test_mae = test(model, test_loader, device)
    print(f'Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}')
    
    # Visualize results for a sample
    if args.visualize:
        test_data = test_dataset[0]
        fig = visualize_results(model, test_data, device, 'wear_prediction_results.png')
        
    print(f"Training completed. Model saved to {args.output_model}")