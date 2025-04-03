# Mesh Graph Neural Network for Material Wear Prediction

## Overview

This project implements a Graph Neural Network (GNN) framework for predicting wear patterns on structural mesh components. By representing the mesh as a graph structure, the model learns to predict node-level wear based on geometric features, material properties, and loading conditions.

## Key Features

- Custom Graph Neural Network architecture combining multiple message-passing schemes
- Edge-aware attention mechanism incorporating physical properties
- Support for both synthetic and real-world mesh data
- Comprehensive visualization tools for model predictions
- Highly accurate wear prediction (RMSE < 0.001 on test data)

## Prerequisites

```bash
# Core dependencies
pip install torch==1.12.0
pip install torch-geometric==2.2.0
pip install numpy==1.24.4  # Use a version below 2.0 for compatibility
pip install matplotlib==3.5.2
pip install pandas==1.5.3
pip install scikit-learn==1.1.3

# Optional for 3D visualization
pip install mpl_toolkits
```

## Dataset Structure

The project expects data to be organized in a directory structure as follows:

```
wear_dataset/
├── sample_01/
│   ├── nodes.csv       # Node coordinates (x,y,z)
│   ├── elements.csv    # Element connectivity 
│   ├── properties.csv  # Material properties (thickness, Young's modulus, density, roller_paths)
│   └── wear.csv        # Target wear values for each node
├── sample_02/
│   ├── ...
...
```

## Usage

### Training with Custom Dataset

```bash
python GNN.py --custom-dataset --dataset-dir wear_dataset --batch-size 1
```

### Training with Synthetic Data

```bash
python GNN.py --synthetic
```

### Making Predictions with Trained Model

```bash
python GNN.py --predict --model-path wear_prediction_model.pt \
              --node-file path/to/nodes.csv \
              --element-file path/to/elements.csv \
              --material-file path/to/material.csv
```

### Visualize Sample Data

```bash
python GNN.py --custom-dataset --dataset-dir wear_dataset --visualize-samples
```

## Model Architecture

The GNN model implements a heterogeneous architecture that combines:

1. **Custom Edge Attention Layer**: Incorporates edge features into the attention mechanism
2. **GATv2 (Graph Attention v2)**: Provides dynamic attention mechanisms that adapt to input features
3. **GraphSAGE**: Emphasizes node feature sampling and aggregation for inductive learning
4. **GCN (Graph Convolutional Network)**: Focuses on spectral graph convolutions

The model processes node features (position, connectivity, material properties) and edge features (distance, stiffness) to predict wear at each node in the mesh.

## Training Process

- **Optimizer**: Adam with weight decay for regularization
- **Learning Rate**: Adaptive scheduling with ReduceLROnPlateau
- **Loss Function**: Mean Squared Error (MSE)
- **Early Stopping**: Prevents overfitting by monitoring validation loss

## Output and Visualization

Training produces:
- `wear_prediction_model.pt`: Saved model weights
- `training_history.png`: Plot of training and validation loss
- `progress_epoch_X.png`: Visualizations of predictions at various epochs
- `wear_prediction_visualization.png`: Final comparison of predicted vs. actual wear

For predictions, the model outputs:
- `wear_predictions.csv`: CSV file with node IDs, coordinates, and predicted wear values
- `wear_prediction.png`: Visualization of the predicted wear pattern

## Results

The model achieves excellent performance metrics on test data:
- Mean Squared Error (MSE): ~0.000001
- Mean Absolute Error (MAE): ~0.000756
- Root Mean Squared Error (RMSE): ~0.000801

## Citation

If you use this code for your research, please cite:


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

### Common Issues

1. **NumPy Version Conflict**: If you encounter NumPy version errors, downgrade to a 1.x version:
   ```bash
   pip install numpy==1.24.4
   ```

2. **Matplotlib 3D Projection Error**: If you encounter issues with 3D plots:
   ```bash
   # Add this import explicitly
   from mpl_toolkits.mplot3d import Axes3D
   ```

3. **CUDA Out of Memory**: Reduce batch size to 1 if you encounter memory issues:
   ```bash
   python GNN.py --custom-dataset --dataset-dir wear_dataset --batch-size 1
   ```

4. **DataLoader Deprecation Warning**: Update imports to use the new path:
   ```python
   from torch_geometric.loader import DataLoader
   ```

## Contact

For questions or support, please contact:
- Email: morrisdarren357@gmail.com
- GitHub: (https://github.com/DARREN-2000/)
