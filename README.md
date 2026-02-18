### Graph Curvature Prediction using MLP (NumPy)

Built a fully connected neural network from scratch (pure NumPy) to predict Gromov δ-hyperbolicity of synthetic graphs.

#### What This Project Does

- Generates 40-node graphs (Erdős–Rényi, Barabási–Albert, Watts–Strogatz)
- Extracts structural and spectral graph features
- Computes exact Gromov δ-hyperbolicity
- Trains an MLP using manual forward pass and backpropagation
- Compares against a closed-form linear regression baseline

#### Results

- Linear Regression: ~0.45 Test R²
- MLP (NumPy): ~0.79 Test R²

The improvement demonstrates nonlinear relationships between graph structure and curvature.

© Built from first principles. No frameworks. Just linear algebra and backpropagation.


