# Layer Sensitivity Profiler

A Python library for analyzing the sensitivity of neural network layers to compression using Matrix Product States (MPS) and Time-Dependent Variational Principle (TDVP) methods.

## Features

- Analyze layer sensitivity to compression with varying bond dimensions
- Support for Linear and MultiheadAttention layers
- Automatic compression ratio calculation
- Visualization of sensitivity curves
- Unit test suite for all components

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/LayerSensitivityProfiler.git
cd LayerSensitivityProfiler
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Here's a simple example of how to use the Layer Sensitivity Profiler:

```python
import torch
from layer_sensitivity_profiler import LayerSensitivityProfiler

# Create your model
model = YourModel()

# Initialize profiler
bond_dims = [2, 4, 8, 16, 32]
profiler = LayerSensitivityProfiler(model, bond_dims)

# Create sample input
sample_input = torch.randn(batch_size, seq_length, hidden_dim)

# Profile model
results = profiler.profile_model(sample_input)

# Plot sensitivity curves
profiler.plot_sensitivity_curves('sensitivity_curves.png')
```

For a complete example, see `example_new.py`.

## Project Structure

- `layer_sensitivity_profiler.py`: Main profiler implementation
- `mps.py`: Matrix Product States implementation
- `tdvp.py`: Time-Dependent Variational Principle implementation
- `tests/`: Unit tests for all components
- `example_new.py`: Example usage with a simple transformer model

## Testing

Run the test suite:
```bash
python tests/run_tests.py
```

## Results

The profiler provides:
1. Sensitivity scores for each layer at different bond dimensions
2. Compression ratios achieved at each bond dimension
3. Visualization of sensitivity vs. bond dimension curves
4. Optimal compression settings based on sensitivity thresholds

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
