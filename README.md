# Ruby Neural Nets

A Ruby playground for implementing, coding, benchmarking, and comparing neural network techniques and libraries. This project provides a framework for building and training neural networks from scratch using Numo (Ruby's numerical array library) and RMagick for image processing.

## Features

- **Dataset Management**: Load and preprocess image datasets with support for training, development, and test splits
- **Neural Network Models**: Implement various neural network architectures (one-layer, multi-layer)
- **Training Framework**: Complete training loop with optimizers, loss functions, and accuracy metrics
- **Visualization**: Confusion matrix plotting using Gnuplot
- **Extensible Architecture**: Modular design for easy addition of new models, optimizers, and loss functions

## Installation

### Prerequisites

1. **Ruby**: Ensure Ruby is installed on your system
2. **ImageMagick**: Download and install ImageMagick from [https://imagemagick.org/script/download.php](https://imagemagick.org/script/download.php)
   - Make sure to include DLL and C/C++ headers during installation
3. **Bundler**: Install Bundler if not already available: `gem install bundler`

### Setup

1. Clone this repository
2. Navigate to the project directory
3. Install dependencies:
   ```bash
   bundle install
   ```

## Usage

### Running the Example

Execute the test script to see a complete example of training a neural network on the numbers dataset:

```bash
bundle exec ruby bin/test
```

This will:
- Load the numbers dataset
- Display dataset statistics
- Train a multi-layer neural network
- Evaluate performance on the development set
- Display a confusion matrix

### Datasets

The project can use datasets in the `dataset/` directory, having the structure `dataset/<dataset_name>/<class_name>/<image_name>.png`.

The following datasets can be used easily:
- **colors**: Classification of colored images (red, green, blue)
- **numbers**: Handwritten digit recognition (0-9), downloaded from https://www.kaggle.com/datasets/ox0000dead/numbers-classification-dataset

Each dataset consists of PNG images organized in class subdirectories.

### Creating Custom Datasets

To use your own dataset:
1. Create a new directory under `dataset/`
2. Organize images in subdirectories named after their classes
3. Ensure all images are in PNG format
4. Instantiate the dataset in your code: `RubyNeuralNets::Dataset.new('your_dataset_name')`

### Code Structure

- `lib/ruby_neural_nets/dataset.rb`: Dataset loading and preprocessing
- `lib/ruby_neural_nets/model.rb`: Base model class
- `lib/ruby_neural_nets/trainer.rb`: Training loop implementation
- `lib/ruby_neural_nets/models/`: Specific model implementations
- `lib/ruby_neural_nets/optimizers/`: Optimization algorithms
- `lib/ruby_neural_nets/losses/`: Loss function implementations

## Contributing

This is a playground project for experimenting with neural networks in Ruby. Feel free to:
- Add new model architectures
- Implement additional optimizers or loss functions
- Experiment with different datasets
- Improve performance or add features

## License

See LICENSE file.
