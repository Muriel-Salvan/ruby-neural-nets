# Ruby Neural Nets

A Ruby playground for implementing, coding, benchmarking, and comparing neural network techniques and libraries. This project provides a framework for building and training neural networks from scratch using Numo (Ruby's numerical array library) and RMagick for image processing.

## Features

- **Dataset Management**: Load and preprocess image datasets with support for training, development, and test splits
- **Neural Network Models**: Implement various neural network architectures (one-layer, multi-layer)
- **Training Framework**: Complete training loop with optimizers, loss functions, and accuracy metrics
- **Gradient Checking**: Built-in gradient checking to verify analytical gradients against numerical approximations
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
- Train a multi-layer neural network with gradient checking enabled
- Evaluate performance on the development set
- Display a confusion matrix

### Gradient Checking

The framework includes built-in gradient checking to verify that analytical gradients match numerical approximations. This helps ensure the correctness of gradient computations. Gradient checking is enabled by default in the test script and will raise an exception if gradients are incorrect.

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
- `lib/ruby_neural_nets/trainer.rb`: Training loop implementation with gradient checking
- `lib/ruby_neural_nets/helpers.rb`: Utility functions, numerical stability checks
- `lib/ruby_neural_nets/models/`: Specific model implementations (one-layer, multi-layer)
- `lib/ruby_neural_nets/optimizers/`: Optimization algorithms (Adam, constant learning rate, etc.)
- `lib/ruby_neural_nets/losses/`: Loss function implementations (cross-entropy, etc.)

## Contributing

This is a playground project for experimenting with neural networks in Ruby. Feel free to:
- Add new model architectures
- Implement additional optimizers or loss functions
- Experiment with different datasets
- Improve performance or add features

## Findings

### One layer model on colors dataset

The one-layer model provides:
* 1 dense layer reducing the dimensions from the input down to the number of classes used for classification,
* 1 softmax activation layer.
It expects a cross-entropy loss function to be used by the trainer, as its gradient computation will use directly dJ/dz = a - y simplification. This simplification only works when softmax activation is combined with the cross-entropy loss.

With just this model, we can already validate a lot of the framework's capabilities and various techniques:
* Normal processing validates that cost is reducing while accuracy is increasing.
* Using minibatches of small sizes validate that cost decreases in average while increasing in small steps.
* Numerical instability can be seen when the Adam optimizer is used, but not when the constant learning rate is used. Those instabilities can be solved by using the N-Layer model that also includes a Batch-normalization layer.
* Numerical instability can be seen when initialization of the parameters is done randomly instead of using Xavier Glorot's algorithm.
* Invalid gradient computations lead to gradient checking issuing a lot of errors, which validates gradients checking itself.

### One layer model on numbers dataset

Using the one-layer model on the numbers model validates the following:
* The learning is quite slow using the constant optimizer, but still gets better and better up to an accuracy of 20% at epoch 100.

## License

See LICENSE file.
