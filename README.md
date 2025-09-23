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
* Invalid gradient computations lead to gradient checking issuing a lot of errors, which validates gradients checking itself. Those errors usually are visible in the first 5 epochs.
* Once gradient checking, loss function, forward and backward propagations are fixed, we see that gradient checking is nearly constant (around 1e-7) for all kinds of datasets, optimizers, minibatches sizes and models being used.
* Adding gradient checking severly impacts performance, as forward propagation is run an additional 2 * nbr_gradient_checks_samples * nbr_parameters times.
* Adding numerical instability checks severly impacts performance as well.

### One layer model on numbers dataset

Using the one-layer model on the numbers model validates the following:
* The learning is quite slow using the constant optimizer, but still gets better and better up to an accuracy of 20% at epoch 100.
* The learning is more noisy (cost function is doing bounces) but much faster with the Adam optimizer, up to an accuracy of 57% at epoch 100.
* In both cases there is nearly no variance between training and dev sets.
* When using the Adam optimizer with a bigger learning rate (0.002) we see numerical instability.
* When using the N-layer model with 0 hidden layers and without BatchNormalization layers, we see the exact same behavior, which validates the computation of softmax gradients without the dz = a - y shortcut done in OneLayer model.
* We see that removing gradient checking is not modifying any result, proving that gradient checking does not leak in computations.

### N layer model on colors dataset

* We see that using BatchNormalization layers allow the Adam optimizer to be used without numerical instability.
* We see that the Adam optimizer converges more slowly on simple datasets like the colors one, but gets better results than the Constant optimizer on complex datasets like the numbers one.

### N layer model on numbers dataset

* [A] We see that just adding the BatchNormalization layer allows the Adam optimizer to be less noisy (cost function is decreasing globally without big bounces) and more quickly converge towards the optimum, reaching 58% accuracy at epoch 37, and 92% (with variance 3% with dev set) at epoch 100. Those figures were obtained without adding any hidden layer in the model.
* [B] We see that using minibatches the convergence is more noisy but accuracy gets high faster (92% was reached around epoch 70-75). However the final accuracy is around the same as without minibatches, with less variance (around 2% with the dev set).
* [C] Adding to [A] 1 hidden layer of 100 units with ReLU activation makes accuracy go up slower (epoch 37 got 50% and 58% was reached at epoch 43) and reached 85% accuracy (with 88% on dev set) at epoch 100.
* [D] Adding to [C] a BatchNormalization layer between the dense and ReLU hidden layers makes accuracy goes up faster (50% was reached at epoch 30, 85% at epoch 67 and 95% at epoch 100). Variance is nearly 0. This confirms the tendency that adding BatchNormalization layers after Dense ones make accuracy go up faster.
* [E] Replacing in [D] 1 hidden layer of 100 units with 10 hidden layers of 10 units each (with ReLU but without BatchNormalization in between them) gives really bad results. Accuracy is increasing and decreasing, the confusion matrix shows that the network is only capable of learning 2 classes at the same time, and ends with 18% accuracy at epoch 100. Cost function first decreases, but then becomes quite chaotic after epoch 50, while still globally decreasing.
* [F] Adding to [E] BatchNormalization layers to each one of the previous 10 hidden layers removes the chaos occuring after epoch 50, and smoothes a lot the accuracy curve, without increasing it (ending at 16% on epoch 100). This confirms the tendency that it's better to have a few big layers than a lot of small layers.
* [G] Using minibatches in [F] (size 50) on previous setup adds a lot of noise, makes cost globally constant and does not improve accuracy (average 15% at epoch 100). This confirms the tendency that using minibatches does not bring benefits apart from memory consumption performance.
* [H] Adding to [A] 3 hidden layers of 400, 200 and 100 units respectively with BatchNormalization and ReLU activations makes accuracy increase much slower (16% at epoch 37, 18% at epoch 50) reaching 37% at epoch 100 (nearly no variance).
* [I] Adding to [A] 1 hidden layer of 100 units with BatchNormalization and tanh activation seems to increase accuracy slower than [D]: 50% was reached at epoch 42 and 83% at epoch 100, with nearly no variance.
* [J] Changing tanh activation with sigmoid from [I] got the exact same results as with tanh.
* [K] Changing tanh activation with leaky ReLU from [I] got best results: 50% was reached at epoch 29 and 95% at epoch 100. Less than 1% of variance with dev set.

## License

See LICENSE file.
