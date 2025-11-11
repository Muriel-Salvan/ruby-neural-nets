# Ruby Neural Nets

A Ruby playground for implementing, coding, benchmarking, and comparing neural network techniques and libraries. This project provides a framework for building and training neural networks from scratch using Numo (Ruby's numerical array library) and RMagick for image processing.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Usage](#usage)
  - [Running the Example](#running-the-example)
    - [Key Options Explained](#key-options-explained)
  - [Running Multiple Experiments](#running-multiple-experiments)
    - [Basic Multiple Experiments](#basic-multiple-experiments)
    - [Advanced Experiment Configuration](#advanced-experiment-configuration)
    - [Experiment Features](#experiment-features)
    - [Experiment Output](#experiment-output)
  - [Gradient Checking](#gradient-checking)
  - [Datasets](#datasets)
  - [Creating Custom Datasets](#creating-custom-datasets)
  - [Data Augmentation](#data-augmentation)
    - [Available Augmentation Layers](#available-augmentation-layers)
    - [Data Processing Pipeline](#data-processing-pipeline)
      - [Preprocessing Phase](#preprocessing-phase)
      - [Augmentation Phase](#augmentation-phase)
      - [Batching Phase](#batching-phase)
    - [Example Usage](#example-usage)
- [Code Structure](#code-structure)
- [Contributing](#contributing)
- [Testing](#testing)
- [Findings and experiments](#findings-and-experiments)
  - [One layer model on colors dataset](#one-layer-model-on-colors-dataset)
  - [One layer model on numbers dataset](#one-layer-model-on-numbers-dataset)
  - [N layer model on colors dataset](#n-layer-model-on-colors-dataset)
  - [N layer model on numbers dataset](#n-layer-model-on-numbers-dataset)
  - [N layer model using PyTorch](#n-layer-model-using-pytorch)
    - [Experiment [A]: Same parameters as with Numo implementation](#experiment-a-same-parameters-as-with-numo-implementation)
    - [Experiment [B]: Measuring model randomness effect](#experiment-b-measuring-model-randomness-effect)
    - [Experiment [C]: Measuring dataset randomness effect](#experiment-c-measuring-dataset-randomness-effect)
  - [Performance benchmarks](#performance-benchmarks)
    - [Ruby Numo](#ruby-numo)
    - [Torch.rb](#torchrb)
- [License](#license)

## Features

- **Experiment Management**: Run multiple experiments with different configurations in a single command, each with unique IDs and separate progress tracking
- **Layered Datasets**: Modular dataset processing framework with composable layers (partitioning, shuffling, caching, encoding, minibatching) enabling reusable features between Numo and PyTorch implementations
- **Data Augmentation**: Built-in data augmentation capabilities with Clone and modular image transformation layers for expanding datasets through duplication and various random transformations
- **TorchVision Integration**: TorchVision transforms for Ruby providing image preprocessing pipelines compatible with PyTorch workflows
- **Dataset Management**: Load and preprocess image datasets with support for training, development, and test splits using extensible data loader architecture
- **Neural Network Models**: Implement various neural network architectures (one-layer, multi-layer) with modular layers including Dense, Batch Normalization, Dropout, and activations (ReLU, Leaky ReLU, Sigmoid, Softmax, Tanh)
- **Training Framework**: Complete training loop with optimizers, loss functions, and accuracy metrics, featuring a simplified architecture with externalized GradientChecker, ProgressTracker, and Profiler components
- **Weight Decay (L2 Regularization)**: Built-in L2 regularization support across all optimizers to prevent overfitting and improve generalization
- **Gradient Checking**: Built-in gradient checking to verify analytical gradients against numerical approximations, configurable to run every n epochs
- **Profiling**: Optional epoch profiling with HTML reports generated using ruby-prof to analyze performance bottlenecks
- **OpenBLAS Linear Algebra**: Fast matrix operations powered by OpenBLAS through numo-linalg for improved computational performance
- **Visualization**: Confusion matrix plotting using Gnuplot, with real-time parameter visualization in progress tracker graphs
- **Progress tracking**: Automatic timing and summary reporting
- **Named Parameters**: Parameters include names for better identification and visualization
- **Logger Mixin**: Unified logging system with ISO8601 UTC timestamps and class name prefixes, supporting both regular and debug (lazy-evaluated) logging across all major components
- **Extensible Architecture**: Modular design for easy addition of new models, optimizers, and loss functions
- **Early Stopping**: Automatic early stopping based on development set accuracy to prevent overfitting, with configurable patience and visual markers on training graphs

## Installation

### Prerequisites

1. **Ruby**: Ensure Ruby is installed on your system
2. **ImageMagick**: Download and install ImageMagick from [https://imagemagick.org/script/download.php](https://imagemagick.org/script/download.php)
   - Make sure to include DLL and C/C++ headers during installation
   - Use the Q16-x64-dll.exe version, not the HDRI or static version (details [https://github.com/rmagick/rmagick?tab=readme-ov-file#windows](here)).
3. **Bundler**: Install Bundler if not already available: `gem install bundler`
4. **libTorch**: Download the C++ library from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/), and configure bundler to use it: `bundle config set build.torch-rb --with-torch-dir=/path/to/libtorch-shared-with-deps-2.9.0+cu126/libtorch/`
5. **OpenBLAS** (optional): For improved matrix computation performance, install OpenBLAS and set the `OPEN_BLAS_PATH` environment variable to the path containing the OpenBLAS library files (e.g., `OPEN_BLAS_PATH=/path/to/openblas/lib`). If not set, the framework will run without OpenBLAS acceleration.

### Setup

1. Clone this repository
2. Navigate to the project directory
3. Install dependencies:
   ```bash
   bundle install
   ```

## Usage

### Running the Example

The `run` tool provides a comprehensive CLI interface for configuring and running neural network training experiments. Use `bundle exec ruby bin/run --help` to see all available options.

Run the following command to see a complete example of training a neural network on the numbers dataset:

```bash
bundle exec ruby bin/run --dataset=numbers
```

This runs with default settings:
- **Dataset**: Handwritten digits (0-9) from the numbers dataset
- **Model**: OneLayer (simple softmax classification)
- **Optimizer**: Adam with learning rate 0.001
- **Training**: 100 epochs, full batch (no minibatches)
- **Accuracy**: ClassesNumo (standard accuracy measurement)
- **Loss**: CrossEntropy
- **Checks**: Gradient checking enabled (exceptions on failure), instability checks via byebug

#### Key Options Explained

- **`--dataset`**: Choose dataset (colors, numbers)
  - Controls which dataset to load and train on

- **`--model`**: Select neural network architecture (OneLayer, NLayers, etc.)
  - Controls the model type and layer configuration
  - Use `--layers` to specify hidden layer sizes (comma-separated integers)

- **`--optimizer`**: Choose optimization algorithm (Adam, Constant, ExponentialDecay, etc.)
  - Controls how model parameters are updated during training
  - Use `--learning-rate` to set learning rate, `--decay` for decay-based optimizers, `--weight-decay` for L2 regularization (default: 0.0)

- **`--data-loader`**: Select data loading method (NumoImageMagick, NumoVips, TorchVips, TorchImageMagick)
  - Controls the dataset processing pipeline (partitioning, shuffling, caching, encoding, minibatching)

- **`--loss`**: Choose loss function (CrossEntropy, CrossEntropyTorch)
  - Defines the training objective function

- **`--accuracy`**: Choose accuracy metric (ClassesNumo, ClassesTorch)
  - Defines how predictions are evaluated

- **`--nbr-epochs`**: Number of training iterations (default: 100)
- **`--max-minibatch-size`**: Mini-batch size for memory-limited training (default: 5000, full batch)
- **`--gradient-checks`**: Enable/disable automatic gradient verification (byebug/exception/off/warning)
- **`--instability-checks`**: Enable/disable numerical instability monitoring (same options)
- **`--profiling`**: Enable performance profiling with HTML reports (boolean, default: false)
- **`--debug`**: Enable debug mode for verbose logging output (boolean, default: false)
  - When enabled, shows detailed debug messages from model forward/backward propagation and other internal operations
  - Debug messages use lazy evaluation to avoid performance overhead when disabled
- **`--display-graphs`**: Display GnuPlot graphs during training (boolean, default: true)
  - Controls whether to show real-time visualization of training progress, cost, accuracy, and parameter evolution
  - When disabled, training runs faster without graphical output
  - Useful for headless environments or when visualization is not needed
- **`--early-stopping-patience`**: Number of epochs to wait for development set loss improvement before stopping training (integer, default: 10)
  - Monitors development set loss (including L2 regularization) and notifies when no improvement occurs for the specified number of epochs
  - Training continues, but a red circle marker is placed on the graphs at the early stopping epoch
  - Use `--eval-dev` to enable development set evaluation (required for early stopping)
  - Example: `--early-stopping-patience 5` will notify early stopping after 5 epochs without improvement

- **`--model-seed`**: Random number generator seed for model initialization and parameters (integer, default: 0)
  - Controls the randomness in model parameter initialization
  - Use specific seeds for reproducible model initialization across runs

- **`--dataset-seed`**: Random number generator seed for dataset shuffling and data order (integer, default: 0)
  - Controls the randomness in dataset shuffling and data ordering
  - Use specific seeds for reproducible data loading and shuffling across runs

- **`--nbr-clones`**: Number of times each element should be cloned in the Clone dataset wrapper layer (integer, default: 1)
  - Controls data augmentation through sample duplication
  - Use values > 1 to expand dataset size by duplicating each sample multiple times
  - Useful for balancing datasets or increasing training data volume

- **`--rot-angle`**: Maximum rotation angle in degrees for random image transformations (integer, default: 0)
  - Controls random rotation data augmentation (rotation between -angle and +angle)
  - Use values > 0 to enable random image rotations during training
  - Helps improve model robustness to image orientation variations

- **`--resize`**: Resize dimensions [width, height] for image transformations (integer,integer, default: 110,110)
  - Controls image resizing as part of data augmentation
  - Use to change image dimensions before training
  - Example: `--resize 64,64` resizes all images to 64x64 pixels

- **`--noise-intensity`**: Intensity of Gaussian noise for random image transformations (float, default: 0)
  - Controls the standard deviation of Gaussian noise added to images for data augmentation
  - Use values > 0 to enable Gaussian noise data augmentation
  - Example: `--noise-intensity 0.1` adds Gaussian noise with scaled standard deviation

- **`--grayscale`**: Convert images to grayscale, reducing the number of channels from 3 to 1 (boolean, default: false)
  - Reduces channel count from RGB to grayscale during preprocessing
  - When enabled, applies transformation before resizing for reduced memory usage

- **`--adaptive-invert`**: Apply adaptive color inversion based on top left pixel intensity (boolean, default: false)
  - Inverts image colors if the top left pixel has intensity in the lower half range (< 0.5)
  - Applied after grayscale conversion if both options are enabled
  - Helps improve model robustness to inverted color schemes

- **`--trim`**: Trim images to remove borders and restore original aspect ratio by adding borders with the color of pixel 0,0 (boolean, default: false)
  - Applies trimming before resizing to maintain aspect ratio consistency
  - Useful for datasets with varying border sizes around content

- **`--minmax-normalize`**: Scale image data to always be within the range 0 to 1 (boolean, default: false)
  - Applies min-max normalization to pixel values during preprocessing, after possible grayscale conversion

- **`--track-layer`**: Specify a layer name to be tracked for a given number of hidden units (string,integer, can be used multiple times)
  - Allows monitoring specific layer parameters during training
  - Format: `--track-layer layer_name,num_units`
  - Example: `--track-layer L0_Dense_W,10 --track-layer L4_Dense_W,10`
  - Useful for visualizing parameter evolution in specific layers during training

- **`--display-samples`**: Number of samples to display in the progress graph (integer, default: 0)
  - Shows input samples from each minibatch as grid visualizations using GnuPlot
  - Format: `--display-samples num_samples`
  - Example: `--display-samples 4` displays a grid of 4 input samples per progress update
  - Useful for monitoring what the model is training on and verifying data loading
  - Applies to both training and development set evaluations when `--eval-dev` is enabled

- **`--dropout-rate`**: Dropout rate for regularization in NLayers model (float, default: 0.0)
  - Controls the fraction of units to drop during training to prevent overfitting
  - Use values between 0.0 (no dropout) and 1.0 (drop all units)
  - Example: `--dropout-rate 0.5` drops 50% of units randomly during training

- **`--experiment`**: Start a new experiment configuration (can be used multiple times)
  - Allows running multiple experiments with different configurations in a single command
  - Each experiment can have its own unique ID and configuration
  - Example: `--experiment --exp-id=exp1 --dataset=colors --experiment --exp-id=exp2 --dataset=numbers`
  - This will run two separate experiments: exp1 (colors dataset) and exp2 (numbers dataset)

- **`--exp-id`**: Set experiment ID for identification (string, default: 'main')
  - Provides a unique identifier for each experiment when running multiple experiments
  - Useful for distinguishing between different experiment runs in output and visualizations
  - Example: `--exp-id=baseline` sets the experiment ID to 'baseline'

- **`--training-times`**: Number of times to repeat training with the same configuration (integer, default: 1)
  - Runs the same experiment multiple times to measure variance and consistency
  - Useful for statistical analysis of model performance across multiple runs

- **`--eval-dev`**: Evaluate model on development set during training (boolean, default: true)
  - Controls whether to evaluate the model on the development set after each epoch
  - Required for early stopping and development set accuracy tracking

- **`--dump-minibatches`**: Save minibatches as image files to disk (boolean, default: false)
  - Outputs each minibatch as individual image files for debugging and visualization
  - Useful for inspecting data preprocessing and augmentation effects

The run will:
- Load the specified dataset and display statistics
- Train the neural network with selected configuration
- Show training progress, including cost, accuracy, and optional parameter visualizations
- Evaluate final performance on the development set
- Display a confusion matrix showing prediction accuracy per class

### Running Multiple Experiments

The framework supports running multiple experiments with different configurations in a single command using the `--experiment` flag. Each experiment can have its own unique configuration and will be tracked separately with its own progress visualization.

#### Basic Multiple Experiments

To run multiple experiments, separate them with the `--experiment` flag:

```bash
bundle exec ruby bin/run \
  --exp-id=experiment1 --dataset=colors --model=OneLayer --optimizer=Constant \
  --experiment \
  --exp-id=experiment2 --dataset=numbers --model=NLayers --optimizer=Adam
```

This will run two separate experiments:
- **experiment1**: OneLayer model on colors dataset with Constant optimizer
- **experiment2**: NLayers model on numbers dataset with Adam optimizer

#### Advanced Experiment Configuration

Each experiment can have completely different configurations:

```bash
bundle exec ruby bin/run \
  --exp-id=baseline --dataset=numbers --model=OneLayer --optimizer=Constant --nbr-epochs=50 \
  --experiment \
  --exp-id=optimized --dataset=numbers --model=NLayers --optimizer=Adam --learning-rate=0.01 --layers=100,50 \
  --experiment \
  --exp-id=comparison --dataset=colors --model=NLayers --optimizer=ExponentialDecay --decay=0.95
```

#### Experiment Features

- **Unique IDs**: Each experiment gets a unique identifier (with automatic suffixing for duplicates)
- **Separate Progress Tracking**: Each experiment has its own cost/accuracy curves and confusion matrices
- **Multiple Training Runs**: Use `--training-times` to run the same experiment configuration multiple times
- **Development Set Evaluation**: Each experiment can optionally evaluate on the development set using `--eval-dev`
- **Parameter Tracking**: Use `--track-layer` to visualize specific layer parameters for each experiment

#### Experiment Output

When running multiple experiments, you'll see:
- Combined progress display with experiment IDs: `[Epoch X] [Exp experiment1] [Minibatch Y] - Cost Z, Training accuracy W%`
- Separate graphs for each experiment's cost, accuracy, and confusion matrix
- Individual parameter visualizations when using `--track-layer`
- Final evaluation results for each experiment

This allows for easy comparison of different architectures, hyperparameters, and training strategies in a single run.

### Gradient Checking

The framework includes built-in gradient checking to verify that analytical gradients match numerical approximations. This helps ensure the correctness of gradient computations. Gradient checking is enabled by default in the test script and will raise an exception if gradients are incorrect.

### Datasets

The project can use datasets in the `datasets/` directory, having the structure `datasets/<dataset_name>/<class_name>/<image_name>.png`.

The following datasets can be used easily:
- **colors**: Classification of colored images (red, green, blue)
- **numbers**: Handwritten digit recognition (0-9), downloaded from https://www.kaggle.com/datasets/ox0000dead/numbers-classification-dataset

Each dataset consists of PNG images organized in class subdirectories.

### Creating Custom Datasets

To use your own dataset:
1. Create a new directory under `datasets/`
2. Organize images in subdirectories named after their classes
3. Ensure all images are in PNG format
4. Instantiate the dataset in your code: `RubyNeuralNets::Dataset.new('your_dataset_name')`

### Data Augmentation

The framework includes built-in data augmentation capabilities through composable dataset wrapper layers. These layers can be configured via command-line options and are automatically applied to the Numo data loader pipeline.

#### Available Augmentation Layers

**Clone Layer** (`--nbr-clones`)
- Duplicates each dataset element multiple times
- Useful for expanding small datasets or balancing class distributions
- Example: `--nbr-clones 3` triples the dataset size by duplicating each sample 3 times

**Image Transformation Layers** (`--rot-angle`, `--resize`, `--noise-intensity`)
- Apply random image transformations using ImageMagick through separate specialized layers:
  - **ImageResize**: Resizes images to target dimensions
  - **ImageRotate**: Applies random rotations between `-angle` and `+angle` degrees
  - **ImageCrop**: Crops images back to target dimensions after transformations, centered
  - **ImageNoise**: Adds Gaussian noise to images
- Example: `--rot-angle 45 --resize 64,64 --noise-intensity 0.1` enables resizing to 64x64, random rotations up to 45 degrees, and adds Gaussian noise
- The layers are applied in order: Resize → Rotate → Crop → Noise to maintain consistent transformations
- Helps improve model robustness to image orientation, scale variations, and noise

#### Data Processing Pipeline

The Numo data loader splits data processing into three phases that work together to prepare data for training:

##### Preprocessing Phase
Applied once for each dataset partition (training, dev, test), these are deterministic transformations that can be cached for performance:
1. **ImagesFromFiles**: Load images from disk as ImageMagick objects
2. **OneHotEncoder**: Convert labels to one-hot encoding
3. **ImageResize**: Resize images to target dimensions
4. **CacheMemory**: Cache processed data in memory for faster access

##### Augmentation Phase
Applied only during training, these are random transformations that increase dataset variety and improve model robustness:
1. **Clone**: Duplicate samples (if `--nbr-clones > 1`)
2. **ImageRotate**: Apply random rotations (if `--rot-angle > 0`)
3. **ImageCrop**: Crop images back to target dimensions after transformations
4. **ImageNoise**: Add Gaussian noise (if `--noise-intensity > 0`)

##### Batching Phase
Applied to prepare final training batches, these handle shuffling and data grouping:
1. **ImageToNumo**: Convert images to flattened Numo DFloat arrays (raw pixel values)
2. **NumoNormalize**: Normalize Numo arrays to [0,1] range by dividing by a factor (255 for Vips, 65535 for ImageMagick)
3. **EpochShuffler**: Shuffle data between epochs
4. **Minibatch**: Group data into minibatches

#### Example Usage

```bash
# Basic data augmentation
bundle exec ruby bin/run --dataset=numbers --nbr-clones 2 --rot-angle 30

# Aggressive augmentation for small datasets
bundle exec ruby bin/run --dataset=colors --nbr-clones 5 --rot-angle 90 --resize 64,64

# Augmentation with noise for robustness
bundle exec ruby bin/run --dataset=numbers --noise-intensity 0.1 --rot-angle 45
```

This configuration will:
- Load each image 5 times (once original + 4 duplicates)
- Apply random rotation between -90° and +90° to each image
- Resize all images to 64x64 pixels
- Result in a 5x larger dataset with varied image orientations and sizes

For the noise example:
- Add Gaussian noise with intensity 0.1 to each image
- Apply random rotation between -45° and +45°
- Helps improve model robustness to noise and orientation variations

### Code Structure

- `lib/ruby_neural_nets/accuracy.rb`: Base accuracy measurement class
- `lib/ruby_neural_nets/accuracies/`: Accuracy metric implementations (ClassesNumo, ClassesTorch)
- `lib/ruby_neural_nets/data_loader.rb`: Base data loader framework
- `lib/ruby_neural_nets/data_loaders/`: Data loader implementations (NumoImageMagick, NumoVips, Torch) configuring layered dataset processing
- `lib/ruby_neural_nets/dataset.rb`: Base dataset class
- `lib/ruby_neural_nets/datasets/`: Dataset processing layers (Wrapper, Partitioning, Shuffling, Caching, Encoding, Minibatching, Image transformations, Data augmentation)
- `lib/ruby_neural_nets/experiment.rb`: Experiment management system for running multiple configurations
- `lib/ruby_neural_nets/gradient_checker.rb`: Gradient checking for validation
- `lib/ruby_neural_nets/helpers.rb`: Utility functions and numerical stability checks
- `lib/ruby_neural_nets/initializers/`: Parameter initialization algorithms (Glorot, Rand, Zero, One)
- `lib/ruby_neural_nets/logger.rb`: Logger mixin providing timestamped logging with lazy-evaluated debug messages
- `lib/ruby_neural_nets/loss.rb`: Base loss function class
- `lib/ruby_neural_nets/losses/`: Loss function implementations (CrossEntropy, CrossEntropyTorch)
- `lib/ruby_neural_nets/model.rb`: Base model class for neural networks
- `lib/ruby_neural_nets/models/`: Specific model implementations (OneLayer, NLayers, NLayersTorch)
  - `lib/ruby_neural_nets/models/layers/`: Individual neural network layers (Dense, BatchNormalization, Dropout, ReLU, Sigmoid, Softmax, Tanh, LeakyReLU)
  - `lib/ruby_neural_nets/models/activation_layer.rb`: Activation layer implementation
  - `lib/ruby_neural_nets/models/layer.rb`: Base layer class
- `lib/ruby_neural_nets/optimizer.rb`: Base optimizer class
- `lib/ruby_neural_nets/optimizers/`: Optimization algorithms (Adam, Constant, ExponentialDecay, AdamTorch)
- `lib/ruby_neural_nets/options.rb`: Command-line options parsing and class discovery
- `lib/ruby_neural_nets/parameter.rb`: Parameter management with optimization integration
- `lib/ruby_neural_nets/parameters/`: Parameter implementations (Torch)
- `lib/ruby_neural_nets/profiler.rb`: Performance profiling with HTML reports
- `lib/ruby_neural_nets/progress_tracker.rb`: Training progress visualization and tracking
- `lib/ruby_neural_nets/trainer.rb`: Training loop implementation with gradient checking
- `lib/ruby_neural_nets/torch/`: PyTorch integration utilities
- `lib/ruby_neural_nets/torchvision/`: TorchVision transforms for Ruby
- `lib/ruby_neural_nets/torchvision/transforms/`: Individual TorchVision transform implementations
- `lib/ruby_neural_nets/transform_helpers/`: Shared image transformation utilities used across dataset layers

## Contributing

This is a playground project for experimenting with neural networks in Ruby. Feel free to:
- Add new model architectures
- Implement additional optimizers or loss functions
- Experiment with different datasets
- Improve performance or add features

## Testing

The project uses RSpec for unit testing to ensure the correctness of neural network implementations and training components.

### Running Tests

To run the test suite:

```bash
bundle exec rspec
```

This will execute all tests in the `spec/` directory.

### Test Structure

- **`spec/trainer_spec.rb`**: Tests the Trainer class functionality
  - Verifies training progress tracking and reporting
  - Tests cost and accuracy recording across epochs and minibatches
  - Uses mocked file access and synthetic datasets for reliable testing

### Test Features

- **Mocked File System**: Tests use mocked file system operations to avoid dependencies on actual dataset files
- **Synthetic Data**: Generates deterministic test data for consistent test results
- **Progress Validation**: Ensures training progress is properly tracked and reported
- **Component Integration**: Tests the integration between Trainer, ProgressTracker, and Experiment classes

### Writing Tests

When adding new tests:
1. Place test files in the `spec/` directory with `_spec.rb` suffix
2. Use descriptive test names and contexts
3. Mock external dependencies (file system, random data) for reliable execution
4. Test both success and failure scenarios
5. Verify numerical outputs are within expected ranges

## Findings and experiments

### One layer model on colors dataset

```bash
bundle exec ruby ./bin/run --dataset=colors --data-loader=NumoImageMagick --accuracy=ClassesNumo --model=OneLayer --optimizer=Constant
```

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

Performance:
* After 7 epochs, accuracy for both training and dev sets stay at 100%.

### One layer model on numbers dataset

```bash
bundle exec ruby ./bin/run --dataset=numbers --data-loader=NumoImageMagick --accuracy=ClassesNumo --model=OneLayer --optimizer=Constant
```

Using the one-layer model on the numbers model validates the following:
* When using the Adam optimizer with a bigger learning rate (0.002) we see that accuracy and cost keep increasing and decreasing. That means the model can't learn anymore certainly due to gradient descent overshooting constantly over the minimum.
* When using the Adam optimizer with a bigger learning rate (0.003) we see numerical instability starting the third epoch.
* When using the N-layer model with 0 hidden layers (`--model=NLayers --layers=`), without BatchNormalization layers and with bias in Dense layers, we see the exact same behavior, which validates the computation of softmax gradients without the dz = a - y shortcut done in OneLayer model.
* We see that removing gradient checking is not modifying any result, proving that gradient checking does not leak in computations.

Performance:
* The learning is quite slow using the constant optimizer, but still gets better and better up to an accuracy of 20% around epoch 80 that stagnates afterwards. Variance is about 4%, meaning the model has difficulty to learn but tends to overfit a bit the training set.
* The learning is more noisy (cost function is doing bounces) but much faster with the Adam optimizer, up to an accuracy of 55% at epoch 100. However we see variance increasing a lot starting epoch 60, as dev accuracy stays around 20%. This confirms the overfitting that is also visible with the constant optimizer.

### N layer model on colors dataset

```bash
bundle exec ruby ./bin/run --dataset=colors --data-loader=NumoImageMagick --accuracy=ClassesNumo --model=NLayers --optimizer=Adam
```

Observations:
* We see that using BatchNormalization layers allow the Adam optimizer to be used without numerical instability.

Performance:
* We see that the Adam optimizer converges more slowly on simple datasets like the colors one (both dev and training sets have 100% accuracy on epoch 59 instead of 7), but gets better results than the Constant optimizer on complex datasets like the numbers one.

### N layer model on numbers dataset

See [docs/n_layer_model_on_numbers.md](docs/n_layer_model_on_numbers.md) for details.

### N layer model using PyTorch

#### Experiment [A]: Same parameters as with Numo implementation

```bash
bundle exec ruby ./bin/run -dataset=numbers --accuracy=ClassesTorch --model=NLayersTorch --optimizer=AdamTorch --loss=CrossEntropyTorch --gradient-checks=off --data-loader=TorchVips --instability-checks=off --grayscale=true --minmax-normalize=true --adaptive-invert=true --trim=true --resize=16,16 --noise-intensity=0 --rot-angle=0 --nbr-clones=1 --layers=16 --max-minibatch-size=100000 --learning-rate=1e-2 --dropout=0 --weight-decay=0.00 --display-samples=4 --track-layer=l0_linear.weight,8
```

![A](docs/torch_numbers/a.png)

* We see the same behaviour as with Numo implementation, with a smaller dev accuracy (94%), resulting in 6% variance.
* After using 64 bit floats in Torch, the accuracy goes up much faster.
* We can check that using ::Torch::NN::LogSoftmax layer with ::Torch::NN::NLLLoss loss is equivalent (but slower) than not using this last layer with ::Torch::NN::CrossEntropyLoss.
* We see that normalizing inputs between -1 and 1 with mean 0 instead of 0 and 1 with mean 0.5 does not change the preformance of the training.

#### Experiment [B]: Measuring model randomness effect

```bash
bundle exec ruby bin/run --dataset=numbers --accuracy=ClassesTorch --model=NLayersTorch --optimizer=AdamTorch --loss=CrossEntropyTorch --gradient-checks=off --data-loader=TorchVips --instability-checks=off --grayscale=true --minmax-normalize=true --adaptive-invert=true --trim=true --resize=16,16 --layers=16 --max-minibatch-size=100000 --learning-rate=1e-2 --training-times=5
```

![B](docs/torch_numbers/b.png)

* Different seeds produce a variance of around 3% on dev accuracy, and less than 1% on training accuracy.

#### Experiment [C]: Measuring dataset randomness effect

```bash
bundle exec ruby bin/run --instability-checks=off --dataset=numbers --accuracy=ClassesTorch --model=NLayersTorch --optimizer=AdamTorch --loss=CrossEntropyTorch --gradient-checks=off --data-loader=TorchVips --grayscale=true --minmax-normalize=true --adaptive-invert=true --trim=true --resize=16,16 --layers=16 --max-minibatch-size=100000 --learning-rate=1e-2 --experiment --dataset=numbers --accuracy=ClassesTorch --model=NLayersTorch --optimizer=AdamTorch --loss=CrossEntropyTorch --gradient-checks=off --data-loader=TorchVips --grayscale=true --minmax-normalize=true --adaptive-invert=true --trim=true --resize=16,16 --layers=16 --max-minibatch-size=100000 --learning-rate=1e-2 --experiment --dataset=numbers --accuracy=ClassesTorch --model=NLayersTorch --optimizer=AdamTorch --loss=CrossEntropyTorch --gradient-checks=off --data-loader=TorchVips --grayscale=true --minmax-normalize=true --adaptive-invert=true --trim=true --resize=16,16 --layers=16 --max-minibatch-size=100000 --learning-rate=1e-2 --experiment --dataset=numbers --accuracy=ClassesTorch --model=NLayersTorch --optimizer=AdamTorch --loss=CrossEntropyTorch --gradient-checks=off --data-loader=TorchVips --grayscale=true --minmax-normalize=true --adaptive-invert=true --trim=true --resize=16,16 --layers=16 --max-minibatch-size=100000 --learning-rate=1e-2
```

![C](docs/torch_numbers/c.png)

* Different seeds produce a variance of around 3% on dev accuracy, and less than 1% on training accuracy.

### Performance benchmarks

The benchmarks are made on CPU, under VirtualBox kubuntu, using 100 epochs on training on numbers dataset (caching all data preparation in memory), with 1 layer of 100 units, without minibatches.
Absolute values are meaningless as this setup is far from being optimal. However relative values give some comparison ideas between frameworks and algorithms, on the training part.

Here are the command lines used:

```bash
# Numo using Vips
bundle exec ruby ./bin/run --dataset=numbers --gradient-checks=off --instability-checks=off --grayscale=true --minmax-normalize=true --trim=true --resize=110,110 --data-loader=NumoVips --accuracy=ClassesNumo --model=NLayers --optimizer=Adam --loss=CrossEntropy
# Numo using ImageMagick
bundle exec ruby ./bin/run --dataset=numbers --gradient-checks=off --instability-checks=off --grayscale=true --minmax-normalize=true --trim=true --resize=110,110 --data-loader=NumoImageMagick --accuracy=ClassesNumo --model=NLayers --optimizer=Adam --loss=CrossEntropy
# Torch using Vips
bundle exec ruby ./bin/run --dataset=numbers --gradient-checks=off --instability-checks=off --grayscale=true --minmax-normalize=true --trim=true --resize=110,110 --data-loader=TorchVips --accuracy=ClassesTorch --model=NLayersTorch --optimizer=AdamTorch --loss=CrossEntropyTorch
# Torch using ImageMagick
bundle exec ruby ./bin/run --dataset=numbers --gradient-checks=off --instability-checks=off --grayscale=true --minmax-normalize=true --trim=true --resize=110,110 --data-loader=TorchImageMagick --accuracy=ClassesTorch --model=NLayersTorch --optimizer=AdamTorch --loss=CrossEntropyTorch
```

| Experiment              | Elapsed time | Memory consumption (GB) | Final dev accuracy |
| ----------------------- | ------------ | ----------------------- | ------------------ |
| Torch using ImageMagick | 3m 31s       | 850                     | 95%                |
| Numo using ImageMagick  | 3m 47s       | 750                     | 96%                |
| Numo using Vips         | 5m 16s       | 750                     | 90%                |
| Torch using Vips        | 5m 23s       | 850                     | 93%                |

Analysis: Overall performance is consistent between experiments:
* ImageMagick processing is more efficient than Vips (big factor), and results in better accuracy.
* Torch is slightly performing better than Numo Ruby implementation (small factor).

## License

See LICENSE file.
Here are the command lines used:

```bash
# Numo using Vips
bundle exec ruby ./bin/run --dataset=numbers --data-loader=NumoVips --accuracy=ClassesNumo --model=NLayers --optimizer=Adam --gradient-checks=off --instability-checks=off --grayscale=true --minmax-normalize=true --trim=true
# Numo using ImageMagick
bundle exec ruby ./bin/run --dataset=numbers --data-loader=NumoImageMagick --accuracy=ClassesNumo --model=NLayers --optimizer=Adam --gradient-checks=off --instability-checks=off --grayscale=true --minmax-normalize=true --trim=true
# Torch using Vips
bundle exec ruby ./bin/run --dataset=numbers --data-loader=TorchVips --accuracy=ClassesTorch --model=NLayersTorch --optimizer=AdamTorch --loss=CrossEntropyTorch --gradient-checks=off --instability-checks=off --grayscale=true --minmax-normalize=true --trim=true
```

| Experiment             | Elapsed time | Memory consumption (GB) | Final dev accuracy |
| ---------------------- | ------------ | ----------------------- | ------------------ |
| Numo using Vips        | 15m 12s      | 0.9                     | 90%                |
| Numo using ImageMagick | 2m 24s       | 0.9                     | 96%                |
| Torch using Vips       | 3m 28s       | 1.0                     | 93%                |

## License

See LICENSE file.
