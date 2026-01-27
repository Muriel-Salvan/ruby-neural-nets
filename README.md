# Ruby Neural Nets

A Ruby playground for implementing, coding, benchmarking, and comparing neural network techniques and libraries. This project provides a framework for building and training neural networks from scratch using Numo (Ruby's numerical array library) and RMagick for image processing.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Usage](#usage)
  - [Training models](#training-models)
    - [Key Options Explained](#key-options-explained)
  - [Running Multiple Experiments](#running-multiple-experiments)
    - [Basic Multiple Experiments](#basic-multiple-experiments)
    - [Advanced Experiment Configuration](#advanced-experiment-configuration)
    - [Experiment Features](#experiment-features)
    - [Experiment Output](#experiment-output)
  - [Gradient Checking](#gradient-checking)
  - [ONNX Model Generation](#onnx-model-generation)
    - [Generating ONNX Models](#generating-onnx-models)
    - [ONNX Model Structure](#onnx-model-structure)
    - [Usage with Other Frameworks](#usage-with-other-frameworks)
  - [Inference](#inference)
    - [Running Inference](#running-inference)
    - [Inference Features](#inference-features)
  - [Datasets](#datasets)
    - [Video Processing](#video-processing)
  - [Creating Custom Datasets](#creating-custom-datasets)
    - [Video Dataset Example](#video-dataset-example)
  - [Data Augmentation](#data-augmentation)
    - [Available Augmentation Layers](#available-augmentation-layers)
    - [Data Processing Pipeline](#data-processing-pipeline)
      - [Preprocessing Phase](#preprocessing-phase)
      - [Augmentation Phase](#augmentation-phase)
      - [Batching Phase](#batching-phase)
    - [Example Usage](#example-usage)
- [Findings and experiments](#findings-and-experiments)
- [Contributing](#contributing)
  - [Code Structure](#code-structure)
  - [Testing](#testing)
    - [Running Tests](#running-tests)
    - [Test Structure](#test-structure)
    - [Test Features](#test-features)
    - [Writing Tests](#writing-tests)
    - [Regenerating onnx_pb.rb](#regenerating-onnx_pbrb)
- [Building the Docker build image](#building-the-docker-build-image)
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
- **ONNX Model Generation**: Generate ONNX (Open Neural Network Exchange) model files for deployment in other deep learning frameworks
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

1. **Ruby**: Ensure Ruby is installed on your system.
2. Install some Rubygems in your Ruby installation:
   - **Bundler**: `gem install bundler`.
3. **ImageMagick**: Download and install ImageMagick from [https://imagemagick.org/script/download.php](https://imagemagick.org/script/download.php)
   - Make sure to include DLL and C/C++ headers during installation.
   - Use the Q16-x64-dll.exe version, not the HDRI or static version (details [https://github.com/rmagick/rmagick?tab=readme-ov-file#windows](here)).
4. **ffmpeg**: Install FFmpeg for video processing and MP4 support (`apt install ffmpeg` or download it from [ffmpeg.org](https://www.ffmpeg.org/)). Required for extracting frames from MP4 videos.
5. Linux specific dependencies: On Linux the following dependencies need to be installed (not needed on Windows):
   - **xdg-utils**: Install XDG utilities to display images properly (`apt install xdg-open`).
   - **libmagickwand-dev**: Install development headers for ImageMagick (`apt install libmagickwand-dev`).
   - **libyaml-dev**: Install development headers of libyaml (`apt install libyaml-dev`).
   - **libvips**: Install the libvips library (`apt install libvips42`).
   - **gnuplot**: Install the gnuplot package (`apt install gnuplot`).
   - **xrandr**: Install the X11 server utils package (`apt install x11-xserver-utils`).
   - **protobuf**: Install Google's Protobuf (`apt install protobuf-compiler`).
6. **libTorch** (Linux only): Download the C++ library from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/), and configure it:
   - In case you use also Cuda from WSL: `bundle config set --local build.torch-rb "--with-torch-dir=/path/to/libtorch-shared-with-deps-2.9.1+cu130/libtorch/ --with-cuda-ldflags=-L/usr/lib/wsl/lib"`
   - Otherwise: `bundle config set --local build.torch-rb --with-torch-dir=/path/to/libtorch-shared-with-deps-2.9.1+cu130/libtorch/`
   - Clone the https://github.com/ankane/torchvision-ruby repository next to ruby_neural_nets and modify its Gemfile to depend on numo-narray-alt instead of numo-narray.
7. **OpenBLAS** (optional): On Windows only, for improved matrix computation performance, install OpenBLAS and set the `OPEN_BLAS_PATH` environment variable to the path containing the OpenBLAS library files (e.g., `OPEN_BLAS_PATH=/path/to/openblas/lib`). If not set, the framework will run without OpenBLAS acceleration.
8. **Cuda** and **CudNN**: On WSL only, Cuda and CudNN can be installed this way:
   1. First install NVidia drivers in Windows 11 (not WSL).
   2. Link Cuda library properly in WSL (`sudo ln -s /usr/lib/wsl/lib/libcuda.so /usr/lib/libcuda.so`).
   3. Install CudNN library in WSL:
      - `wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb`
      - `sudo dpkg -i cuda-keyring_1.1-1_all.deb`
      - `sudo apt update`
      - `sudo apt -y install cudnn`
   4. Install Cuda library in WSL:
      - `wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb`
      - `sudo dpkg -i cuda-keyring_1.1-1_all.deb`
      - `sudo apt update`
      - `sudo apt -y install cuda-toolkit-13-0`
   5. Update the cache of ldconfig (`sudo ldconfig`).

### Setup

1. Clone this repository
2. Navigate to the project directory
3. Install dependencies:
   ```bash
   bundle install
   ```
   In case the compilation fails by lack of memory (this can happen in WSL), use `bundle install --jobs 1`.

## Usage

### Training models

The `train` tool provides a comprehensive CLI interface for configuring and running neural network training experiments. Use `bundle exec ruby bin/train --help` to see all available options.

Run the following command to see a complete example of training a neural network on the numbers dataset:

```bash
bundle exec ruby bin/train --dataset=numbers
```

This runs with the numbers dataset and default settings for other parameters:
- **Dataset**: Handwritten digits (0-9) from the numbers dataset (to be downloaded separately from [Kaggle](https://www.kaggle.com/datasets/ox0000dead/numbers-classification-dataset))
- **Model**: OneLayer (simple softmax classification)
- **Optimizer**: Adam with learning rate 0.001
- **Training**: 100 epochs, full batch (no minibatches)
- **Accuracy**: ClassesNumo (standard accuracy measurement)
- **Loss**: CrossEntropy
- **Checks**: Gradient checking enabled (exceptions on failure), instability checks via byebug

#### Key Options Explained

- **`--dataset`**: Choose dataset (colors, numbers)
  - Controls which dataset to load and train on

- **`--datasets-path`**: Directory containing datasets (string, default: './datasets')
  - Specifies the path to the directory where datasets are stored
  - Datasets should be organized as subdirectories within this path
  - Example: `--datasets-path /path/to/my/datasets` uses datasets from a custom directory

- **`--model`**: Select neural network architecture (OneLayer, NLayers, etc.)
  - Controls the model type and layer configuration
  - Use `--layers` to specify hidden layer sizes (comma-separated integers)

- **`--optimizer`**: Choose optimization algorithm (Adam, Constant, ExponentialDecay, etc.)
  - Controls how model parameters are updated during training
  - Use `--learning-rate` to set learning rate, `--decay` for decay-based optimizers, `--weight-decay` for L2 regularization (default: 0.0)

- **`--data-loader`**: Select data loading method (NumoImageMagick, NumoVips, TorchVips, TorchImageMagick)
  - Controls the dataset processing pipeline (partitioning, shuffling, caching, encoding, minibatching)
  - Note: Only NumoImageMagick and TorchImages data loaders support MP4 video processing

- **`--video-slices-sec`**: Number of seconds between video frames extracted from MP4 files (float, default: 1.0)
  - Controls the temporal resolution when processing video datasets
  - Smaller values produce more frames (higher temporal resolution), larger values produce fewer frames
  - Example: `--video-slices-sec=0.5` extracts frames every 0.5 seconds from MP4 files
  - Only applies when MP4 files are present in the dataset; ignored for PNG-only datasets

- **`--partitions`**: Hash of partition names and their proportion percentages [default: { training: 0.7, dev: 0.15, test: 0.15 }]
  - Controls how the dataset is split into training, development, and test partitions
  - Example: `--partitions '{"training": 0.8, "dev": 0.1, "test": 0.1}'` sets 80% training, 10% dev, 10% test

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

- **`--flatten`**: Flatten image data to 1D array for models that expect flat input vectors (boolean, default: true)
  - Controls whether to flatten image data to 1D arrays during batching
  - When enabled, applies Flatten transform after tensor conversion for models expecting flat input vectors
  - When disabled, keeps image data in original multi-dimensional format (e.g., [channels, height, width])
  - Useful for models that require different input shapes or when working with convolutional networks
  - Example: `--flatten false` keeps image data in multi-dimensional format

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
bundle exec ruby bin/train \
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
bundle exec ruby bin/train \
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

### ONNX Model Generation

The `generate_onnx` tool generates ONNX (Open Neural Network Exchange) model files that can be used with other deep learning frameworks and deployment environments.

#### Generating ONNX Models

To generate an ONNX model file:

```bash
bundle exec ruby bin/generate_onnx
```

This command:
- Creates a OneLayer neural network model configured for 110x110x3 RGB images
- Configures the model for 3-class classification
- Uses the same architecture and parameter initialization as the Ruby implementation
- Saves the model as `models/one_layer.onnx`
- Outputs a success message when complete

The generated ONNX model includes:
- **Input Layer**: Accepts images with shape [1, 110×110×3] (batch_size, flattened_pixels)
- **Dense Layer**: Fully connected layer with weights and bias
- **Softmax Activation**: Produces class probabilities for the 3 output classes
- **ONNX IR Version 8**: Compatible with modern ONNX runtime environments

#### ONNX Model Structure

The generated model follows the ONNX standard format with:
- **Gemm Operation**: Implements the dense layer using matrix multiplication
- **Softmax Operation**: Applied with axis=1 for proper class probability distribution
- **Tensor Prototypes**: Properly typed float tensors for weights, biases, and inputs/outputs
- **Graph Metadata**: Includes model name and producer information

#### Usage with Other Frameworks

The generated `models/one_layer.onnx` file can be:
- Loaded in Python using `onnxruntime` for inference
- Used with ONNX-compatible deployment tools
- Integrated into web applications via ONNX Web Runtime
- Converted to other formats (TensorFlow, TensorRT, etc.) using ONNX converters

### Inference

The `infer` tool provides a comprehensive CLI interface for running inference experiments on trained neural network models. It allows you to load models and datasets, perform forward propagation, and visualize both input and output results.

#### Running Inference

To run inference on a trained model:

```bash
bundle exec ruby bin/infer --dataset=numbers --model=OneLayer
```

This command:
- Loads the specified dataset (numbers) and displays statistics
- Creates the specified model architecture (OneLayer) with appropriate dimensions
- Performs forward propagation on the training data
- Displays both source input images and inferred output images
- Logs inference results for each minibatch processed

The inference process works with all the same configuration options as training, including:
- **Dataset Selection**: Choose from available datasets (colors, numbers)
- **Model Architecture**: Select the neural network model (OneLayer, NLayers, etc.)
- **Data Loader**: Choose the data processing pipeline (NumoImageMagick, NumoVips, TorchVips, TorchImageMagick)
- **Image Transformations**: Apply the same preprocessing as training (resize, rotate, noise, grayscale, etc.)
- **Model Parameters**: Configure layer sizes, dropout rates, and other model-specific settings
- **Debug Options**: Enable debug logging and instability checks for detailed troubleshooting

#### Inference Features

- **Model Loading**: Instantiates the same model architecture used during training with matching parameters
- **Data Processing**: Applies identical preprocessing and data augmentation as training (except for random augmentations)
- **Forward Propagation**: Performs inference-only operations without backpropagation or parameter updates
- **Visualization**: Displays both input and output images for each processed minibatch
- **Progress Tracking**: Shows inference progress with detailed logging of each minibatch
- **Experiment Support**: Supports multiple inference experiments with different configurations in a single run
- **Debug Output**: Provides detailed logging of tensor shapes and values when debug mode is enabled

The inference tool is particularly useful for:
- **Model Validation**: Verifying that trained models produce expected outputs on test data
- **Visual Inspection**: Examining how the model processes different types of input images
- **Debugging**: Identifying issues with model architecture or data preprocessing
- **Performance Analysis**: Understanding model behavior on specific samples or classes
- **Result Exploration**: Comparing inputs and outputs to gain insights into model decision-making

### Datasets

The project can use datasets in the `datasets/` directory, having the structure `datasets/<dataset_name>/<class_name>/<image_name>.png` or `datasets/<dataset_name>/<class_name>/<video_name>.mp4`.

The following datasets can be used easily:
- **colors**: Classification of colored images (red, green, blue)
- **numbers**: Handwritten digit recognition (0-9), downloaded from https://www.kaggle.com/datasets/ox0000dead/numbers-classification-dataset

Each dataset consists of PNG images and/or MP4 videos organized in class subdirectories. The framework automatically detects and handles both file formats:
- **PNG images**: Loaded directly as static images
- **MP4 videos**: Automatically sliced into frames at regular intervals for training

#### Video Processing

When MP4 files are present in the dataset, the framework automatically processes them by:
1. **Video Slicing**: Extracts frames from videos at regular time intervals
2. **Frame Extraction**: Uses FFmpeg to capture screenshots at specified time offsets
3. **Mixed Datasets**: Seamlessly combines both static images and video frames in the same dataset

The `--video-slices-sec` parameter controls the interval between extracted frames:
- **Default**: 1.0 seconds between frames
- **Example**: A 3.5-second video with `--video-slices-sec=1.0` produces 4 frames (at 0s, 1s, 2s, 3s)
- **Fine Control**: Use smaller values (e.g., 0.5) for more frames, larger values for fewer frames

This feature enables training on video datasets where temporal information is important, such as action recognition, video classification, or motion analysis tasks.

### Creating Custom Datasets

To use your own dataset:
1. Create a new directory under `datasets/`
2. Organize images and/or videos in subdirectories named after their classes
3. Ensure all images are in PNG format and videos are in MP4 format
4. Instantiate the dataset in your code: `RubyNeuralNets::Dataset.new('your_dataset_name')`

#### Video Dataset Example

For video datasets, you can mix PNG images and MP4 videos:

```
datasets/
└── my_video_dataset/
    ├── action_jump/
    │   ├── jump_01.png
    │   ├── jump_02.mp4
    │   └── jump_03.mp4
    ├── action_run/
    │   ├── run_01.mp4
    │   └── run_02.mp4
    └── action_walk/
        ├── walk_01.png
        └── walk_02.mp4
```

Training on a video dataset:
```bash
bundle exec ruby bin/train --dataset=my_video_dataset --video-slices-sec=0.5 --data-loader=NumoImageMagick
```

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
bundle exec ruby bin/train --dataset=numbers --nbr-clones 2 --rot-angle 30

# Aggressive augmentation for small datasets
bundle exec ruby bin/train --dataset=colors --nbr-clones 5 --rot-angle 90 --resize 64,64

# Augmentation with noise for robustness
bundle exec ruby bin/train --dataset=numbers --noise-intensity 0.1 --rot-angle 45

# Video processing with high temporal resolution
bundle exec ruby bin/train --dataset=my_video_dataset --video-slices-sec=0.5 --data-loader=NumoImageMagick

# Video processing with mixed dataset (images + videos)
bundle exec ruby bin/train --dataset=action_dataset --video-slices-sec=1.0 --nbr-clones 2 --rot-angle 15
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

For the video examples:
- Extract frames from MP4 videos every 0.5 seconds (high temporal resolution) or 1.0 seconds (standard resolution)
- Automatically combine video frames with any static images in the dataset
- Apply the same data augmentation pipeline to both video frames and static images

## Findings and experiments

See [docs/experiments.md](docs/experiments.md) for detailed experimental results and performance benchmarks.

## Contributing

This is a playground project for experimenting with neural networks in Ruby. Feel free to:
- Add new model architectures
- Implement additional optimizers or loss functions
- Experiment with different datasets
- Improve performance or add features

Development rules that should be followed are documented in the [rules](rules/all.md) file.

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
- `lib/ruby_neural_nets/inferer.rb`: Base inference class with common forward propagation logic
- `lib/ruby_neural_nets/trainer.rb`: Training loop implementation that inherits from Inferer, with gradient checking and training-specific logic
- `lib/ruby_neural_nets/torch/`: PyTorch integration utilities
- `lib/ruby_neural_nets/torchvision/`: TorchVision transforms for Ruby
- `lib/ruby_neural_nets/torchvision/transforms/`: Individual TorchVision transform implementations
- `lib/ruby_neural_nets/transform_helpers/`: Shared image transformation utilities used across dataset layers

### Testing

The project uses RSpec for unit testing to ensure the correctness of neural network implementations and training components.

#### Running Tests

To run the test suite:

```bash
bundle exec rspec
```

This will execute all tests in the `spec/` directory.

A nice helper has been made available to run those tests using WSL from PowerShell (useful for automation tasks and running both Windows and Linux specific tests): `.\tools\rspec_wsl.cmd`. Feel free to modify it to your local setup.

#### Test Structure

The test suite follows a structured organization:

- **`spec/scenarios/`**: Unit test scenarios grouped by interface kind being tested
  - `spec/scenarios/models/`: Tests for model functionality (OneLayer, NLayers)
  - `spec/scenarios/trainers/`: Tests for trainer functionality
  - `spec/scenarios/data_loaders/`: Tests for data loader functionality
    - Shared scenarios in `spec/scenarios/data_loaders/shared/` are used across both NumoImageMagick and NumoVips implementations to ensure consistent behavior while accounting for their different quantization ranges (65535 for ImageMagick, 255 for Vips)
  - Platform-specific scenarios in `spec/scenarios.<platform>/` directories contain tests for platform-dependent functionality, such as Torch-based models and data loaders that require platform-specific native libraries (e.g., `spec/scenarios.x86_64-linux/models/` for Linux x86_64 Torch model tests, `spec/scenarios.x86_64-linux/data_loaders/` for Linux x86_64 Torch data loader tests)
- **`spec/ruby_neural_nets_test/`**: Unit test framework and helpers
  - `spec/ruby_neural_nets_test/helpers.rb`: Test helper methods and utilities
- **`spec/spec_helper.rb`**: RSpec configuration and load path setup
- **`.rspec`**: RSpec options file that automatically requires spec_helper.rb

#### Test Features

- **Mocked File System**: Tests use the fakefs gem to mock file system operations, avoiding dependencies on actual dataset files
- **Synthetic Data**: Generates deterministic PNG image files with pixel data based on class labels for consistent test results
- **Progress Validation**: Ensures training progress is properly tracked and reported
- **Component Integration**: Tests the integration between Trainer, ProgressTracker, and Experiment classes

#### Writing Tests

When adding new tests:
1. Place test files in the appropriate `spec/scenarios/<interface_kind>/` directory with `_spec.rb` suffix
2. Use descriptive test names and contexts
3. Use with_test_dir to create test data from files
4. Mock external dependencies (random data, external APIs) for reliable execution
5. Test both success and failure scenarios
6. Verify numerical outputs are within expected ranges

### Regenerating onnx_pb.rb

onnx_pb.rb file contains the Protobuf definition of the ONNX format for Ruby. Here are the steps needed to regenerate it:
1. Git clone https://github.com/onnx/onnx
2. Install the protobuf compiler: `sudo apt install protobuf-compiler`
3. Generate the Ruby protobuf bindings: `cd onnx/onnx && protoc --ruby_out=. onnx.proto`
4. Move the generated onnx_pb.rb in lib/ruby_neural_nets/onnx/onnx_pb.rb

The ONNX protobuf format is described [here](https://github.com/onnx/onnx/blob/main/docs/IR.md).

### Building the Docker build image

This guide explains how to build and deploy the Docker image for the Ruby Neural Nets project to GitHub Container Registry (ghcr.io).

#### Prerequisites

1. Docker installed on your system
2. GitHub account with access to this repository
3. GitHub Personal Access Token with `write:packages` scope

#### Building the Docker Image

To build the Docker image locally:

```bash
docker build --file ./docker/Dockerfile.builder --tag ruby-neural-nets-builder:ubuntu25.10 .
```

#### Authenticating with GitHub Container Registry

1. Log in to GitHub Container Registry:

```bash
echo $GITHUB_TOKEN | docker login ghcr.io --username YOUR_GITHUB_USERNAME --password-stdin
```

Replace `YOUR_GITHUB_USERNAME` with your GitHub username and ensure `GITHUB_TOKEN` contains your personal access token.

#### Tagging and Pushing the Image

1. Tag the image for ghcr.io:

```bash
docker tag ruby-neural-nets-builder:ubuntu25.10 ghcr.io/muriel-salvan/ruby-neural-nets-builder:ubuntu25.10
```

2. Push the image to ghcr.io:

```bash
docker push ghcr.io/muriel-salvan/ruby-neural-nets-builder:ubuntu25.10
```

## License

See LICENSE file.
