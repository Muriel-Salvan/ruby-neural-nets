require 'ruby_neural_nets/data_loader'
require 'ruby_neural_nets/datasets/labeled_files'
require 'ruby_neural_nets/datasets/labeled_data_partitioner'
require 'ruby_neural_nets/datasets/cache_memory'
require 'ruby_neural_nets/datasets/epoch_shuffler'
require 'ruby_neural_nets/datasets/labeled_torch_images'
require 'ruby_neural_nets/datasets/minibatch_torch'
require 'ruby_neural_nets/torchvision/transforms/vips_to_image_magick'
require 'ruby_neural_nets/torchvision/transforms/image_magick_trim'
require 'ruby_neural_nets/torchvision/transforms/image_magick_resize'
require 'ruby_neural_nets/torchvision/transforms/image_magick_grayscale'
require 'ruby_neural_nets/torchvision/transforms/image_magick_minmax_normalize'
require 'ruby_neural_nets/torchvision/transforms/image_magick_adaptive_invert'
require 'ruby_neural_nets/torchvision/transforms/image_magick_to_tensor'
require 'ruby_neural_nets/torchvision/transforms/remove_alpha'
require 'ruby_neural_nets/torchvision/transforms/flatten'
require 'ruby_neural_nets/torchvision/transforms/to_double'

module RubyNeuralNets

  module DataLoaders

    class Torch < DataLoader

      # Constructor
      #
      # Parameters::
      # * *dataset* (String): The dataset name
      # * *max_minibatch_size* (Integer): Max size each minibatch should have
      # * *dataset_seed* (Integer): Random number generator seed for dataset shuffling and data order
      # * *nbr_clones* (Integer): Number of times each element should be cloned
      # * *rot_angle* (Float): Maximum rotation angle in degrees for random image transformations
      # * *grayscale* (bool): Convert images to grayscale, reducing channels from 3 to 1
      # * *adaptive_invert* (bool): Apply adaptive color inversion based on top left pixel intensity
      # * *trim* (bool): Trim images to remove borders and restore original aspect ratio
      # * *resize* (Array): Resize dimensions [width, height] for image transformations
      # * *noise_intensity* (Float): Intensity of Gaussian noise for image transformations
      # * *minmax_normalize* (bool): Scale image data to always be within the range 0 to 1
      def initialize(dataset:, max_minibatch_size:, dataset_seed:, nbr_clones:, rot_angle:, grayscale:, adaptive_invert:, trim:, resize:, noise_intensity:, minmax_normalize:)
        @nbr_clones = nbr_clones
        @rot_angle = rot_angle
        @grayscale = grayscale
        @adaptive_invert = adaptive_invert
        @trim = trim
        @resize = resize
        @noise_intensity = noise_intensity
        @minmax_normalize = minmax_normalize
        super(dataset:, max_minibatch_size:, dataset_seed:)
      end

      # Instantiate a partitioned dataset.
      #
      # Parameters::
      # * *name* (String): Dataset name containing real data
      # * *rng* (Random): The random number generator to be used
      # * *numo_rng* (Numo::Random::Generator): The Numo random number generator to be used
      # Result::
      # * LabeledDataPartitioner: The partitioned dataset.
      def new_partitioned_dataset(name:, rng:, numo_rng:)
        Datasets::LabeledDataPartitioner.new(
          Datasets::LabeledFiles.new(name:),
          rng:
        )
      end

      # Return a preprocessing dataset for this data loader.
      #
      # Parameters::
      # * *dataset* (Dataset): The partitioned dataset
      # Result::
      # * Dataset: The dataset with preprocessing applied
      def new_preprocessing_dataset(dataset)
        Datasets::CacheMemory.new(
          Datasets::LabeledTorchImages.new(dataset, preprocessing_transforms)
        )
      end

      # Return an augmentation dataset for this data loader.
      # This is only used for the training dataset.
      #
      # Parameters::
      # * *preprocessed_dataset* (Dataset): The preprocessed dataset
      # * *rng* (Random): The random number generator to be used
      # * *numo_rng* (Numo::Random::Generator): The Numo random number generator to be used
      # Result::
      # * Dataset: The dataset with augmentation applied
      def new_augmentation_dataset(preprocessed_dataset, rng:, numo_rng:)
        # For Torch, we need to apply augmentation transforms
        # Since LabeledTorchImages already applies transforms, we need to modify it
        # For now, return the preprocessed dataset - augmentation transforms need more work
        preprocessed_dataset
      end

      # Return a batching dataset for this data loader.
      #
      # Parameters::
      # * *augmented_dataset* (Dataset): The augmented dataset
      # * *rng* (Random): The random number generator to be used
      # * *numo_rng* (Numo::Random::Generator): The Numo random number generator to be used
      # * *max_minibatch_size* (Integer): The required minibatch size
      # Result::
      # * Dataset: The dataset with batching applied
      def new_batching_dataset(augmented_dataset, rng:, numo_rng:, max_minibatch_size:)
        Datasets::MinibatchTorch.new(
          Datasets::EpochShuffler.new(
            augmented_dataset,
            rng:
          ),
          max_minibatch_size:
        )
      end

      private

      # Get the preprocessing transforms for TorchVision
      #
      # Result::
      # * Array: Array of TorchVision transforms for preprocessing
      def preprocessing_transforms
        transforms = [
          RubyNeuralNets::TorchVision::Transforms::VipsToImageMagick.new
        ]
        # Apply trim if specified (works on ImageMagick images)
        transforms << RubyNeuralNets::TorchVision::Transforms::ImageMagickTrim.new if @trim
        # Apply resize (works on ImageMagick images, after trim)
        transforms << RubyNeuralNets::TorchVision::Transforms::ImageMagickResize.new(@resize)
        # Apply grayscale if specified (works on ImageMagick images)
        transforms << RubyNeuralNets::TorchVision::Transforms::ImageMagickGrayscale.new if @grayscale
        # Apply min-max normalization if specified (works on ImageMagick images)
        transforms << RubyNeuralNets::TorchVision::Transforms::ImageMagickMinmaxNormalize.new if @minmax_normalize
        # Apply adaptive invert if specified (works on ImageMagick images)
        transforms << RubyNeuralNets::TorchVision::Transforms::ImageMagickAdaptiveInvert.new if @adaptive_invert
        # Apply tensor-level transforms
        transforms + [
          RubyNeuralNets::TorchVision::Transforms::ImageMagickToTensor.new,
          RubyNeuralNets::TorchVision::Transforms::RemoveAlpha.new,
          RubyNeuralNets::TorchVision::Transforms::Flatten.new,
          RubyNeuralNets::TorchVision::Transforms::ToDouble.new,
        ]
      end

    end

  end

end
