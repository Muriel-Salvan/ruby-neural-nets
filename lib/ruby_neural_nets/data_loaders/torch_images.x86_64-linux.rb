require 'ruby_neural_nets/data_loader'
require 'ruby_neural_nets/datasets/labeled_files'
require 'ruby_neural_nets/datasets/labeled_data_partitioner'
require 'ruby_neural_nets/datasets/cache_memory'
require 'ruby_neural_nets/datasets/epoch_shuffler'
require 'ruby_neural_nets/datasets/indexer'
require 'ruby_neural_nets/datasets/labeled_torch_images'
require 'ruby_neural_nets/datasets/minibatch_torch'
require 'ruby_neural_nets/torchvision/transforms/image_magick_trim'
require 'ruby_neural_nets/torchvision/transforms/image_magick_resize'
require 'ruby_neural_nets/torchvision/transforms/image_magick_rotate'
require 'ruby_neural_nets/torchvision/transforms/image_magick_noise'
require 'ruby_neural_nets/torchvision/transforms/image_magick_grayscale'
require 'ruby_neural_nets/torchvision/transforms/image_magick_minmax_normalize'
require 'ruby_neural_nets/torchvision/transforms/image_magick_adaptive_invert'
require 'ruby_neural_nets/torchvision/transforms/file_to_image_magick'
require 'ruby_neural_nets/torchvision/transforms/image_magick_remove_alpha'
require 'ruby_neural_nets/torchvision/transforms/cache'
require 'ruby_neural_nets/torchvision/transforms/flatten'
require 'ruby_neural_nets/torchvision/transforms/to_double'
require 'ruby_neural_nets/torchvision/transforms/image_magick_to_tensor'

module RubyNeuralNets

  module DataLoaders

    # Data loader that can read images and videos and provide ImageMagick images out of it
    class TorchImages < DataLoader

      # Constructor
      #
      # Parameters::
      # * *datasets_path* (String): The datasets path
      # * *dataset* (String): The dataset name
      # * *max_minibatch_size* (Integer): Max size each minibatch should have
      # * *dataset_seed* (Integer): Random number generator seed for dataset shuffling and data order
      # * *partitions* (Hash<Symbol, Float>): List of partitions and their proportion percentage
      # * *nbr_clones* (Integer): Number of times each element should be cloned
      # * *rot_angle* (Float): Maximum rotation angle in degrees for random image transformations
      # * *grayscale* (bool): Convert images to grayscale, reducing channels from 3 to 1
      # * *adaptive_invert* (bool): Apply adaptive color inversion based on top left pixel intensity
      # * *trim* (bool): Trim images to remove borders and restore original aspect ratio
      # * *resize* (Array): Resize dimensions [width, height] for image transformations
      # * *noise_intensity* (Float): Intensity of Gaussian noise for image transformations
      # * *minmax_normalize* (bool): Scale image data to always be within the range 0 to 1
      # * *flatten* (bool): Flatten image data to 1D array for models that expect flat input vectors
      # * *video_slices_sec* (Float): Number of seconds of each video slice used to extract images from MP4 files
      def initialize(datasets_path:, dataset:, max_minibatch_size:, dataset_seed:, partitions:, nbr_clones:, rot_angle:, grayscale:, adaptive_invert:, trim:, resize:, noise_intensity:, minmax_normalize:, flatten:, video_slices_sec:)
        @nbr_clones = nbr_clones
        @rot_angle = rot_angle
        @grayscale = grayscale
        @adaptive_invert = adaptive_invert
        @trim = trim
        @resize = resize
        @noise_intensity = noise_intensity
        @minmax_normalize = minmax_normalize
        @flatten = flatten
        @video_slices_sec = video_slices_sec
        super(datasets_path:, dataset:, max_minibatch_size:, dataset_seed:, partitions:)
      end

      private

      # Instantiate a partitioned dataset.
      #
      # Parameters::
      # * *datasets_path* (String): The datasets path
      # * *name* (String): Dataset name containing real data
      # * *rng* (Random): The random number generator to be used
      # * *numo_rng* (Numo::Random::Generator): The Numo random number generator to be used
      # * *partitions* (Hash<Symbol, Float>): List of partitions and their proportion percentage
      # Result::
      # * LabeledDataPartitioner: The partitioned dataset.
      def new_partitioned_dataset(datasets_path:, name:, rng:, numo_rng:, partitions:)
        Datasets::LabeledDataPartitioner.new(
          Datasets::FileToImageMagick.new(
            Datasets::LabeledFiles.new(datasets_path:, name:),
            video_slices_sec: @video_slices_sec
          ),
          partitions:,
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
        transforms = [
          TorchVision::Transforms::ImageMagickRemoveAlpha.new
        ]
        transforms << TorchVision::Transforms::ImageMagickTrim.new if @trim
        transforms << TorchVision::Transforms::ImageMagickResize.new(@resize)
        transforms << TorchVision::Transforms::ImageMagickGrayscale.new if @grayscale
        transforms << TorchVision::Transforms::ImageMagickMinmaxNormalize.new if @minmax_normalize
        transforms << TorchVision::Transforms::ImageMagickAdaptiveInvert.new if @adaptive_invert

        Datasets::CacheMemory.new(
          Datasets::Indexer.new(
            Datasets::TorchTransformImages.new(
              dataset,
              [TorchVision::Transforms::Cache.new(transforms)]
            )
          )
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
        # Create augmentation transforms
        augmentation_transforms = []
        augmentation_transforms << TorchVision::Transforms::ImageMagickRotate.new(@rot_angle, rng) if @rot_angle > 0
        augmentation_transforms << TorchVision::Transforms::ImageMagickNoise.new(@noise_intensity, numo_rng) if @noise_intensity > 0

        Datasets::TorchTransformImages.new(
          Datasets::Clone.new(
            preprocessed_dataset,
            nbr_clones: @nbr_clones
          ),
          augmentation_transforms
        )
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
            Datasets::TorchTransformImages.new(
              augmented_dataset,
              [
                TorchVision::Transforms::ImageMagickToTensor.new
              ] +
              (@flatten ? [TorchVision::Transforms::Flatten.new] : []) +
              [
                TorchVision::Transforms::ToDouble.new
              ]
            ),
            rng:
          ),
          max_minibatch_size:
        )
      end

    end

  end

end
