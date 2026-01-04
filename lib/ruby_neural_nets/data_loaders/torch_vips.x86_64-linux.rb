require 'ruby_neural_nets/data_loader'
require 'ruby_neural_nets/datasets/labeled_files'
require 'ruby_neural_nets/datasets/labeled_data_partitioner'
require 'ruby_neural_nets/datasets/cache_memory'
require 'ruby_neural_nets/datasets/epoch_shuffler'
require 'ruby_neural_nets/datasets/labeled_torch_images'
require 'ruby_neural_nets/datasets/minibatch_torch'
require 'ruby_neural_nets/torchvision/transforms/vips_trim'
require 'ruby_neural_nets/torchvision/transforms/vips_resize'
require 'ruby_neural_nets/torchvision/transforms/vips_rotate'
require 'ruby_neural_nets/torchvision/transforms/vips_noise'
require 'ruby_neural_nets/torchvision/transforms/vips_grayscale'
require 'ruby_neural_nets/torchvision/transforms/vips_minmax_normalize'
require 'ruby_neural_nets/torchvision/transforms/vips_adaptive_invert'
require 'ruby_neural_nets/torchvision/transforms/file_to_vips'
require 'ruby_neural_nets/torchvision/transforms/vips_remove_alpha'
require 'ruby_neural_nets/torchvision/transforms/vips_8_bits'
require 'ruby_neural_nets/torchvision/transforms/cache'
require 'ruby_neural_nets/torchvision/transforms/flatten'
require 'ruby_neural_nets/torchvision/transforms/to_double'

module RubyNeuralNets

  module DataLoaders

    class TorchVips < DataLoader

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
      def initialize(datasets_path:, dataset:, max_minibatch_size:, dataset_seed:, partitions:, nbr_clones:, rot_angle:, grayscale:, adaptive_invert:, trim:, resize:, noise_intensity:, minmax_normalize:, flatten:)
        @nbr_clones = nbr_clones
        @rot_angle = rot_angle
        @grayscale = grayscale
        @adaptive_invert = adaptive_invert
        @trim = trim
        @resize = resize
        @noise_intensity = noise_intensity
        @minmax_normalize = minmax_normalize
        @flatten = flatten
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
          Datasets::LabeledFiles.new(datasets_path:, name:),
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
          RubyNeuralNets::TorchVision::Transforms::FileToVips.new,
          RubyNeuralNets::TorchVision::Transforms::VipsRemoveAlpha.new
        ]
        transforms << RubyNeuralNets::TorchVision::Transforms::VipsTrim.new if @trim
        transforms << RubyNeuralNets::TorchVision::Transforms::VipsResize.new(@resize)
        transforms << RubyNeuralNets::TorchVision::Transforms::VipsGrayscale.new if @grayscale
        transforms << RubyNeuralNets::TorchVision::Transforms::VipsMinmaxNormalize.new if @minmax_normalize
        transforms << RubyNeuralNets::TorchVision::Transforms::VipsAdaptiveInvert.new if @adaptive_invert

        Datasets::CacheMemory.new(
          Datasets::LabeledTorchImages.new(
            dataset,
            [RubyNeuralNets::TorchVision::Transforms::Cache.new(transforms)]
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
        augmentation_transforms << RubyNeuralNets::TorchVision::Transforms::VipsRotate.new(@rot_angle, rng) if @rot_angle > 0
        augmentation_transforms << RubyNeuralNets::TorchVision::Transforms::VipsNoise.new(@noise_intensity, numo_rng) if @noise_intensity > 0

        Datasets::LabeledTorchImages.new(
          Datasets::Clone.new(
            preprocessed_dataset.files_dataset,
            nbr_clones: @nbr_clones
          ),
          preprocessed_dataset.transforms + augmentation_transforms
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
            Datasets::LabeledTorchImages.new(
              augmented_dataset.files_dataset,
              augmented_dataset.transforms + [
                RubyNeuralNets::TorchVision::Transforms::Vips8Bits.new,
                ::TorchVision::Transforms::ToTensor.new
              ] +
              (@flatten ? [RubyNeuralNets::TorchVision::Transforms::Flatten.new] : []) +
              [
                RubyNeuralNets::TorchVision::Transforms::ToDouble.new
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
