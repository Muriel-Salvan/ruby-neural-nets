require 'ruby_neural_nets/data_loader'
require 'ruby_neural_nets/datasets/labeled_files'
require 'ruby_neural_nets/datasets/labeled_data_partitioner'
require 'ruby_neural_nets/datasets/cache_memory'
require 'ruby_neural_nets/datasets/epoch_shuffler'
require 'ruby_neural_nets/datasets/labeled_torch_images'
require 'ruby_neural_nets/datasets/minibatch_torch'

module RubyNeuralNets

  module DataLoaders

    class Torch < DataLoader

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
          Datasets::LabeledTorchImages.new(dataset)
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



    end

  end

end
