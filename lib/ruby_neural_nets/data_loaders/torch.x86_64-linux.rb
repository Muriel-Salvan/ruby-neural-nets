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

      # Return a minibatch dataset for this data loader, from a dataset that has already been partitioned.
      #
      # Parameters::
      # * *dataset* (Dataset): The partitioned dataset serving data for the minibatches
      # * *rng* (Random): The random number generator to be used
      # * *numo_rng* (Numo::Random::Generator): The Numo random number generator to be used
      # * *max_minibatch_size* (Integer): The required minibatch size
      # Result::
      # * Dataset: The dataset that will serve data as minibatches
      def new_minibatch_dataset(dataset:, rng:, numo_rng:, max_minibatch_size:)
        Datasets::MinibatchTorch.new(
          Datasets::EpochShuffler.new(
            Datasets::CacheMemory.new(
              Datasets::LabeledTorchImages.new(dataset)
            ),
            rng:
          ),
          max_minibatch_size:
        )
      end

    end

  end

end
