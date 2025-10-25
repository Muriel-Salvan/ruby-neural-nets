require 'ruby_neural_nets/data_loader'
require 'ruby_neural_nets/datasets/labeled_files'
require 'ruby_neural_nets/datasets/labeled_data_partitioner'
require 'ruby_neural_nets/datasets/images_from_files'
require 'ruby_neural_nets/datasets/image_normalize'
require 'ruby_neural_nets/datasets/one_hot_encoder'
require 'ruby_neural_nets/datasets/cache_memory'
require 'ruby_neural_nets/datasets/epoch_shuffler'
require 'ruby_neural_nets/datasets/minibatch'

module RubyNeuralNets
  
  module DataLoaders
        
    class Numo < DataLoader

      # Instantiate a partitioned dataset.
      #
      # Parameters::
      # * *name* (String): Dataset name containing real data
      # * *rng* (Random): The random number generator to be used
      # Result::
      # * LabeledDataPartitioner: The partitioned dataset.
      def new_partitioned_dataset(name:, rng:)
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
      # * *max_minibatch_size* (Integer): The required minibatch size
      # Result::
      # * Dataset: The dataset that will serve data as minibatches
      def new_minibatch_dataset(dataset:, rng:, max_minibatch_size:)
        Datasets::Minibatch.new(
          Datasets::EpochShuffler.new(
            Datasets::CacheMemory.new(
              Datasets::OneHotEncoder.new(
                Datasets::ImageNormalize.new(
                  Datasets::ImagesFromFiles.new(dataset)
                )
              )
            ),
            rng:
          ),
          max_minibatch_size:
        )
      end

    end

  end

end
