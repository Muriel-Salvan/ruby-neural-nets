require 'ruby_neural_nets/data_loader'
require 'ruby_neural_nets/datasets/labeled_files'
require 'ruby_neural_nets/datasets/labeled_data_partitioner'
require 'ruby_neural_nets/datasets/images_from_files'
require 'ruby_neural_nets/datasets/one_hot_encoder'
require 'ruby_neural_nets/datasets/cache_memory'
require 'ruby_neural_nets/datasets/epoch_shuffler'
require 'ruby_neural_nets/datasets/minibatch'

module RubyNeuralNets
  
  module DataLoaders
        
    class Numo < DataLoader

      # Create an elements dataset that maps to the labels and is partitioned.
      # The following instance variables can be used to initialize the dataset correctly:
      # * *@dataset_name* (String): The dataset name.
      #
      # Result::
      # * Dataset: The dataset that will serve data with Y being true labels, unencoded
      def new_elements_labels_dataset
        RubyNeuralNets::Datasets::LabeledDataPartitioner.new(
          RubyNeuralNets::Datasets::LabeledFiles.new(name: @dataset_name)
        )
      end

      # Create an elements dataset for this data loader.
      # The following instance variables can be used to initialize the dataset correctly:
      # * *@dataset_name* (String): The dataset name.
      # * *@elements_labels_dataset* (Dataset): The elements dataset previously created with true labels.
      #
      # Result::
      # * Dataset: The dataset that will serve data with X and Y being prepared for training
      def new_elements_dataset
        RubyNeuralNets::Datasets::EpochShuffler.new(
          RubyNeuralNets::Datasets::CacheMemory.new(
            RubyNeuralNets::Datasets::OneHotEncoder.new(
              RubyNeuralNets::Datasets::ImagesFromFiles.new(@elements_labels_dataset)
            )
          )
        )
      end

      # Return a minibatch dataset for this data loader.
      # The following instance variables can be used to initialize the dataset correctly:
      # * *@dataset_name* (String): The dataset name.
      # * *@elements_labels_dataset* (Dataset): The elements dataset previously created with true labels.
      # * *@elements_dataset* (Dataset): The elements dataset previously created.
      #
      # Parameters::
      # * *max_minibatch_size* (Integer): The required minibatch size
      # Result::
      # * Dataset: The dataset that will serve data as minibatches
      def new_minibatch_dataset(max_minibatch_size:)
        Datasets::Minibatch.new(@elements_dataset, max_minibatch_size:)
      end

    end

  end

end
