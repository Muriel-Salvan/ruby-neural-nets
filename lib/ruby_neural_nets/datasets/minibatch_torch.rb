require 'ruby_neural_nets/datasets/minibatch'
require 'ruby_neural_nets/torch/sample_folder_dataset'

module RubyNeuralNets

  module Datasets

    # Dataset of minibatches, to be used by Torch.rb
    class MinibatchTorch < Minibatch

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): Dataset providing elements to be served in minibatches
      # * *max_minibatch_size* (Integer): Max size each minibatch should have [default: 1000]
      def initialize(dataset, max_minibatch_size: 1000)
        super
        # Don't use shuffle, as the dataset given is an EpochShuffler and already handles shuffling data before it gets batched.
        @torch_data_loader = ::Torch::Utils::Data::DataLoader.new(@dataset, batch_size: @max_minibatch_size, shuffle: false)
      end

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * x: The element X of the dataset
      # * y: The element Y of the dataset
      def [](index)
        # This interface only needs each.
        raise 'Not implemented as only the each method is supposed to be needed'
      end

      # Iterate over all elements
      #
      # Parameters::
      # * Code: Code called for each element iterated on
      def each
        @torch_data_loader.each do |(inputs, labels)|
          yield inputs, [labels, inputs.shape[0]]
        end
      end

    end

  end

end
