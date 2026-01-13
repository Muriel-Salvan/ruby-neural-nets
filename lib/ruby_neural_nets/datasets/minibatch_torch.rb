require 'ruby_neural_nets/datasets/minibatch'
require 'ruby_neural_nets/torch/sample_folder_dataset'
require 'ruby_neural_nets/minibatches/torch'

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
        # Use lazy evaluation by passing the dataset directly and providing a custom collate_fn
        @torch_data_loader = ::Torch::Utils::Data::DataLoader.new(
          @dataset,
          batch_size: @max_minibatch_size,
          shuffle: false,
          collate_fn: method(:custom_collate_fn)
        )
      end

      # Custom collate function that processes Sample objects lazily
      #
      # Parameters::
      # * *batch* (Array<Sample>): Array of Sample objects to be collated
      # Result::
      # * Array: Array containing [inputs, labels] tensors
      def custom_collate_fn(batch)
        # Use the default collate behavior to create tensors
        [default_convert(batch.map(&:input)), default_convert(batch.map(&:target))]
      end

      # Default convert method similar to Torch's DataLoader
      #
      # Parameters::
      # * *batch* (Array): Array of elements to convert
      # Result::
      # * Object: Converted batch (tensor or other format)
      def default_convert(batch)
        return batch if batch.empty?
        
        elem = batch[0]
        case elem
        when ::Torch::Tensor
          ::Torch.stack(batch, 0)
        when Integer
          ::Torch.tensor(batch)
        when Array
          batch.transpose.map { |v| default_convert(v) }
        else
          batch
        end
      end

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Sample: The sample containing input and target data
      def [](index)
        # This interface only needs each.
        raise 'Not implemented as only the each method is supposed to be needed'
      end

      # Iterate over all elements
      #
      # Parameters::
      # * *Code*: Code called for each element iterated on
      def each
        return to_enum(:each) unless block_given?
        
        @torch_data_loader.each do |(inputs, labels)|
          yield Minibatches::Torch.new(-> { inputs }, -> { labels })
        end
      end

    end

  end

end
