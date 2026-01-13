require 'ruby_neural_nets/datasets/wrapper'
require 'ruby_neural_nets/sample'

module RubyNeuralNets

  module Datasets

    # Dataset encoding labels as integer indexes
    class Indexer < Wrapper

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): Dataset providing labels to be one-hot encoded
      def initialize(dataset)
        super
        @labels = dataset.labels.sort.map.with_index { |label, idx| [label, idx] }.to_h
      end

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Sample: The sample containing input and target data
      def [](index)
        sample = @dataset[index]
        Sample.new(
          -> { sample.input },
          -> { @labels[sample.target] }
        )
      end

      # Return the underlying dataset's label for a given output label of this dataset layer
      #
      # Parameters::
      # * *y* (Object): Label, as returned by the [] method
      # Result::
      # * Object: Corresponding underlying label
      def underlying_label(y)
        found_label, _found_index = @labels.find { |_select_label, select_index| select_index == y }
        @dataset.underlying_label(found_label)
      end

    end

  end

end
