require 'ruby_neural_nets/datasets/wrapper'

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
        @labels = dataset.group_by { |x, label| label }.keys.sort.map.with_index { |label, idx| [label, idx] }.to_h
      end

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Object: The element X of the dataset
      # * Object: The element Y of the dataset
      def [](index)
        x, label = @dataset[index]
        [x, @labels[label]]
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
