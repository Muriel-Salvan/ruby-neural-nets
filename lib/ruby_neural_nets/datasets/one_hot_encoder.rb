require 'ruby_neural_nets/datasets/wrapper'
require 'ruby_neural_nets/sample'

module RubyNeuralNets

  module Datasets

    # Dataset encoding labels as one-hot vectors
    class OneHotEncoder < Wrapper

      # Get the mapping of labels to one-hot vectors
      #   Hash< String, Array< Integer > >
      attr_reader :one_hot_labels

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): Dataset providing labels to be one-hot encoded
      def initialize(dataset)
        super
        labels = dataset.labels.sort
        # Compute the map of one-hot vectors corresponding to each label
        # Hash< label, Array< Integer > >
        @one_hot_labels = labels.map.with_index do |label, idx|
          one_hot_vector = [0] * labels.size
          one_hot_vector[idx] = 1
          [
            label,
            one_hot_vector
          ]
        end.to_h
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
          -> { @one_hot_labels[sample.target] }
        )
      end

      # Return the underlying dataset's label for a given output label of this dataset layer
      #
      # Parameters::
      # * *y* (Object): Label, as returned by the [] method
      # Result::
      # * Object: Corresponding underlying label
      def underlying_label(y)
        found_label, _found_index = @one_hot_labels.find { |_select_label, select_vector| select_vector == y }
        @dataset.underlying_label(found_label)
      end

    end

  end

end
