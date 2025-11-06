require 'ruby_neural_nets/datasets/wrapper'

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
        labels = dataset.group_by { |x, label| label }.keys.sort
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
      # * Object: The element X of the dataset
      # * Object: The element Y of the dataset
      def [](index)
        x, label = @dataset[index]
        [x, @one_hot_labels[label]]
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
