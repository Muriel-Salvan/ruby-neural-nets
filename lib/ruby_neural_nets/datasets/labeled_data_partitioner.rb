require 'ruby_neural_nets/datasets/wrapper'

module RubyNeuralNets

  module Datasets

    # Dataset partitioning among types (train, dev, test).
    # Keeps same proportions between types.
    # Randomize the partitions' items.
    class LabeledDataPartitioner < Wrapper

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): Dataset giving underlying labeled data
      # * *partitions* (Hash<Symbol, Float>): List of partitions and their proportion percentage [default: { training: 0.7, dev: 0.15, test: 0.15 }]
      def initialize(dataset, partitions: { training: 0.7, dev: 0.15, test: 0.15 })
        super(dataset)

        raise "Partitions percentages don't sum to 1: #{partitions}" unless partitions.values.sum == 1.0

        # By default, use the first partition
        @partition = partitions.keys.first
        # For each label, get the indexes of elements belonging to that label
        # Hash< label, Array< index > >
        labeled_indexes = {}
        @dataset.each_with_index do |(_x, label), index|
          labeled_indexes[label] = [] unless labeled_indexes.key?(label)
          labeled_indexes[label] << index
        end
        # Shuffle indexes and partition each label
        # Hash< label, Hash< partition, Array< index > > >
        partition_labeled_indexes = labeled_indexes.to_h do |label, indexes|
          shuffled_indexes = indexes.shuffle(random: RubyNeuralNets::Helpers.dataset_rng)
          next_index = 0
          [
            label,
            partitions.to_h do |partition, partition_percentage|
              first_index = next_index
              next_index = next_index + (indexes.size * partition_percentage).to_i + 1
              [partition, shuffled_indexes[first_index..next_index - 1]]
            end
          ]
        end
        # Partition the labeled indexes
        # Hash< partition, Array< index > >
        @partitions = partitions.keys.to_h do |partition|
          [
            partition,
            partition_labeled_indexes.values.inject([]) { |total_indexes, label_indexes| total_indexes + label_indexes[partition] }
          ]
        end
      end

      # Select a partition
      #
      # Parameters::
      # * *partition* (Symbol): Partition to be selected
      # * *code* (Proc): Optional code to be called with the partition selected.
      #   If provided, then the previous partition will be selected back at the end of the blockés execution.
      def select_partition(partition)
        if block_given?
          old_partition = @partition
          begin
            @partition = partition
            yield
          ensure
            @partition = old_partition
          end
        else
          @partition = partition
        end
      end

      # Return available partitions
      #
      # Result::
      # * Array<Symbol>: Available partitions
      def partitions
        @partitions.keys
      end

      # Return the dataset size
      #
      # Result::
      # * Integer: Number of elements in this dataset
      def size
        @partitions[@partition].size
      end

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * x: The element X of the dataset
      # * y: The element Y of the dataset
      def [](index)
        @dataset[@partitions[@partition][index]]
      end

    end

  end

end
