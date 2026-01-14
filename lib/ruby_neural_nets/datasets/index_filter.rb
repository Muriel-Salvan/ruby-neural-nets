require 'ruby_neural_nets/datasets/wrapper'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that filters the dataset to only include specific indexes.
    # Supports individual indexes and ranges (e.g., "1,3,5-8").
    # "all" means all available indexes.
    class IndexFilter < Wrapper

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): Dataset to be filtered
      # * *filter* (String): Filter specification ("all" or comma-separated list of indexes/ranges)
      def initialize(dataset, filter:)
        super(dataset)

        if filter == 'all'
          @indexes = (0...@dataset.size).to_a
        else
          @indexes = parse_filter(filter)
        end
      end

      # Return the dataset size
      #
      # Result::
      # * Integer: Number of elements in this dataset
      def size
        @indexes.size
      end

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Sample: The sample containing input and target data
      def [](index)
        @dataset[@indexes[index]]
      end

      private

      # Parse the filter string into an array of indexes
      #
      # Parameters::
      # * *filter* (String): Filter specification
      # Result::
      # * Array<Integer>: List of indexes to include
      def parse_filter(filter)
        indexes = Set.new
        filter.split(',').each do |part|
          if part.include?('-')
            # Range like "5-8"
            start_str, end_str = part.split('-')
            start_idx = Integer(start_str.strip)
            end_idx = Integer(end_str.strip)
            (start_idx..end_idx).each { |idx| indexes << idx }
          else
            # Individual index
            indexes << Integer(part.strip)
          end
        end
        indexes.to_a.sort
      end

    end

  end

end
