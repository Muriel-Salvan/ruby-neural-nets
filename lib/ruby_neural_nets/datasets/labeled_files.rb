require 'ruby_neural_nets/dataset'

module RubyNeuralNets

  module Datasets

    # Dataset of labeled files
    class LabeledFiles < Dataset

      # Get the list of labels
      #   Array<String>
      attr_reader :labels

      # Constructor
      #
      # Parameters::
      # * *name* (String): Name of the dataset to read (sub-directory of the datasets directory)
      def initialize(name:)
        super()
        @name = name
        @labels = Dir.glob("#{root}/*").
          select { |file| File.directory?(file) }.
          map { |file| File.basename(file) }.
          sort
        @dataset = @labels.
          map { |label| Dir.glob("#{root}/#{label}/*").sort.map { |file| [file, label] } }.
          flatten(1)
        raise "No data in dataset #{name}" if @dataset.empty?
      end

      # Return the dataset size
      #
      # Result::
      # * Integer: Number of elements in this dataset
      def size
        @dataset.size
      end

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * x: The element X of the dataset
      # * y: The element Y of the dataset
      def [](index)
        @dataset[index]
      end

      private

      # Get the root folder of the dataset
      #
      # Result::
      # * String: The root folder
      def root
        "./datasets/#{@name}"
      end

    end

  end

end
