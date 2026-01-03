require 'ruby_neural_nets/dataset'

module RubyNeuralNets

  module Datasets

    # Dataset of labeled files
    class LabeledFiles < Dataset

      # Get the list of labels
      #   Array<String>
      attr_reader :labels

      # Name given to the default label when the dataset is not labelled
      DEFAULT_LABEL = 'no_label'

      # Constructor
      #
      # Parameters::
      # * *datasets_path* (String): The datasets path
      # * *name* (String): Name of the dataset to read (sub-directory of the datasets directory)
      def initialize(datasets_path:, name:)
        super()
        @datasets_path = datasets_path
        @name = name
        @labels = Dir.glob("#{root}/*").
          select { |file| File.directory?(file) }.
          map { |file| File.basename(file) }.
          sort
        if @labels.empty?
          @labels = [DEFAULT_LABEL]
          @dataset = Dir.glob("#{root}/*").sort.map { |file| [file, DEFAULT_LABEL] }
        else
          @dataset = @labels.
            map { |label| Dir.glob("#{root}/#{label}/*").sort.map { |file| [file, label] } }.
            flatten(1)
        end
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
      # * Object: The element X of the dataset
      # * Object: The element Y of the dataset
      def [](index)
        @dataset[index]
      end

      private

      # Get the root folder of the dataset
      #
      # Result::
      # * String: The root folder
      def root
        "#{@datasets_path}/#{@name}"
      end

    end

  end

end
