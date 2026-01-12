require 'torchvision'

module RubyNeuralNets

  module Torch
    
    # Dataset that uses an underlying Dataset for a given dataset type
    class SampleFolderDataset < ::TorchVision::Datasets::DatasetFolder

      # Return the mapping between the class and index
      #   Hash< Object, Integer >
      attr_reader :class_to_idx

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): The dataset containing files
      # * *transforms* (Array): Array of TorchVision transforms to apply
      def initialize(dataset, transforms = [])
        @dataset = dataset
        super(@dataset.root, transform: ::TorchVision::Transforms::Compose.new(transforms))
        # Do nothing in the loader, as the loader is part of the transforms layers
        @loader = proc { |path| path }
      end

      private
      
      def find_classes(dir)
        [@dataset.labels, @dataset.labels.map.with_index.to_h]
      end

      def make_dataset(directory, class_to_idx, extensions, is_valid_file)
        @dataset.map { |sample| [sample.input, class_to_idx[sample.target]] }
      end

    end

  end

end
