require 'ruby_neural_nets/datasets/images_from_files'
require 'ruby_neural_nets/datasets/wrapper'
require 'ruby_neural_nets/torch/sample_folder_dataset'

module RubyNeuralNets

  module Datasets

    # Dataset serving TorchVision images.
    class LabeledTorchImages < Wrapper

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): Dataset giving underlying labeled data
      def initialize(dataset)
        super
        @torch_dataset = RubyNeuralNets::Torch::SampleFolderDataset.new(@dataset)
      end

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Object: The element X of the dataset
      # * Object: The element Y of the dataset
      def [](index)
        @torch_dataset[index]
      end

      # Get some images stats.
      # Those are supposed to be the same for all samples from the dataset and can be used to compute the model's architecture.
      #
      # Result::
      # * Hash: Image stats:
      #   * *rows* (Integer): Number of rows
      #   * *cols* (Integer): Number of columns
      #   * *channels* (Integer): Number of channels
      def image_stats
        RubyNeuralNets::Datasets::ImagesFromFiles.new(@dataset).image_stats
      end

    end

  end

end
