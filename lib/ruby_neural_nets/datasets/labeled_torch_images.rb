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
      # * *transforms* (Array): Array of TorchVision transforms to apply
      def initialize(dataset, transforms = [])
        super(dataset)
        @transforms = transforms
        @torch_dataset = RubyNeuralNets::Torch::SampleFolderDataset.new(@dataset, @transforms)
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
      #   * *rows* (Integer or nil): Number of rows if it applies to all images, or nil otherwise
      #   * *cols* (Integer or nil): Number of columns if it applies to all images, or nil otherwise
      #   * *channels* (Integer or nil): Number of channels if it applies to all images, or nil otherwise
      def image_stats
        stats = RubyNeuralNets::Datasets::ImagesFromFiles.new(@dataset).image_stats
        @transforms.each do |transform|
          if transform.is_a?(::TorchVision::Transforms::Resize)
            stats[:cols], stats[:rows] = transform.instance_variable_get(:@size)
          end
        end
        stats
      end

    end

  end

end
