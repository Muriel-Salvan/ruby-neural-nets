require 'ruby_neural_nets/datasets/file_to_vips'
require 'ruby_neural_nets/datasets/wrapper'
require 'ruby_neural_nets/torch/sample_folder_dataset'
require 'ruby_neural_nets/torchvision/transforms/image_magick_grayscale'
require 'ruby_neural_nets/torchvision/transforms/image_magick_resize'

module RubyNeuralNets

  module Datasets

    # Dataset serving TorchVision images.
    class LabeledTorchImages < Wrapper

      # Get the list of transforms
      attr_reader :transforms

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

      # Give access to the underlying files dataset
      #
      # Result::
      # * Dataset: Dataset serving files to Torch's folder dataset
      def files_dataset
        @dataset
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
        transformed_stats(RubyNeuralNets::Datasets::FileToVips.new(@dataset).image_stats, @transforms)
      end

      private

      # Get image stats from a existing image stats processed through a list of transformers
      #
      # Parameters::
      # * *stats* (Hash): Existing image stats
      # * *transforms* (Array<::Torch::NN::Module): List of transforms
      # Result::
      # * Hash: Resulting stats
      def transformed_stats(stats, transforms)
        new_stats = stats.clone
        transforms.each do |transform|
          case transform
          when TorchVision::Transforms::Cache
            new_stats = transformed_stats(new_stats, transform.transforms)
          when ::TorchVision::Transforms::Resize
            new_stats[:cols], new_stats[:rows] = transform.instance_variable_get(:@size)
          when TorchVision::Transforms::ImageMagickResize, TorchVision::Transforms::VipsResize
            new_stats[:cols], new_stats[:rows] = transform.size
          when TorchVision::Transforms::ImageMagickGrayscale, TorchVision::Transforms::VipsGrayscale
            new_stats[:channels] = 1
          when TorchVision::Transforms::VipsRemoveAlpha
            new_stats[:channels] -= 1
          end
        end
        new_stats
      end

    end

  end

end
