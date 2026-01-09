require 'torchvision'
require 'ruby_neural_nets/datasets/wrapper'
require 'ruby_neural_nets/torchvision/transforms/image_magick_grayscale'
require 'ruby_neural_nets/torchvision/transforms/image_magick_resize'
require 'ruby_neural_nets/torchvision/transforms/vips_grayscale'
require 'ruby_neural_nets/torchvision/transforms/vips_resize'
require 'ruby_neural_nets/torchvision/transforms/vips_remove_alpha'

module RubyNeuralNets

  module Datasets

    # Dataset transforming images using TorchVision
    class TorchTransformImages < Wrapper

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): Dataset giving underlying labeled data
      # * *transforms* (Array): Array of TorchVision transforms to apply
      def initialize(dataset, transforms = [])
        super(dataset)
        @transforms = transforms
        @torch_transform = ::TorchVision::Transforms::Compose.new(@transforms)
      end

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Object: The element X of the dataset
      # * Object: The element Y of the dataset
      def [](index)
        x, y = @dataset[index]
        [@torch_transform.call(x), y]
      end

      # Get some images stats.
      # Those are supposed to be the same for all samples from the dataset and can be used to compute the model's architecture.
      #
      # Result::
      # * Hash: Image stats:
      #   * *rows* (Integer or nil): Number of rows if it applies to all images, or nil otherwise
      #   * *cols* (Integer or nil): Number of columns if it applies to all images, or nil otherwise
      #   * *channels* (Integer or nil): Number of channels if it applies to all images, or nil otherwise
      #   * *depth* (Integer or nil): Depth (number of bits) used to encode pixel channel's values if it applies to all images, or nil otherwise
      def image_stats
        TorchTransformImages.transformed_stats(@dataset.image_stats, @transforms)
      end

      # Get image stats from a existing image stats processed through a list of transformers
      #
      # Parameters::
      # * *stats* (Hash): Existing image stats
      # * *transforms* (Array<::Torch::NN::Module): List of transforms
      # Result::
      # * Hash: Resulting stats
      def self.transformed_stats(stats, transforms)
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
          when TorchVision::Transforms::ImageMagickRemoveAlpha, TorchVision::Transforms::VipsRemoveAlpha
            new_stats[:channels] -= 1 if new_stats[:channels] == 2 || new_stats[:channels] == 4
          end
        end
        new_stats
      end

    end

  end

end
