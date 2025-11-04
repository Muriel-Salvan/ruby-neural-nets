require 'ruby-vips'
require 'ruby_neural_nets/datasets/wrapper'

module RubyNeuralNets

  module Datasets

    # Dataset of images read from files using Vips
    class FileToVips < Wrapper

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Object: The element X of the dataset
      # * Object: The element Y of the dataset
      def [](index)
        file, y = @dataset[index]
        [Vips::Image.new_from_file(file), y]
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
        # Assume all images have the same properties as the first one
        sample_image = Vips::Image.new_from_file(@dataset[0][0])
        {
          rows: sample_image.height,
          cols: sample_image.width,
          channels: sample_image.bands
        }
      end

    end

  end

end
