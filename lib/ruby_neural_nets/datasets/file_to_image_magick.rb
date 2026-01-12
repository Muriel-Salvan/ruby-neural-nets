require 'rmagick'
require 'ruby_neural_nets/datasets/wrapper'
require 'ruby_neural_nets/sample'

module RubyNeuralNets

  module Datasets

    # Dataset of images read from files using ImageMagick
    class FileToImageMagick < Wrapper

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Sample: The sample containing input and target data
      def [](index)
        sample = @dataset[index]
        Sample.new(
          -> { Magick::ImageList.new(sample.input).first },
          -> { sample.target }
        )
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
        # Assume all images have the same properties as the first one
        sample_image = Magick::ImageList.new(@dataset[0].input).first
        {
          rows: sample_image.rows,
          cols: sample_image.columns,
          channels:
            case sample_image.colorspace
            when Magick::GRAYColorspace
              1
            when Magick::RGBColorspace, Magick::SRGBColorspace
              3
            else
              raise "Unknown colorspace: #{sample_image.colorspace}"
            end,
          depth: sample_image.quantum_depth
        }
      end

    end

  end

end
