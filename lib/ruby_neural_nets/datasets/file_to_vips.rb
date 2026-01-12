require 'ruby-vips'
require 'ruby_neural_nets/datasets/wrapper'
require 'ruby_neural_nets/sample'

module RubyNeuralNets

  module Datasets

    # Dataset of images read from files using Vips
    class FileToVips < Wrapper

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Sample: The sample containing input and target data
      def [](index)
        sample = @dataset[index]
        Sample.new(
          -> { Vips::Image.new_from_file(sample.input) },
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
        sample_image = Vips::Image.new_from_file(@dataset[0].input)
        {
          rows: sample_image.height,
          cols: sample_image.width,
          channels: sample_image.bands,
          depth:
            case sample_image.get('bits-per-sample')
            when 1, 8
              # Monochrome images are also treated as 8 bits
              8
            when 16
              16
            else
              raise "Unsupported bits per sample: #{sample_image.get('bits-per-sample')}"
            end
        }
      end

    end

  end

end
