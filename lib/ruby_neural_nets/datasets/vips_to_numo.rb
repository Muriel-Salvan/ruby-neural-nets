require 'ruby_neural_nets/datasets/wrapper'
require 'ruby_neural_nets/sample'
require 'fiddle'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that converts Vips images to flattened Numo DFloat objects.
    class VipsToNumo < Wrapper

      # Access an element of the dataset
      #
      # Parameters:::
      # * *index* (Integer): Index of the dataset element to access
      # Result:::
      # * Sample: The sample containing input and target data
      def [](index)
        sample = @dataset[index]
        Sample.new(
          -> do
            # Convert Vips image to Numo DFloat without intermediate Ruby arrays
            (
              case @dataset.image_stats[:depth]
              when 8
                Numo::UInt8
              when 16
                Numo::UInt16
              else
                raise "Unsupported depth: #{@dataset.image_stats[:depth]}"
              end
            ).from_binary(sample.input.write_to_memory).cast_to(Numo::DFloat)
          end,
          -> { sample.target }
        )
      end

      # Convert an element to an image
      #
      # Parameters::
      # * *element* (Object): The X element returned by [] method
      # Result::
      # * Object: The Vips image representation
      def to_image(element)
        stats = image_stats
        rows = stats[:rows]
        cols = stats[:cols]
        channels = stats[:channels]
        # Convert Numo array to Ruby array
        scaled_flat = element.to_a.flatten.map(&:to_i)
        # Create Vips image from memory
        if channels == 1
          # Grayscale
          Vips::Image.new_from_memory(scaled_flat.pack('C*'), cols, rows, 1, :uchar)
        else
          # Color - interleave channels
          interleaved = []
          (0...rows).each do |r|
            (0...cols).each do |c|
              (0...channels).each do |ch|
                interleaved << scaled_flat[(r * cols + c) * channels + ch]
              end
            end
          end
          Vips::Image.new_from_memory(interleaved.pack('C*'), cols, rows, channels, :uchar)
        end
      end

    end

  end

end
