require 'ruby_neural_nets/datasets/wrapper'
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
      # * Object: The element X of the dataset
      # * Object: The element Y of the dataset
      def [](index)
        image, y = @dataset[index]
        # Convert Vips image to Numo DFloat without intermediate Ruby arrays
        [Numo::UInt8.from_binary(image.write_to_memory).cast_to(Numo::DFloat), y]
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
