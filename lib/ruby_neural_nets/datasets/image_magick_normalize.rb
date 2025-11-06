require 'ruby_neural_nets/datasets/wrapper'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that normalizes images to [0,1] range.
    class ImageMagickNormalize < Wrapper

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Object: The element X of the dataset
      # * Object: The element Y of the dataset
      def [](index)
        image, y = @dataset[index]
        [image.dispatch(0, 0, image.columns, image.rows, Helpers.image_pixels_map(image), true), y]
      end

      # Convert an element to an image
      #
      # Parameters::
      # * *element* (Object): The X element returned by [] method
      # Result::
      # * Object: The ImageMagick image representation
      def to_image(element)
        stats = image_stats
        rows = stats[:rows]
        cols = stats[:cols]
        channels = stats[:channels]
        # Convert Numo array to Ruby array and scale to 0-65535
        scaled_flat = ((element * 65535).round).to_a.flatten.map(&:to_i)
        # Create image and import pixels
        img = Magick::Image.new(cols, rows)
        img.import_pixels(
          0,
          0,
          cols,
          rows,
          channels == 1 ? 'I' : 'RGB',
          channels == 1 ? scaled_flat : (scaled_flat.each_slice(cols * channels).map { |row_flat| row_flat.each_slice(channels).to_a }.flatten)
        )
        img
      end

    end

  end

end
