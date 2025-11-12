require 'fileutils'
require 'rmagick'

module RubyNeuralNetsTest
  module Helpers

    # Helper method to generate PNG file content as a string.
    #
    # This method creates a PNG image with the specified dimensions and fills it with pixel values.
    # The pixels parameter can be an array of pixel values or a hash with color key for uniform color.
    #
    # Parameters::
    # * *width* (Integer): The width of the image in pixels
    # * *height* (Integer): The height of the image in pixels
    # * *pixels* (Array<Integer> or Hash): Array of pixel values or Hash to describe the image's features.
    #   When used as a Hash, the following properties are possible:
    #   * *color* (Array<Integer>): A color description (as many values as desired channels) that fills the whole image.
    # Result::
    # * String: The PNG file content as a binary string
    def png(width, height, pixels)
      # Create a new image
      image = Magick::Image.new(width, height) do |img|
        img.format = 'PNG'
      end

      pixel_map, pixel_values =
        if pixels.is_a?(Hash)
          # Uniform color
          color = pixels[:color]
          case color.size
          when 1
            # Grayscale
            ['I', Array.new(width * height, color[0])]
          when 3
            # RGB
            ['RGB', color * (width * height)]
          else
            raise ArgumentError, "Unsupported color array length: #{color.size}"
          end
        else
          # Pixel values array or uniform color
          case pixels.size
          when width * height
            # Grayscale pixels
            ['I', pixels]
          when width * height * 3
            # RGB pixels
            ['RGB', pixels]
          else
            raise ArgumentError, "Unsupported pixels array length: #{pixels.size} for an image of size #{width}x#{height}"
          end
        end
      image.import_pixels(0, 0, width, height, pixel_map, pixel_values)

      image.to_blob
    end

    # Helper method to setup mocked filesystem using fakefs
    #
    # This method sets up a virtual filesystem using fakefs with the provided files,
    # and mocks Magick::ImageList.new to support loading images from the virtual filesystem.
    #
    # Parameters::
    # * *files_hash* (Hash<String, String>): A hash mapping file paths to their contents (as strings, e.g., PNG data blobs)
    # * Proc: Code block to execute within the fake filesystem context
    #
    # The method creates directories as needed and writes the files to the fakefs.
    # It also mocks Magick::ImageList.new to read image data from the virtual filesystem
    # for any file path passed to it, allowing image loading in tests without real files.
    def with_test_fs(files_hash)
      require 'fakefs/spec_helpers'

      FakeFS.with_fresh do
        # TODO: Understand why this step is needed. Normally FakeFS block should clear the file system and ensure isolation
        # FakeFS::FileSystem.clear
        files_hash.each do |path, content|
          dir = File.dirname(path)
          FileUtils.mkdir_p(dir) unless Dir.exist?(dir)
          File.write(path, content)
        end

        # Mock Magick::ImageList.new to return images from the fakefs files
        Magick::ImageList.define_singleton_method(:new) do |*args|
          # Read the PNG data from fakefs
          png_data = File.read(args.first)
          # Create image from blob
          image = Magick::Image.from_blob(png_data).first
          # Create ImageList and add the image
          image_list = Magick::ImageList.allocate
          image_list.send(:initialize)
          image_list << image
          image_list
        end

        yield
      end
    end

    # Compute distance between 2 arrays, element-wise
    #
    # Parameters::
    # * *array_1* (Array or Numo::DFloat): First array
    # * *array_2* (Array or Numo::DFloat): Second array
    # Result::
    # * Array: The distance
    def array_distance(array_1, array_2)
      array_1.to_a.zip(array_2.to_a).map { |(e_1, e_2)| (e_1 - e_2).abs }
    end

    # Expect a given array to have all its elements within the range of anothr array
    #
    # Parameters::
    # * *array_1* (Array or Numo::DFloat): First array
    # * *array_2* (Array or Numo::DFloat): Second array
    # * *threshold* (Float): Threshold range [default: 0.01]
    def expect_array_within(array_1, array_2, threshold = 0.01)
      expect(array_distance(array_1, array_2).max).to be_within(threshold).of(0)
    end

    # Expect a given array not to have all its elements within the range of anothr array
    #
    # Parameters::
    # * *array_1* (Array or Numo::DFloat): First array
    # * *array_2* (Array or Numo::DFloat): Second array
    # * *threshold* (Float): Threshold range [default: 0.01]
    def expect_array_not_within(array_1, array_2, threshold = 0.01)
      expect(array_distance(array_1, array_2).max).not_to be_within(threshold).of(0)
    end

    # Get a byebug session with fakefs
    def self.fakefs_byebug
      FakeFS::FileSystem.clone("#{__dir__}/../../lib")
      FakeFS::FileSystem.clone("#{__dir__}/../../spec")
      FakeFS::FileSystem.clone("#{__dir__}/../../vendor")
      byebug
    end
    
    # Get a byebug session with fakefs
    def fakefs_byebug
      RubyNeuralNetsTest::Helpers.fakefs_byebug
    end

  end

end
