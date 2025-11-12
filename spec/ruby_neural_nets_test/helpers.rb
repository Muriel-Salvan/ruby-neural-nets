require 'fileutils'
require 'rmagick'

module RubyNeuralNetsTest
  module Helpers

    # Helper method to generate PNG file content as a string.
    #
    # This method creates a PNG image with the specified dimensions and fills it with a uniform color.
    # The color parameter is an array representing the channel values (e.g., [gray] for grayscale, [r,g,b] for RGB).
    #
    # Parameters::
    # * *width* (Integer): The width of the image in pixels
    # * *height* (Integer): The height of the image in pixels
    # * *color* (Array<Integer>): An array of channel values (e.g., [gray_value] for grayscale, [r,g,b] for RGB)
    # Result::
    # * String: The PNG file content as a binary string
    def png(width, height, color)
      # Create a new image
      image = Magick::Image.new(width, height) do |img|
        img.format = 'PNG'
      end

      # Determine pixel format based on color array length
      case color.size
      when 1
        # Grayscale
        image.import_pixels(0, 0, width, height, 'I', Array.new(width * height, color[0]))
      when 3
        # RGB
        image.import_pixels(0, 0, width, height, 'RGB', color * (width * height))
      else
        raise ArgumentError, "Unsupported color array length: #{color.size}"
      end

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

      FakeFS do
        # TODO: Understand why this step is needed. Normally FakeFS block should clear the file system and ensure isolation
        FakeFS::FileSystem.clear
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
    
  end
end
