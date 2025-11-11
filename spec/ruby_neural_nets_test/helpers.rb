module RubyNeuralNetsTest
  module Helpers
    # Helper method to setup mocked filesystem using fakefs
    #
    # This method sets up a virtual filesystem using fakefs with the provided files,
    # and mocks Magick::ImageList.new to support loading images from the virtual filesystem.
    #
    # Parameters::
    # * *files_hash* (Hash<String, String>): A hash mapping file paths to their contents (as strings, e.g., PNG data blobs)
    #
    # The method creates directories as needed and writes the files to the fakefs.
    # It also mocks Magick::ImageList.new to read image data from the virtual filesystem
    # for any file path passed to it, allowing image loading in tests without real files.
    def self.setup_test_filesystem(files_hash)
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
    end
  end
end
