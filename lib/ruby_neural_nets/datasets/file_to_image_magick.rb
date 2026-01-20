require 'rmagick'
require 'streamio-ffmpeg'
require 'tmpdir'
require 'stringio'
require 'ruby_neural_nets/datasets/wrapper'
require 'ruby_neural_nets/sample'

# Monkey patch Open3 and IO to silence ffmpeg output
require 'open3'

# Patch Open3.capture3
module Open3
  class << self
    alias :original_capture3 :capture3
    def capture3(*cmd, **opts)
      # Check if this is an ffmpeg command
      cmd_str = Array(cmd).flatten.join(' ')
      if cmd_str.include?('ffmpeg')
        stdout, stderr, status = original_capture3(*cmd, **opts)
        # Return empty stdout but keep stderr and status
        ['', stderr, status]
      else
        original_capture3(*cmd, **opts)
      end
    end
  end
end

# Patch IO.popen
module IO
  class << self
    alias :original_popen :popen
    def popen(cmd, mode = 'r', **opts)
      # Check if this is an ffmpeg command
      cmd_str = cmd.is_a?(Array) ? cmd.join(' ') : cmd.to_s
      if cmd_str.include?('ffmpeg') && mode.include?('r')
        # For reading from ffmpeg, redirect stdout to /dev/null but keep stderr
        # This is a bit tricky, let's try to modify the command
        if cmd.is_a?(Array)
          modified_cmd = cmd + ['2>&1', '>/dev/null']
          original_popen(modified_cmd, mode, **opts)
        else
          modified_cmd = "#{cmd} 2>&1 >/dev/null"
          original_popen(modified_cmd, mode, **opts)
        end
      else
        original_popen(cmd, mode, **opts)
      end
    end
  end
end

module RubyNeuralNets

  module Datasets

    # Dataset of images read from files using ImageMagick
    class FileToImageMagick < Wrapper

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): The dataset containing file paths as inputs
      # * *video_slices_sec* (Float): Number of seconds of each video slice used to extract images from MP4 files
      def initialize(dataset, video_slices_sec:)
        super(dataset)
        @video_slices_sec = video_slices_sec
        @index_mapping = build_index_mapping
      end

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Sample: The sample containing input and target data
      def [](index)
        file_info = @index_mapping[index]
        sample = @dataset[file_info[:original_index]]
        
        Sample.new(
          -> { get_image(file_info) },
          -> { sample.target }
        )
      end

      # Get the size of the dataset
      #
      # Result::
      # * Integer: Number of elements in the dataset
      def size
        @index_mapping.size
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
        sample_image = get_image(@index_mapping.first)
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

      private

      # Execute a block while silencing stdout, but keeping stderr visible for errors
      #
      # Parameters::
      # * *block* (Proc): The block to execute
      # Result::
      # * Object: The result of the block execution
      def silence_output(&block)
        original_stdout = $stdout
        $stdout = StringIO.new
        begin
          yield
        ensure
          $stdout = original_stdout
        end
      end

      # Build the index mapping for all files
      #
      # Result::
      # * Array<Hash>: Array of file info hashes with file_path and time_offset
      def build_index_mapping
        mapping = []
        @dataset.each_with_index do |sample, original_index|
          file_path = sample.input
          extension = File.extname(file_path).downcase
          case extension
          when '.png'
            mapping << {
              file_path:,
              original_index:
            }
          when '.mp4'
            (FFMPEG::Movie.new(file_path).duration / @video_slices_sec).ceil.times do |slice_idx|
              mapping << {
                file_path:,
                time_offset: slice_idx * @video_slices_sec,
                original_index:
              }
            end
          else
            raise "Unsupported file extension: #{extension}."
          end
        end
        mapping
      end

      # Get an image from a file, handling both PNG and MP4
      #
      # Parameters::
      # * *file_info* (Hash): File information
      # Result::
      # * Magick::Image: The loaded image
      def get_image(file_info)
        if file_info.key?(:time_offset)
          # Use streamio-ffmpeg to extract the frame in memory
          video = FFMPEG::Movie.new(file_info[:file_path])
          # Create a temporary file for the screenshot
          Dir.mktmpdir do |temp_dir|
            temp_file = "#{temp_dir}/frame.png"
            # Take screenshot at the specified time
            video.screenshot(temp_file, seek_time: file_info[:time_offset])
            # Load the screenshot with ImageMagick
            Magick::ImageList.new(temp_file).first
          end
        else
          Magick::ImageList.new(file_info[:file_path]).first
        end
      end

    end

  end

end
