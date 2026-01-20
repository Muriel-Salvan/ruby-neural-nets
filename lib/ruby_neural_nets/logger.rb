require 'time'
require 'logger'

module RubyNeuralNets

  # Logger mixin that provides logging capabilities to classes
  module Logger

    # Class-level variable to store the debug mode state
    @debug_mode = false

    # Class-level variable to store the logger instance
    @logger = ::Logger.new(STDOUT)

    class << self
      attr_accessor :debug_mode
      attr_accessor :logger
    end

    # Log a message using the configured logger with caller class name
    #
    # Parameters::
    # * *message* (String): The message to log
    def log(message)
      Logger.logger.info "#{log_prefix} #{message}"
    end

    # Log a debug message using the configured logger only when debug mode is enabled.
    # Uses lazy evaluation to avoid computing the message when debug is disabled.
    # Includes caller class name.
    #
    # Parameters::
    # * *message_proc* (Proc): A proc that returns a message string when called
    def debug(&message_proc)
      return unless Logger.debug_mode

      Logger.logger.debug "#{log_prefix} [DEBUG] #{message_proc.call}"
    end

    # Convert matrix data into a formatted string for logging
    #
    # Parameters::
    # * *data* (Object): The matrix data to format (Numo::DFloat, Torch::Tensor, etc.)
    # * *max_elements* (Integer): Maximum number of elements to display per dimension [default: 9]
    # Result::
    # * String: Formatted string representation of the matrix data
    def data_to_str(data, max_elements: 9)
      type, original_shape, size, stats, ruby_array_info =
        case data
        when Numo::DFloat, Numo::SFloat
          ["Numo::DFloat", data.shape, data.size, calculate_array_stats(data), extract_numo_subset(data, max_elements)]
        when ::Torch::Tensor
          ["Torch::Tensor", data.shape, data.numel, calculate_tensor_stats(data), extract_torch_subset(data, max_elements)]
        else
          raise "Unsupported data type: #{data.class} (value: #{data.inspect})"
        end
      "#{type} shape=#{original_shape.inspect} size=#{size}" +
        " stats={mean=#{stats[:mean]}, std=#{stats[:std]}, min=#{stats[:min]}, max=#{stats[:max]}}" +
        " values#{ruby_array_info[:truncated] ? '_truncated' : ''}=" +
        Numo::DFloat[*ruby_array_info[:array]].inspect
    end

    private

    # Calculate statistics for a Numo array
    #
    # Parameters::
    # * *array* (Numo::DFloat): The array to analyze
    # Result::
    # * Hash: Statistics hash
    def calculate_array_stats(array)
      {
        mean: array.mean.round(6),
        std: Math.sqrt(array.var).round(6),
        min: array.min.round(6),
        max: array.max.round(6)
      }
    end

    # Calculate statistics for a Torch tensor using tensor methods
    #
    # Parameters::
    # * *tensor* (Torch::Tensor): The tensor to analyze
    # Result::
    # * Hash: Statistics hash
    def calculate_tensor_stats(tensor)
      # Use tensor methods to avoid full conversion to Ruby array
      float_tensor = tensor.float
      {
        mean: float_tensor.mean.item.round(6),
        std: float_tensor.std.item.round(6),
        min: tensor.min.item.round(6),
        max: tensor.max.item.round(6)
      }
    end

    # Extract subset from Numo array (generic for any shape)
    #
    # Parameters::
    # * *array* (Numo::DFloat): The array to extract from
    # * *max_elements* (Integer): Maximum elements per dimension
    # Result::
    # * Hash: Contains Ruby array and truncation info
    def extract_numo_subset(array, max_elements)
      shape = array.shape
      total_elements = array.size

      # Calculate new shape with truncation
      new_shape = shape.map { |dim| [dim, max_elements].min }

      # Build slicing arguments dynamically
      slicing_args = new_shape.map { |dim| 0...dim }

      # Extract subset while preserving shape
      subset_array = array[*slicing_args]

      ruby_array = subset_array.to_a
      { array: ruby_array, truncated: total_elements > ruby_array.flatten.size }
    end

    # Extract subset from Torch tensor (generic for any shape)
    #
    # Parameters::
    # * *tensor* (Torch::Tensor): The tensor to extract from
    # * *max_elements* (Integer): Maximum elements per dimension
    # Result::
    # * Hash: Contains Ruby array and truncation info
    def extract_torch_subset(tensor, max_elements)
      shape = tensor.shape
      return { array: tensor.to_a, truncated: false } if shape.empty?

      total_elements = tensor.numel

      # Calculate new shape with truncation
      new_shape = shape.map { |dim| [dim, max_elements].min }

      # Build slicing arguments dynamically
      slicing_args = new_shape.map { |dim| 0...dim }

      # Extract subset while preserving shape
      subset_tensor = tensor[*slicing_args]

      ruby_array = subset_tensor.to_a
      { array: ruby_array, truncated: total_elements > ruby_array.flatten.size }
    end

    # Generate a log message prefix with class name
    #
    # Result::
    # * String: The log prefix
    def log_prefix
      "[#{self.class.name.split('::').last}]"
    end

  end

end
