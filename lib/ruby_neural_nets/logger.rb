require 'time'

module RubyNeuralNets

  # Logger mixin that provides logging capabilities to classes
  module Logger

    # Class-level variable to store the debug mode state
    @debug_mode = false

    class << self
      attr_accessor :debug_mode
    end

    # Log a message to stdout with ISO8601 UTC timestamp and caller class name
    #
    # Parameters::
    # * *message* (String): The message to log
    def log(message)
      puts "#{log_prefix} #{message}"
    end

    # Log a debug message to stdout only when debug mode is enabled.
    # Uses lazy evaluation to avoid computing the message when debug is disabled.
    # Includes ISO8601 UTC timestamp and caller class name.
    #
    # Parameters::
    # * *message_proc* (Proc): A proc that returns the message string when called
    def debug(&message_proc)
      return unless RubyNeuralNets::Logger.debug_mode
      
      puts "#{log_prefix} [DEBUG] #{message_proc.call}"
    end

    private

    # Generate the log message prefix with timestamp and class name
    #
    # Result::
    # * String: The log prefix
    def log_prefix
      "[#{Time.now.utc.iso8601}] [#{self.class.name.split('::').last}]"
    end

  end

end
