module RubyNeuralNets

  # Logger mixin that provides logging capabilities to classes
  module Logger

    # Class-level variable to store the debug mode state
    @debug_mode = false

    class << self
      attr_accessor :debug_mode
    end

    # Log a message to stdout
    #
    # Parameters::
    # * *message* (String): The message to log
    def log(message)
      puts message
    end

    # Log a debug message to stdout only when debug mode is enabled.
    # Uses lazy evaluation to avoid computing the message when debug is disabled.
    #
    # Parameters::
    # * *message_proc* (Proc): A proc that returns the message string when called
    def debug(&message_proc)
      return unless RubyNeuralNets::Logger.debug_mode
      
      message = message_proc.call
      puts "[DEBUG] #{message}"
    end

  end

end
