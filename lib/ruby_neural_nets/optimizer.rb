module RubyNeuralNets

  # Base class for optimizers
  class Optimizer

    # Set the current epoch being processed
    #
    # Parameters::
    # * *idx_epoch* (Integer): The epoch being processed
    def start_epoch(idx_epoch)
      @idx_epoch = idx_epoch
    end

    # Initialize the optimizer's specific parameters of trainable tensors
    #
    # Parameters::
    # * *parameter* (Parameter): The parameter tensoe to initialize
    def init_parameter(parameter)
      # By default, nothing to do.
      # Subclasses can override that.
    end

  end

end
