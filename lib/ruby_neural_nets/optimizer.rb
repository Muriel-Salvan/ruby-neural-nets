require 'numo/narray'

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

    # Adapt some parameters from a differential value to apply to them.
    #
    # Parameters::
    # * *parameter* (Parameter): Parameters to update
    # * *diff* (Numo::DFloat): Corresponding differential to apply to this parameter
    # Result::
    # * Numo::DFloat: New parameter values to take into account for next epoch
    def learn_from_diff(parameter, diff)
      puts "[Optimizer] - Learning #{parameter.name} with diff #{diff.mean}"
      new_params = parameter.values - diff
      Helpers.check_instability(new_params)
      new_params
    end

  end

end
