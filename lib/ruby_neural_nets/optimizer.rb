require 'numo/narray'
require 'ruby_neural_nets/logger'

module RubyNeuralNets

  # Base class for optimizers
  class Optimizer
    include Logger

    # Teach a given set of parameters
    #
    # Parameters::
    # * *parameters* (Array<Parameter>): Model's parameters that need to be learned
    def teach_parameters(parameters)
      parameters.each { |param| param.link_to_optimizer(self) }
    end

    # Set the current epoch being processed
    #
    # Parameters::
    # * *idx_epoch* (Integer): The epoch being processed
    def start_epoch(idx_epoch)
      @idx_epoch = idx_epoch
    end

    # Set the current minibatch being processed
    #
    # Parameters::
    # * *idx_minibatch* (Integer): The minibatch index being processed
    def start_minibatch(idx_minibatch)
      @idx_minibatch = idx_minibatch
    end

    # Handle a step after back-propogation
    def step
      # By default, nothing to do.
      # Subclasses can override that.
    end

    # Initialize the optimizer's specific parameters of trainable tensors
    #
    # Parameters::
    # * *parameter* (Parameter): The parameter tensor to initialize
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
      debug { "Learning #{parameter.name} with diff #{diff.mean}" }
      new_params = parameter.values - diff
      Helpers.check_instability(new_params)
      new_params
    end

  end

end
