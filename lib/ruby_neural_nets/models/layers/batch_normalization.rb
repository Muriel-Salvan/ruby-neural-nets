require 'numo/narray'
require 'ruby_neural_nets/initializers/one'
require 'ruby_neural_nets/initializers/zero'
require 'ruby_neural_nets/models/layer'

module RubyNeuralNets

  module Models

    module Layers

      # Batch normalization layer
      class BatchNormalization < Layer

        # Constructor
        #
        # Parameters::
        # * *model* (Model): Model that is using this layer
        # * *n_x* (Integer): Number of input units
        def initialize(model:, n_x:)
          super(model:, n_x:)
          # Gamma parameter [n_x, 1], initialized to 1
          @gamma = register_parameters([n_x, 1], Initializers::One)
          # Beta parameter [n_x, 1], initialized to 0
          @beta = register_parameters([n_x, 1], Initializers::Zero)
          @epsilon = 1e-8
        end

        # Forward propagate an input through this layer
        #
        # Parameters::
        # * *input* (Numo::DFloat): The input [n_x, m]
        # Result::
        # * Numo::DFloat: The corresponding layer output
        def forward_propagate(input)
          @cache[:input] = input
          mean = input.mean(axis: 1, keepdims: true)
          @cache[:mean] = mean
          var = ((input - mean) ** 2).mean(axis: 1, keepdims: true)
          @cache[:var] = var
          sqrt_var_eps = Numo::NMath.sqrt(var + @epsilon)
          @cache[:sqrt_var_eps] = sqrt_var_eps
          x_hat = (input - mean) / sqrt_var_eps
          @cache[:x_hat] = x_hat
          @gamma.values * x_hat + @beta.values
        end

        # Backward propagate an input da (coming from next layers) through this layer.
        #
        # Parameters::
        # * *da* (Numo::DFloat): The input da coming from the next layers
        # Result::
        # * Numo::DFloat: The corresponding layer output da
        def backward_propagate(da)
          m = @cache[:input].shape[1]
          x_hat = @cache[:x_hat]
          mean = @cache[:mean]
          var = @cache[:var]
          sqrt_var_eps = @cache[:sqrt_var_eps]
          input = @cache[:input]

          # Gradients for beta and gamma
          dbeta = da.sum(axis: 1, keepdims: true)
          dgamma = (da * x_hat).sum(axis: 1, keepdims: true)

          # Gradient for x_hat (using current gamma)
          dx_hat = da * @gamma.values

          # Gradient for var
          dvar = (dx_hat * (input - mean) * -0.5 * (var + @epsilon) ** -1.5).sum(axis: 1, keepdims: true)

          # Gradient for mean
          dmean = (dx_hat * -1 / sqrt_var_eps).sum(axis: 1, keepdims: true) + dvar * -2 * ((input - mean).sum(axis: 1, keepdims: true) / m)

          # Gradient for input to pass to previous layers
          prev_da = dx_hat / sqrt_var_eps + dvar * 2 * (input - mean) / m + dmean / m

          # Update parameters after computing prev_da (to avoid influencing backward chain)
          @beta.learn(dbeta)
          @gamma.learn(dgamma)

          prev_da
        end

      end

    end

  end

end
