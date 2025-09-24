require 'numo/narray'
require 'ruby_neural_nets/initializers/glorot'
require 'ruby_neural_nets/initializers/zero'
require 'ruby_neural_nets/models/layer'

module RubyNeuralNets

  module Models

    module Layers

      # Dense layer outputing a given number of units
      class Dense < Layer

        # Constructor
        #
        # Parameters::
        # * *nbr_units* (Integer): The number of units for this dense layer
        def initialize(nbr_units:)
          @nbr_units = nbr_units
        end

        # Initialize parameters.
        # This method is optional and is always called once a layer is linked to a model.
        def initialize_parameters
          # Layer weights [nbr_units, n_x]
          # Use the Xavier Glorot normal initialization to avoid exploding gradients
          @w = register_parameters([@nbr_units, @n_x], Initializers::Glorot, name: "L#{@idx_layer}_Dense_W")
          # Layer bias [nbr_units, 1]
          @b = register_parameters([@nbr_units, 1], Initializers::Zero, name: "L#{@idx_layer}_Dense_B")
        end

        # Get the output dimension of this layer
        #
        # Result::
        # * Integer: Output dimension of the layer
        def n_y
          @nbr_units
        end

        # Forward propagate an input through this layer
        #
        # Parameters::
        # * *input* (Numo::DFloat): The input
        # Result::
        # * Numo::DFloat: The corresponding layer output
        def forward_propagate(input)
          back_propagation_cache[:input] = input
          @w.values.dot(input) + @b.values
        end

        # Backward propagate an input da (coming from next layers) through this layer.
        # Some examples of the purpose of this function:
        # * da * g'(z) in the case of an activation layer using a function g.
        # * W.transpose.dot(da) in the case of a dense layer.
        # Learning and optiomization can be called on eventual parameters by calling w = learn(w, dw).
        #
        # Parameters::
        # * *da* (Numo::DFloat): The input da coming from the next layers
        # Result::
        # * Numo::DFloat: The corresponding layer output da
        def backward_propagate(da)
          # Compute gradient to pass to previous layer using current weights before updating
          prev_da = @w.values.transpose.dot(da)
          # Update parameters after computing prev_da to avoid using updated weights in the backward chain
          @w.learn(da.dot(back_propagation_cache[:input].transpose))
          @b.learn(da.sum(axis: 1, keepdims: true))
          prev_da
        end

      end

    end

  end

end
