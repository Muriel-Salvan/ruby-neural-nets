require 'numo/narray'
require 'ruby_neural_nets/helpers'
require 'ruby_neural_nets/initializers/glorot_normal'
require 'ruby_neural_nets/initializers/zero'
require 'ruby_neural_nets/model'

module RubyNeuralNets

  module Models

    class OneLayer < Model

      # Define a model for processing images and outputting a given number of classes
      #
      # Parameters::
      # * *rows* (Integer): Number of rows per image
      # * *cols* (Integer): Number of columns per image
      # * *channels* (Integer): Number of channels per image
      # * *nbr_classes* (Integer): Number of classes to identify
      def initialize(rows, cols, channels, nbr_classes)
        super
        n_x = rows * cols * channels
        # Define a very simple neural net with 1 softmax layer to categorize the 10 numbers
        # Softmax layer weights [nbr_classes, n_x]
        @w_1 = register_parameters([nbr_classes, n_x], Initializers::GlorotNormal, name: 'W')
        # Softmax layer bias [nbr_classes, 1]
        @b_1 = register_parameters([nbr_classes, 1], Initializers::Zero, name: 'B')
      end

      # Perform the forward propagation given an input layer
      #
      # Parameters::
      # * *x* (Object): The input layer
      # * *train* (Boolean): Are we in training mode? [default: false]
      # Result::
      # * Object: The corresponding output layer
      def forward_propagate(x, train: false)
        # Cache some variables
        @back_propagation_cache[:x] = x if train
        # Forward propagate the minibatch
        # Shape [nbr_classes, minibatch.size]
        z_1 = @w_1.values.dot(x) + @b_1.values
        Helpers.check_instability(z_1)
        # Shape [nbr_classes, minibatch.size]
        output = Helpers.softmax(z_1)
        Helpers.check_instability(output, types: %i[not_finite zero one])
        output
      end

      # Perform the gradient descent, given the predicted output and real one.
      # Prerequisite: forward_propagate must be called prior to this.
      #
      # Parameters::
      # * *da* (Object): The loss derivative from the model predicted output
      # * *a* (Object): The predicted output
      # * *minibatch* (RubyNeuralNets::Minibatch): The minibatch containing real output and size
      # * *loss* (Object): The computed loss
      def gradient_descent(da, a, minibatch, loss)
        m = minibatch.size
        target = minibatch.target
        # Backward propagate the minibatch
        # For softmax + cross-entropy, use the combined derivative
        # dJ/dz = a - target (mathematically equivalent to the full chain rule)
        # Shape [nbr_classes, minibatch.size]
        dz_1 = a - target

        # Shape [nbr_classes, n_x]
        dw_1 = dz_1.dot(@back_propagation_cache[:x].transpose) / m
        Helpers.check_instability(dw_1)
        # Shape [nbr_classes, 1]
        db_1 = dz_1.sum(axis: 1, keepdims: true) / m
        Helpers.check_instability(db_1)

        # Perform gradient descent on parameters
        @w_1.learn(dw_1)
        @b_1.learn(db_1)
      end

    end

  end

end
