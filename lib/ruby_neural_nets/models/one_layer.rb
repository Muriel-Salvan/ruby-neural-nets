require 'ruby_neural_nets/helpers'
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
      # * *optimizer* (Optimizer): Optimizer to be used
      def initialize(rows, cols, channels, nbr_classes, optimizer:)
        super
        n_x = rows * cols * channels
        # Define a very simple neural net with 1 softmax layer to categorize the 10 numbers
        # Softmax layer weights [nbr_classes, n_x]
        @w_1 = Numo::DFloat.new(nbr_classes, n_x).rand
        # Softmax layer bias [nbr_classes, 1]
        @b_1 = Numo::DFloat.zeros(nbr_classes, 1)
      end

      # Perform the forward propagation given an input layer
      #
      # Parameters::
      # * *x* (Numo::DFloat): The input layer
      # Result::
      # * Numo::DFloat: The last layer's output
      def forward_propagate(x)
        # Cache some variables
        @cache_x = x
        # Forward propagate the minibatch
        # Shape [nbr_classes, minibatch.size]
        z_1 = @w_1.dot(x) + @b_1
        # Shape [nbr_classes, minibatch.size]
        Helpers.softmax(z_1)
      end

      # Perform the gradient descent, given the predicted output and real one.
      # Prerequisite: forward_propagate must be called prior to this.
      #
      # Parameters::
      # * *da* (Numo::DFloat): The loss derivative from the model predicted output
      # * *a* (Numo::DFloat): The predicted output
      # * *y* (Numo::DFloat): The real output
      def gradient_descent(da, a, y)
        m = y.shape[1]
        # Backward propagate the minibatch
        # Shape [nbr_classes, minibatch.size]
        dz_1 = a - y
        # Shape [nbr_classes, n_x]
        dw_1 = dz_1.dot(@cache_x.transpose) / m
        # Shape [nbr_classes, 1]
        db_1 = dz_1.sum(axis: 1, keepdims: true) / m

        # Perform gradient descent on parameters
        @w_1 = learn(@w_1, dw_1)
        @b_1 = learn(@b_1, db_1)
      end

    end

  end

end