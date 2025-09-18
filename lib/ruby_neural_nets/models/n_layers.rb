require 'ruby_neural_nets/model'
require 'ruby_neural_nets/models/layers/dense'
require 'ruby_neural_nets/models/layers/relu'
require 'ruby_neural_nets/models/layers/softmax'

module RubyNeuralNets

  module Models

    class NLayers < Model

      # Define a model for processing images and outputting a given number of classes
      #
      # Parameters::
      # * *rows* (Integer): Number of rows per image
      # * *cols* (Integer): Number of columns per image
      # * *channels* (Integer): Number of channels per image
      # * *nbr_classes* (Integer): Number of classes to identify
      # * *optimizer* (Optimizer): Optimizer to be used
      # * *layers* (Array<Integer>): List of hidden units per layer before the last one
      def initialize(rows, cols, channels, nbr_classes, optimizer:, layers:)
        super(rows, cols, channels, nbr_classes, optimizer:)
        n_x = rows * cols * channels
        # Define a simple neural net with layers densily connected with sigmoid activation + 1 softmax layer to classify at the end
        @layers = layers.map do |nbr_units|
          layer = Layers::Dense.new(model: self, n_x:, nbr_units:)
          n_x = nbr_units
          [layer, Layers::Relu.new(model: self, n_x:)]
        end.flatten(1) + [
          Layers::Dense.new(model: self, n_x:, nbr_units: nbr_classes),
          Layers::Softmax.new(model: self, n_x: nbr_classes)
        ]
      end

      # Perform the forward propagation given an input layer
      #
      # Parameters::
      # * *x* (Numo::DFloat): The input layer
      # Result::
      # * Numo::DFloat: The last layer's output
      def forward_propagate(x)
        # Forward propagate the minibatch
        a = x
        @layers.each do |layer|
          n_x = a.shape[0]
          a = layer.forward_propagate(a)
          puts "[Model/N-Layers] - Forward propagate #{n_x} => #{layer.class.name.split('::').last} => #{a.shape[0]}."
          # Keep intermediate activations for debugging.
          # This could be removed in case memory becomes an issue.
          layer.instance_variable_set(:@output, a)
          # Check for numerical instability.
          # This could be removed in case processing power becomes an issue.
          puts '[Model/N-Layers] !!! Forward propagation has invalid values. There is numerical instability. !!!' unless a.isfinite.all?
        end
        a
      end

      # Perform the gradient descent, given the predicted output and real one.
      # Prerequisite: forward_propagate must be called prior to this.
      #
      # Parameters::
      # * *da* (Numo::DFloat): The loss derivative from the model predicted output
      # * *a* (Numo::DFloat): The predicted output
      # * *y* (Numo::DFloat): The real output
      def gradient_descent(da, a, y)
        # Backward propagate the minibatch
        @layers.reverse.each do |layer|
          # Check for numerical instability.
          # This could be removed in case processing power becomes an issue.
          puts '[Model/N-Layers] !!! Backward propagation has invalid values. There is numerical instability. !!!' unless da.isfinite.all?
          n_x = da.shape[0]
          da = layer.backward_propagate(da)
          puts "[Model/N-Layers] - Backward propagate #{n_x} => #{layer.class.name.split('::').last} => #{da.shape[0]}."
        end
      end

    end

  end

end