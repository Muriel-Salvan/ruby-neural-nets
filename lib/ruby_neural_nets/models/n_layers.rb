require 'ruby_neural_nets/model'
require 'ruby_neural_nets/models/layers/dense'
require 'ruby_neural_nets/models/layers/sigmoid'
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
          [layer, Layers::Sigmoid.new(model: self, n_x:)]
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
          puts "[Model/N-Layers] - Forward propagate #{a.shape[1]} => #{layer.class.name}..."
          a = layer.forward_propagate(a)
          byebug
          puts "[Model/N-Layers] - Forward propagate #{layer.class.name} => #{a.shape[1]}."
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
        @layers.reverse.each { |layer| da = layer.backward_propagate(da) }
      end

    end

  end

end