require 'numo/narray'
require 'ruby_neural_nets/helpers'
require 'ruby_neural_nets/model'
require 'ruby_neural_nets/models/layers/batch_normalization'
require 'ruby_neural_nets/models/layers/dense'
require 'ruby_neural_nets/models/layers/leaky_relu'
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
      # * *layers* (Array<Integer>): List of hidden units per layer before the last one
      def initialize(rows, cols, channels, nbr_classes, layers:)
        super(rows, cols, channels, nbr_classes)
        @n_x = rows * cols * channels
        # Define a simple neural net with layers densily connected with sigmoid activation + 1 softmax layer to classify at the end
        @layers = []
        layers.each do |nbr_units|
          self << Layers::Dense.new(nbr_units:)
          self << Layers::BatchNormalization.new
          self << Layers::LeakyRelu.new
        end
        self << Layers::Dense.new(nbr_units: nbr_classes)
        self << Layers::BatchNormalization.new
        self << Layers::Softmax.new
      end

      # Add a layer
      #
      # Parameters::
      # * *layer* (Layer): Layer to be added
      def <<(layer)
        layer.link_to_model(
          model: self,
          n_x: @layers.empty? ? @n_x : @layers.last.n_y,
          idx_layer: @layers.size
        )
        @layers << layer
      end

      # Initialize the back propagation cache.
      # This is called before forward propagating.
      def initialize_back_propagation_cache
        @back_propagation_cache = {
          layers: @layers.size.times.map { {} }
        }
      end

      # Perform the forward propagation given an input layer
      #
      # Parameters::
      # * *x* (Object): The input layer
      # * *train* (Boolean): Are we in training mode? [default: false]
      # Result::
      # * Object: The corresponding output layer
      def forward_propagate(x, train: false)
        # Forward propagate the minibatch
        a = x
        @layers.each do |layer|
          n_x = a.shape[0]
          a = layer.forward_propagate(a)
          debug { "Forward propagate #{n_x} => #{layer.class.name.split('::').last} => #{a.shape[0]}." }
          debug { "=> Output data: #{data_to_str(a)}" }
          Helpers.check_instability(a)
          # Keep intermediate activations for debugging.
          # This could be removed in case memory becomes an issue.
          layer.instance_variable_set(:@output, a)
        end
        a
      end

      # Perform the gradient descent, given the predicted output and real one.
      # Prerequisite: forward_propagate must be called prior to this.
      #
      # Parameters::
      # * *da* (Object): The loss derivative from the model predicted output
      # * *a* (Object): The predicted output
      # * *y* (Object): The real output
      # * *loss* (Object): The computed loss
      # * *minibatch_size* (Integer): Minibatch size
      def gradient_descent(da, a, y, loss, minibatch_size)
        # Backward propagate the minibatch
        da = da / minibatch_size
        @layers.reverse.each do |layer|
          Helpers.check_instability(da)
          n_x = da.shape[0]
          da = layer.backward_propagate(da)
          debug { "Backward propagate #{n_x} => #{layer.class.name.split('::').last} => #{da.shape[0]}." }
        end
      end

    end

  end

end
