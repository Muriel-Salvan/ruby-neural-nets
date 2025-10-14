require 'torch'

require 'ruby_neural_nets/model'
require 'ruby_neural_nets/parameters/torch'

module RubyNeuralNets

  module Models

    class NLayersTorch < Model

      class DenseStackNet < Torch::NN::Module

        # Constructor
        #
        # Parameters::
        # * *n_x* (Integer): Input dimension
        # * *layers* (Array<Integer>): List of hidden units per layer before the last one
        # * *nbr_classes* (Integer): Number of classes to identify
        def initialize(n_x:, layers:, nbr_classes:)
          super()
          @layers = layers.map.with_index do |nbr_units, idx_layer|
            linear_module = ::Torch::NN::Linear.new(n_x, nbr_units)
            ::Torch::NN::Init.xavier_uniform!(linear_module.weight)
            ::Torch::NN::Init.zeros!(linear_module.bias)
            layers_group = [
              add_module("l#{idx_layer}_linear", linear_module),
              add_module("l#{idx_layer}_batch_norm1d", ::Torch::NN::BatchNorm1d.new(nbr_units, eps: 1e-8)),
              add_module("l#{idx_layer}_leaky_relu", ::Torch::NN::LeakyReLU.new)
            ]
            n_x = nbr_units
            layers_group
          end.flatten(1)
          final_linear_module = ::Torch::NN::Linear.new(n_x, nbr_classes)
          ::Torch::NN::Init.xavier_uniform!(final_linear_module.weight)
          ::Torch::NN::Init.zeros!(final_linear_module.bias)
          @layers.concat(
            [
              add_module("l#{layers.size}_linear", final_linear_module),
              add_module("l#{layers.size}_batch_norm1d", ::Torch::NN::BatchNorm1d.new(nbr_classes))
            ]
          )
        end

        def forward(x)
          a = x
          @layers.each do |layer|
            n_x = a.shape[1]
            a = layer.call(a)
            puts "[Model/N-LayersTorch] - Forward propagate #{n_x} => #{layer.class.name.split('::').last} => #{a.shape[1]}."
          end
          a
        end

      end

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
        @torch_net = DenseStackNet.new(n_x: rows * cols * channels, layers:, nbr_classes:)
        @parameters = @torch_net.named_parameters.map { |name, torch_parameter| Parameters::Torch.new(name:, torch_parameter:) }
      end

      # Initialize the back propagation cache.
      # This is called before forward propagating.
      def initialize_back_propagation_cache
        # Nothing to do as back-propagation is handled internally
      end

      # Perform the forward propagation given an input layer
      #
      # Parameters::
      # * *x* (Object): The input layer
      # * *train* (Boolean): Are we in training mode? [default: false]
      # Result::
      # * Object: The corresponding output layer
      def forward_propagate(x, train: false)
        if train
          @torch_net.train
        else
          @torch_net.eval
        end
        @torch_net.call(x)
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
        loss.backward
      end

    end

  end

end