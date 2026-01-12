require 'torch'

# Make sure default tensors are using 64-bits floats (dtype = double)
module Torch
  class Tensor
    def self.new(*args)
      DoubleTensor.new(*args)
    end
  end
end

require 'ruby_neural_nets/logger'
require 'ruby_neural_nets/model'
require 'ruby_neural_nets/parameters/torch'

module RubyNeuralNets

  module Models

    # Base class for all Torch models
    class Torch < Model

      # Define a model for processing images and outputting a given number of classes
      #
      # Parameters::
      # * *rows* (Integer): Number of rows per image
      # * *cols* (Integer): Number of columns per image
      # * *channels* (Integer): Number of channels per image
      # * *nbr_classes* (Integer): Number of classes to identify
      # * *torch_net* (::Torch::NN::Module): Torch model that is handled by this model
      def initialize(rows, cols, channels, nbr_classes, torch_net:)
        super(rows, cols, channels, nbr_classes)
        @torch_net = torch_net
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
        debug do
          <<~EO_Debug
            Model buffers:
            #{
              @torch_net.
                named_children.
                select { |_layer_name, mod| !mod.named_buffers.empty? }.
                map do |layer_name, mod|
                  mod.named_buffers.map { |n, b| "* #{layer_name}.#{n}: #{data_to_str(b)}" }.join("\n")
                end.
                join("\n")
            }
          EO_Debug
        end
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
      # * *minibatch* (Minibatch): The minibatch containing real output and size
      # * *loss* (Object): The computed loss
      def gradient_descent(da, a, minibatch, loss)
        loss.backward
      end

    end

  end

end
