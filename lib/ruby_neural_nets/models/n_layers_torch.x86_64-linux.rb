require 'ruby_neural_nets/logger'
require "ruby_neural_nets/models/torch.#{RUBY_PLATFORM}"

module RubyNeuralNets

  module Models

    class NLayersTorch < Torch

      class DenseStackNet < ::Torch::NN::Module
        include Logger

        # Constructor
        #
        # Parameters::
        # * *n_x* (Integer): Input dimension
        # * *layers* (Array<Integer>): List of hidden units per layer before the last one
        # * *nbr_classes* (Integer): Number of classes to identify
        def initialize(n_x:, layers:, nbr_classes:)
          super()
          @layers = layers.map.with_index do |nbr_units, idx_layer|
            linear_module = ::Torch::NN::Linear.new(n_x, nbr_units, bias: false)
            ::Torch::NN::Init.xavier_normal!(linear_module.weight)
            batch_norm_module = ::Torch::NN::BatchNorm1d.new(nbr_units, track_running_stats: false)
            batch_norm_module.register_buffer("running_mean", ::Torch.zeros(nbr_units, dtype: :double))
            batch_norm_module.register_buffer("running_var", ::Torch.ones(nbr_units, dtype: :double))
            n_x = nbr_units
            [
              add_module("l#{idx_layer}_linear", linear_module),
              add_module("l#{idx_layer}_batch_norm1d", batch_norm_module),
              add_module("l#{idx_layer}_leaky_relu", ::Torch::NN::LeakyReLU.new)
            ]
          end.flatten(1)
          final_linear_module = ::Torch::NN::Linear.new(n_x, nbr_classes, bias: false)
          ::Torch::NN::Init.xavier_normal!(final_linear_module.weight)
          final_batch_norm_module = ::Torch::NN::BatchNorm1d.new(nbr_classes, track_running_stats: false)
          final_batch_norm_module.register_buffer("running_mean", ::Torch.zeros(nbr_classes, dtype: :double))
          final_batch_norm_module.register_buffer("running_var", ::Torch.ones(nbr_classes, dtype: :double))
          @layers.concat(
            [
              add_module("l#{layers.size}_linear", final_linear_module),
              add_module("l#{layers.size}_batch_norm1d", final_batch_norm_module)
            ]
          )
        end

        def forward(x)
          a = x
          @layers.each do |layer|
            n_x = a.shape[1]
            a = layer.call(a)
            debug { "Forward propagate #{n_x} => #{layer.class.name.split('::').last} => #{a.shape[1]}." }
            debug { "=> Output data: #{data_to_str(a)}" }
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
        super(rows, cols, channels, nbr_classes, torch_net: DenseStackNet.new(n_x: rows * cols * channels, layers:, nbr_classes:))
      end

    end

  end

end
