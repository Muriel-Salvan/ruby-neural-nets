module RubyNeuralNets
  module TorchVision
    module Transforms
      class Cache < ::Torch::NN::Module

        attr_reader :transforms

        def initialize(transforms)
          @transforms = transforms
          @cache = {}
        end

        def forward(input)
          cache_key = input
          unless @cache.key?(cache_key)
            # Apply all transforms in sequence
            result = input
            @transforms.each do |transform|
              result = transform.forward(result)
            end
            # Cache the final result
            @cache[cache_key] = result
          end
          @cache[cache_key]
        end

      end
    end
  end
end
