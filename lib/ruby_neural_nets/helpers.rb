module RubyNeuralNets

  module Helpers

    # Compute sigmoid of an array
    #
    # Parameters::
    # * *narray* (Numo::DFloat): The array on which we apply sigmoid
    # Result::
    # * Numo::DFloat: Resulting sigmoid
    def self.sigmoid(narray)
      1 / (1 + Numo::DFloat::Math.exp(-narray))
    end

    # Perform safe softmax of an array along its first axis.
    # See https://medium.com/@weidagang/essential-math-for-machine-learning-safe-softmax-1ddcc21c744f
    #
    # Parameters::
    # * *narray* (Numo::DFloat): The array on which we apply softmax
    # Result::
    # * Numo::DFloat: Resulting safe softmax
    def self.softmax(narray)
      safe_array = narray - narray.max(axis: 0, keepdims: true)
      exp_array = Numo::DFloat::Math.exp(safe_array)
      sums = exp_array.sum(axis: 0, keepdims: true)
      exp_array / sums
    end

  end

end
