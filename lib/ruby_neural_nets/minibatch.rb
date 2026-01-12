require 'ruby_neural_nets/sample'

module RubyNeuralNets

  # Base class for minibatch objects
  class Minibatch < Sample

    # Get the size of the minibatch
    #
    # Result::
    # * Integer: The minibatch size
    def size
      raise NotImplementedError
    end

    # Iterate over individual elements of the minibatch
    #
    # Parameters::
    # * *block* (Proc): Block to call for each element
    #   * Parameters::
    #     * *sample* (Sample): The sample being iterated on
    def each_element
      raise NotImplementedError
    end

  end

end
