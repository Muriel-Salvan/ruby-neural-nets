module RubyNeuralNets

  # Base class for minibatch objects
  class Minibatch

    # Get the input data (X) of the minibatch
    #
    # Result::
    # * Object: The minibatch input data
    def x
      raise NotImplementedError
    end

    # Get the labels (Y) of the minibatch
    #
    # Result::
    # * Object: The minibatch labels
    def y
      raise NotImplementedError
    end

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
    #     * *x* (Object): The element X being iterated on
    #     * *y* (Object): The element Y being iterated on
    def each_element(&block)
      raise NotImplementedError
    end

  end

end
