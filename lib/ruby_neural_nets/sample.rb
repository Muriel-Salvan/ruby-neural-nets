module RubyNeuralNets

  # Class representing a sample returned by a dataset with lazy evaluation
  class Sample

    # Constructor
    #
    # Parameters::
    # * *input_proc* (Proc): Proc that evaluates the real input value of this sample
    # * *target_proc* (Proc): Proc that evaluates the real target value of this sample
    def initialize(input_proc, target_proc)
      @input_proc = input_proc
      @target_proc = target_proc
    end

    # Get the input data of the sample (lazy evaluation)
    #
    # Result::
    # * Object: The sample input data
    def input
      @input_proc.call
    end

    # Get the target data of the sample (lazy evaluation)
    #
    # Result::
    # * Object: The sample target data
    def target
      @target_proc.call
    end

  end

end
