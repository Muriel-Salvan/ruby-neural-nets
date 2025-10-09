require 'numo/narray'

module RubyNeuralNets

  # Base class for any accuracy measurement
  class Accuracy

    # Measure accuracy between a predicted output and the real one
    #
    # Parameters::
    # * *output_pred* (Object): Predicted output
    # * *output_real* (Object): Expected real output
    # Result::
    # * Float: Corresponding accuracy
    def measure(output_pred, output_real)
      raise 'Not implemented'
    end

    # Get the confusion matrix between a predicted output and the real one
    #
    # Parameters::
    # * *output_pred* (Object): Predicted output
    # * *output_real* (Object): Expected real output
    # * *minibatch_size* (Integer): The minibatch size
    # Result::
    # * Numo::DFloat: Corresponding confusion matrix
    def confusion_matrix(output_pred, output_real, minibatch_size)
      raise 'Not implemented'
    end

  end

end
