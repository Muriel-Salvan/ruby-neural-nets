module RubyNeuralNets

  class Accuracy

    # Measure accuracy between a predicted output and the real one
    #
    # Parameters::
    # * *output_pred* (Numo::DFloat): Predicted output
    # * *output_real* (Numo::DFloat): Expected real output
    # Result::
    # * Float: Corresponding accuracy
    def measure(output_pred, output_real)
      (output_real - output_pred.gt(0.5)).abs.sum(axis: 0).eq(0).count.to_f / output_real.shape[1]
    end

    # Get the confusion matrix between a predicted output and the real one
    #
    # Parameters::
    # * *output_pred* (Numo::DFloat): Predicted output
    # * *output_real* (Numo::DFloat): Expected real output
    # Result::
    # * Numo::DFloat: Corresponding confusion matrix
    def confusion_matrix(output_pred, output_real)
      output_real.dot(Numo::DFloat[output_pred.gt(0.5).transpose][0,nil,nil])
    end

  end

end
