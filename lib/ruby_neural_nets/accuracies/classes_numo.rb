require 'ruby_neural_nets/accuracy'

module RubyNeuralNets
  
  module Accuracies
    
    # Compute accuracy from classified Numo::DFloat inputs
    class ClassesNumo < Accuracy
      
      # Measure accuracy between a predicted output and the real one
      #
      # Parameters::
      # * *output_pred* (Object): Predicted output
      # * *minibatch* (RubyNeuralNets::Minibatch): Minibatch containing expected real output and size
      # Result::
      # * Float: Corresponding accuracy
      def measure(output_pred, minibatch)
        (minibatch.target - final_prediction(output_pred)).abs.sum(axis: 0).eq(0).count.to_f / minibatch.size
      end

      # Get the confusion matrix between a predicted output and the real one
      #
      # Parameters::
      # * *output_pred* (Object): Predicted output
      # * *minibatch* (RubyNeuralNets::Minibatch): Minibatch containing expected real output and size
      # Result::
      # * Numo::DFloat: Corresponding confusion matrix
      def confusion_matrix(output_pred, minibatch)
        minibatch.target.dot(Numo::DFloat[final_prediction(output_pred).transpose][0, nil, nil])
      end

      private

      # Compute the final prediction from the model output
      #
      # Parameters::
      # * *model_output* (Numo::DFloat): The model output
      # Result::
      # * Numo::DFloat: The final prediction
      def final_prediction(model_output)
        out = Numo::DFloat.zeros(*model_output.shape)
        out[model_output.max_index(axis: 0)] = 1
        out
      end

    end

  end

end
