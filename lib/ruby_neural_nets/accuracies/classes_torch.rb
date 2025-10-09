require 'ruby_neural_nets/accuracy'

module RubyNeuralNets
  
  module Accuracies
    
    # Compute accuracy from classified Torch inputs
    class ClassesTorch < Accuracy
      
      # Measure accuracy between a predicted output and the real one
      #
      # Parameters::
      # * *output_pred* (Object): Predicted output
      # * *output_real* (Object): Expected real output
      # * *minibatch_size* (Integer): The minibatch size
      # Result::
      # * Float: Corresponding accuracy
      def measure(output_pred, output_real, minibatch_size)
        final_prediction(output_pred).eq(output_real).sum.item.to_f / minibatch_size
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
        nbr_classes = output_pred.shape[1]
        true_one_hot = ::Torch.zeros(minibatch_size, nbr_classes).scatter(1, output_real.reshape(minibatch_size, 1), 1)
        pred_one_hot = ::Torch.zeros(minibatch_size, nbr_classes).scatter(1, final_prediction(output_pred).reshape(minibatch_size, 1), 1)
        true_one_hot.t.matmul(pred_one_hot).numo
      end

      private

      # Compute the final prediction from the model output
      #
      # Parameters::
      # * *model_output* (Object): The model output
      # Result::
      # * Torch::Tensor: The final prediction
      def final_prediction(model_output)
        _, predicted = ::Torch.max(model_output.data, 1)
        predicted
      end

    end

  end

end
