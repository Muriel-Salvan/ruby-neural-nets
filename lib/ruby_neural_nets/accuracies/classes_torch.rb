require 'ruby_neural_nets/accuracy'

module RubyNeuralNets
  
  module Accuracies
    
    # Compute accuracy from classified Torch inputs
    class ClassesTorch < Accuracy
      
      # Measure accuracy between a predicted output and the real one
      #
      # Parameters::
      # * *output_pred* (Object): Predicted output
      # * *minibatch* (RubyNeuralNets::Minibatch): Minibatch containing expected real output and size
      # Result::
      # * Float: Corresponding accuracy
      def measure(output_pred, minibatch)
        final_prediction(output_pred).eq(minibatch.target).sum.item.to_f / minibatch.size
      end

      # Get the confusion matrix between a predicted output and the real one
      #
      # Parameters::
      # * *output_pred* (Object): Predicted output
      # * *minibatch* (RubyNeuralNets::Minibatch): Minibatch containing expected real output and size
      # Result::
      # * Numo::DFloat: Corresponding confusion matrix
      def confusion_matrix(output_pred, minibatch)
        nbr_classes = output_pred.shape[1]
        true_one_hot = ::Torch.zeros(minibatch.size, nbr_classes).scatter(1, minibatch.target.reshape(minibatch.size, 1), 1)
        pred_one_hot = ::Torch.zeros(minibatch.size, nbr_classes).scatter(1, final_prediction(output_pred).reshape(minibatch.size, 1), 1)
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
