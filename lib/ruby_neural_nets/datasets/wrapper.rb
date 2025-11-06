require 'ruby_neural_nets/dataset'

module RubyNeuralNets

  module Datasets

    # Dataset wrapping another dataset with default methods.
    # This is meant to be sub-classed.
    class Wrapper < Dataset

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): Dataset to be wrapped
      def initialize(dataset)
        super()
        @dataset = dataset
      end

      # Return the dataset size
      #
      # Result::
      # * Integer: Number of elements in this dataset
      def size
        @dataset.size
      end

      # Prepare the dataset to be served for a given epoch.
      # This is called before starting an epoch.
      # This can be used to generate some data before hand, or shuffle in a particular way.
      def prepare_for_epoch
        @dataset.prepare_for_epoch
      end

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Object: The element X of the dataset
      # * Object: The element Y of the dataset
      def [](index)
        @dataset[index]
      end

      # Return the underlying dataset's label for a given output label of this dataset layer
      #
      # Parameters::
      # * *y* (Object): Label, as returned by the [] method
      # Result::
      # * Object: Corresponding underlying label
      def underlying_label(y)
        # By default we look for the label of the underlying dataset
        @dataset.underlying_label(y)
      end

      # Forward missing method calls to the wrapped dataset.
      #
      # Parameters::
      # * *method_name* (String): Missing method name
      # * *args* (Array): Arguments
      # * *kwargs* (Hash): Kwargs
      # * *code* (Proc): Default yielded code
      # Result::
      # * Object: The propogated result
      def method_missing(method_name, *args, **kwargs, &code)
        @dataset.send(method_name, *args, **kwargs, &code)
      end
      
    end

  end

end
