module RubyNeuralNets

  # Base class representing a dataset, serving data
  class Dataset

    include Enumerable

    # Return the dataset size
    #
    # Result::
    # * Integer: Number of elements in this dataset
    def size
      raise 'Not implemented'
    end

    # Prepare the dataset to be served for a given epoch.
    # This is called before starting an epoch.
    # This can be used to generate some data before hand, or shuffle in a particular way.
    def prepare_for_epoch
      # Does nothing by default
    end

    # Access an element of the dataset
    #
    # Parameters::
    # * *index* (Integer): Index of the dataset element to access
    # Result::
      # * Object: The element X of the dataset
      # * Object: The element Y of the dataset
    def [](index)
      raise 'Not implemented'
    end

    # Return the underlying dataset's label for a given output label of this dataset layer
    #
    # Parameters::
    # * *y* (Object): Label, as returned by the [] method
    # Result::
    # * Object: Corresponding underlying label
    def underlying_label(y)
      # By default there is no translation of the label
      y
    end

    # Iterate over all elements
    #
    # Parameters::
    # * Code: Code called for each element iterated on
    def each
      return to_enum(:each) unless block_given?
      
      size.times.each { |index| yield self[index] }
    end

  end

end
