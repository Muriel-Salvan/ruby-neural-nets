module RubyNeuralNets

  # Base class for optimizers
  class Optimizer

    # Set the current epoch being processed
    #
    # Parameters::
    # * *idx_epoch* (Integer): The epoch being processed
    def start_epoch(idx_epoch)
      @idx_epoch = idx_epoch
    end

  end

end
