module RubyNeuralNets

  module Helpers

    class NumericalInstabilityError < StandardError
    end

    # Initialize the project
    #
    # Parameters::
    # * *seed* (Integer): Random seed to be used [default: 0]
    # * *instability_checks* (Symbol): Behavior when numerical instability is detected [default: :warning]
    #   Possible values are:
    #   * *off*: Don't perform instability checks.
    #   * *warning*: Display found instability in stdout.
    #   * *prompt*: Invoke a byebug prompt at the time instability is found.
    #   * *exception*: Raise an exception.
    def self.init(seed: 0, instability_checks: :warning)
      @instability_checks = instability_checks
      Random.srand(seed)
      Numo::NArray.srand(seed)
    end

    # Compute sigmoid of an array
    #
    # Parameters::
    # * *narray* (Numo::DFloat): The array on which we apply sigmoid
    # Result::
    # * Numo::DFloat: Resulting sigmoid
    def self.sigmoid(narray)
      1 / (1 + Numo::DFloat::Math.exp(-narray))
    end

    # Perform safe softmax of an array along its first axis.
    # See https://medium.com/@weidagang/essential-math-for-machine-learning-safe-softmax-1ddcc21c744f
    #
    # Parameters::
    # * *narray* (Numo::DFloat): The array on which we apply softmax
    # Result::
    # * Numo::DFloat: Resulting safe softmax
    def self.softmax(narray)
      safe_array = narray - narray.max(axis: 0, keepdims: true)
      exp_array = Numo::DFloat::Math.exp(safe_array)
      sums = exp_array.sum(axis: 0, keepdims: true)
      exp_array / sums
    end

    # Check numerical instability of a given tensor.
    #
    # Parameters::
    # * *tensor* (Numo::DFloat): Tensor for which we check numerical instability
    # * *types* (Symbol or Array<Symbol>): Types of instability to check for [default: :not_finite]
    #   Possible values are:
    #   * *not_finite*: Check if any value is Inf or NaN.
    #   * *zero*: Check if any value is 0.
    #   * *one*: Check if any value is 1.
    def self.check_instability(tensor, types: :not_finite)
      if @instability_checks != :off
        errors = []
        [types].flatten.each do |type|
          case type
          when :not_finite
            errors << 'Some values are either Inf or NaN' unless tensor.isfinite.all?
          when :zero
            errors << 'Some values are 0' if tensor.eq(0).any?
          when :one
            errors << 'Some values are 1' if tensor.eq(1).any?
          else
            raise "Unknown instability type: #{type}"
          end
        end
        unless errors.empty?
          error_msg = "Numerical instability detected: #{errors.join(', ')}"
          case @instability_checks
          when :warning
            puts "!!! #{error_msg} !!!"
          when :byebug
            puts "!!! #{error_msg} !!!"
            require 'byebug'
            byebug
            nil # Add this line just for byebug to not display stupid messages because we invoke it at the method's end
          when :exception
            raise NumericalInstabilityError.new(error_msg)
          else
            raise "Unknown instability checks type: #{@instability_checks}"
          end
        end
      end
    end
  
  end

end
