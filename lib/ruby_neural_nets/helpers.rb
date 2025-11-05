require 'rmagick'
require 'tmpdir'
require 'numo/narray'

module RubyNeuralNets

  module Helpers

    class ApplicationError < StandardError
    end

    # Initialize the project
    #
    # Parameters::
    # * *seed* (Integer): Random seed to be used [default: 0]
    # * *instability_checks* (Symbol): Behavior when numerical instability is detected [default: :warning]
    #   Possible values are:
    #   * *off*: Don't perform instability checks.
    #   * Any other value from the handle_error method's behavior parameter.
    def self.init(model_seed: 0, instability_checks: :warning)
      @instability_checks = instability_checks
      # Set model-related random seeds
      Random.srand(model_seed)
      Numo::NArray.srand(model_seed)
      ::Torch.manual_seed(model_seed) if const_defined?('::Torch')
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

    # Perform tanh of an array
    #
    # Parameters::
    # * *narray* (Numo::DFloat): The array on which we apply tanh
    # Result::
    # * Numo::DFloat: Resulting tanh
    def self.tanh(narray)
      exp_array = Numo::DFloat::Math.exp(narray)
      exp_neg_array = Numo::DFloat::Math.exp(-narray)
      (exp_array - exp_neg_array) / (exp_array + exp_neg_array)
    end

    # Compute the euclidian norm of a tensor
    #
    # Parameters::
    # * *tensor* (Numo::DFloat): The tensor to consider
    # Result::
    # * Float: The tensor's euclidian norm
    def self.norm_2(tensor)
      Math.sqrt(tensor.dot(tensor.transpose))
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
          handle_error(error_msg, @instability_checks)
        end
      end
    end
  
    # Handle an application error case, with a given behavior
    #
    # Parameters::
    # * *error_msg* (String): The error message to handle
    # * *behavior* (Symbol): The behavior to use:
    #   * *warning*: Display found instability in stdout.
    #   * *byebug*: Invoke a byebug prompt at the time instability is found.
    #   * *exception*: Raise an exception.
    def self.handle_error(error_msg, behavior)
      case behavior
      when :warning
        puts "!!! #{error_msg} !!!"
      when :byebug
        puts "!!! #{error_msg} !!!"
        require 'byebug'
        byebug
        nil # Add this line just for byebug to not display stupid messages because we invoke it at the method's end
      when :exception
        raise ApplicationError.new(error_msg)
      else
        raise "Unknown application error behavior: #{behavior}"
      end
    end

    # Display an image
    #
    # Parameters::
    # * *image* (Magick::Image or Vips::Image): The image to be displayed
    def self.display_image(image)
      require 'tmpdir'
      Dir.mktmpdir do |temp_dir|
        file_name = "#{temp_dir}/display.png"
        case image
        when Magick::Image
          image.write(file_name)
        when Vips::Image
          image.write_to_file(file_name)
        else
          raise "Unsupported image format: #{image.class}"
        end
        system "#{RUBY_PLATFORM == 'x86_64-linux' ? 'xdg-open' : 'start'} #{file_name}"
        puts 'Press enter to continue...'
        $stdin.gets
      end
    end

    # Return the pixels map used by export/import pixels methods.
    # In case of grayscale images, return only the intensity.
    #
    # Parameters::
    # * *image* (RMagick::Image): The RMagick image
    # Result::
    # * String: Corresponding pixels map
    def self.image_pixels_map(image)
      case image.colorspace
      when Magick::GRAYColorspace
        'I'
      when Magick::RGBColorspace, Magick::SRGBColorspace
        'RGB'
      else
        raise "Unknown colorspace: #{image.colorspace}"
      end
    end

  end

end
