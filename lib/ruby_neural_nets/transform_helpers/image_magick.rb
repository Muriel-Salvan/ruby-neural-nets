module RubyNeuralNets
  module TransformHelpers
    module ImageMagick

      # Apply resize transformation if dimensions differ
      #
      # Parameters::
      # * *image* (Magick::Image): Input image to resize
      # * *target_width* (Integer): Target width
      # * *target_height* (Integer): Target height
      # Result::
      # * Magick::Image: Resized image or original if no resize needed
      def self.resize(image, target_width, target_height)
        if image.columns != target_width || image.rows != target_height
          image.resize(target_width, target_height)
        else
          image
        end
      end

      # Apply grayscale transformation to the image
      #
      # Parameters::
      # * *image* (Magick::Image): Input image to convert to grayscale
      # Result::
      # * Magick::Image: Single-channel grayscale image
      def self.grayscale(image)
        gray_image = image.copy
        gray_image.colorspace = Magick::GRAYColorspace
        gray_image
      end

      # Apply trimming transformation while preserving aspect ratio
      #
      # Parameters::
      # * *image* (Magick::Image): Input image to trim
      # Result::
      # * Magick::Image: Trimmed image with aspect ratio preserved
      def self.trim(image)
        # Store original aspect ratio
        original_aspect_ratio = image.columns.to_f / image.rows
        # Get the bounding box that we want to trim
        bounding_box = image.bounding_box

        # Calculate new dimensions to restore aspect ratio
        trimmed_width = bounding_box.width
        trimmed_height = bounding_box.height
        # Compute the desired height we want for this trimmed width
        desired_trimmed_height = (trimmed_width.to_f / original_aspect_ratio).round
        if desired_trimmed_height > trimmed_height
          # Add rows
          bounding_box.y = [bounding_box.y - (desired_trimmed_height - trimmed_height) / 2, 0].max
          bounding_box.height = desired_trimmed_height
        else
          # Add columns
          desired_trimmed_width = (trimmed_height.to_f * original_aspect_ratio).round
          if desired_trimmed_width > trimmed_width
            bounding_box.x = [bounding_box.x - (desired_trimmed_width - trimmed_width) / 2, 0].max
            bounding_box.width = desired_trimmed_width
          end
        end

        if bounding_box.x != 0 || bounding_box.y != 0 || bounding_box.width != image.columns || bounding_box.height != image.rows
          image.crop(bounding_box.x, bounding_box.y, bounding_box.width, bounding_box.height)
        else
          image
        end
      end

      # Apply rotation transformation
      #
      # Parameters::
      # * *image* (Magick::Image): Input image to rotate
      # * *rot_angle* (Float): Maximum rotation angle in degrees
      # * *rng* (Random): Random number generator
      # Result::
      # * Magick::Image: Rotated image or original if no rotation needed
      def self.rotate(image, rot_angle, rng)
        if rot_angle > 0
          image.virtual_pixel_method = Magick::EdgeVirtualPixelMethod
          image.distort(Magick::ScaleRotateTranslateDistortion, [rng.rand(-rot_angle..rot_angle)])
        else
          image
        end
      end

      # Apply Gaussian noise transformation
      #
      # Parameters::
      # * *image* (Magick::Image): Input image to add noise to
      # * *noise_intensity* (Float): Intensity of Gaussian noise
      # * *numo_rng* (Numo::Random::Generator): Random number generator for Numo
      # Result::
      # * Magick::Image: Image with Gaussian noise added or original if no noise needed
      def self.gaussian_noise(image, noise_intensity, numo_rng)
        if noise_intensity > 0
          pixels_map = Helpers.image_pixels_map(image)
          original_pixels = Numo::DFloat[image.export_pixels(0, 0, image.columns, image.rows, pixels_map)]
          new_image = Magick::Image.new(image.columns, image.rows)
          max_value = max_channel_value(image)
          new_image.import_pixels(
            0,
            0,
            image.columns,
            image.rows,
            pixels_map,
            (original_pixels + numo_rng.normal(shape: original_pixels.shape, loc: 0.0, scale: noise_intensity * max_value)).clip(0, max_value).flatten.to_a
          )
          new_image
        else
          image
        end
      end

      # Apply crop transformation if image is larger than target
      #
      # Parameters::
      # * *image* (Magick::Image): Input image to crop
      # * *target_width* (Integer): Target width
      # * *target_height* (Integer): Target height
      # Result::
      # * Magick::Image: Cropped image or original if no crop needed
      def self.crop(image, target_width, target_height)
        if image.columns > target_width || image.rows > target_height
          image.crop(0, 0, target_width, target_height)
        else
          image
        end
      end

      # Apply adaptive invert transformation to the image
      #
      # Parameters::
      # * *image* (Magick::Image): Input image to potentially invert
      # Result::
      # * Magick::Image: Image with colors inverted if top left pixel intensity is in lower half
      def self.adaptive_invert(image)
        # Invert if intensity is in lower half range
        if image.pixel_color(0, 0).intensity < 32768
          inverted_image = image.copy
          inverted_image.alpha(Magick::DeactivateAlphaChannel)
          inverted_image.negate
        else
          image
        end
      end

      # Apply min-max normalization to the image
      #
      # Parameters::
      # * *image* (Magick::Image): Input image to normalize
      # Result::
      # * Magick::Image: Image with normalized pixel values
      def self.minmax_normalize(image)
        image.normalize_channel(Magick::AllChannels)
      end

      # Remove alpha channel from ImageMagick image
      #
      # Parameters::
      # * *image* (Magick::Image): Input image that may have alpha channel
      # Result::
      # * Magick::Image: Image with alpha channel removed if present
      def self.remove_alpha(image)
        if image.alpha?
          plain_image = image.copy
          plain_image.alpha(Magick::DeactivateAlphaChannel)
          plain_image
        else
          image
        end
      end

      # Get the max value of a channel value of an image
      #
      # Parameters::
      # * *image* (Magick::Image): The image
      # Result::
      # * Integer: The max value a pixel channel can have
      def self.max_channel_value(image)
        2 ** image.quantum_depth - 1
      end

    end
  end
end
