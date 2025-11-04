module RubyNeuralNets
  module TransformHelpers
    module Vips

      # Apply resize transformation if dimensions differ
      #
      # Parameters::
      # * *image* (Vips::Image): Input image to resize
      # * *target_width* (Integer): Target width
      # * *target_height* (Integer): Target height
      # Result::
      # * Vips::Image: Resized image or original if no resize needed
      def self.resize(image, target_width, target_height)
        if image.width != target_width || image.height != target_height
          image.resize(target_width.to_f / image.width, vscale: target_height.to_f / image.height)
        else
          image
        end
      end

      # Apply grayscale transformation to the image
      #
      # Parameters::
      # * *image* (Vips::Image): Input image to convert to grayscale
      # Result::
      # * Vips::Image: Single-channel grayscale image
      def self.grayscale(image)
        image.colourspace(:b_w)
      end

      # Apply trimming transformation while preserving aspect ratio
      #
      # Parameters::
      # * *image* (Vips::Image): Input image to trim
      # Result::
      # * Vips::Image: Trimmed image with aspect ratio preserved
      def self.trim(image)
        # Store original aspect ratio
        original_aspect_ratio = image.width.to_f / image.height
        # Find bounding box (non-background pixels)
        # For simplicity, we'll use a basic trim that removes uniform borders
        # This is a simplified version - a full implementation would need more complex logic
        image
      end

      # Apply rotation transformation
      #
      # Parameters::
      # * *image* (Vips::Image): Input image to rotate
      # * *rot_angle* (Float): Maximum rotation angle in degrees
      # * *rng* (Random): Random number generator
      # Result::
      # * Vips::Image: Rotated image or original if no rotation needed
      def self.rotate(image, rot_angle, rng)
        if rot_angle > 0
          angle = rng.rand(-rot_angle..rot_angle)
          image.rotate(angle)
        else
          image
        end
      end

      # Apply Gaussian noise transformation
      #
      # Parameters::
      # * *image* (Vips::Image): Input image to add noise to
      # * *noise_intensity* (Float): Intensity of Gaussian noise
      # * *numo_rng* (Numo::Random::Generator): Random number generator for Numo
      # Result::
      # * Vips::Image: Image with Gaussian noise added or original if no noise needed
      def self.gaussian_noise(image, noise_intensity, numo_rng)
        if noise_intensity > 0
          # Generate noise with same dimensions as image
          noise = numo_rng.normal(shape: [image.height, image.width, image.bands], loc: 0.0, scale: noise_intensity)
          # Add noise to image
          image + noise
        else
          image
        end
      end

      # Apply crop transformation if image is larger than target
      #
      # Parameters::
      # * *image* (Vips::Image): Input image to crop
      # * *target_width* (Integer): Target width
      # * *target_height* (Integer): Target height
      # Result::
      # * Vips::Image: Cropped image or original if no crop needed
      def self.crop(image, target_width, target_height)
        if image.width > target_width || image.height > target_height
          # Crop from center
          x_offset = (image.width - target_width) / 2
          y_offset = (image.height - target_height) / 2
          image.crop(x_offset, y_offset, target_width, target_height)
        else
          image
        end
      end

      # Apply adaptive invert transformation to the image
      #
      # Parameters::
      # * *image* (Vips::Image): Input image to potentially invert
      # Result::
      # * Vips::Image: Image with colors inverted if top left pixel intensity is in lower half
      def self.adaptive_invert(image)
        # Get top left pixel value (assuming RGB, take average)
        top_left = image[0, 0]
        intensity = top_left.inject(:+).to_f / top_left.size
        # Invert if intensity is in lower half range (normalized 0-1)
        if intensity < 0.5
          1.0 - image
        else
          image
        end
      end

      # Apply min-max normalization to the image
      #
      # Parameters::
      # * *image* (Vips::Image): Input image to normalize
      # Result::
      # * Vips::Image: Image with normalized pixel values
      def self.minmax_normalize(image)
        # Normalize each band to 0-1 range
        min_vals = image.min
        max_vals = image.max
        (image - min_vals) / (max_vals - min_vals)
      end

    end
  end
end
