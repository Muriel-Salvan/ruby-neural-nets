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
          image.resize(target_width.to_f / image.width, vscale: target_height.to_f / image.height, kernel: :mitchell)
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
        # Find bounding box of non-background pixels
        left, top, width, height = image.find_trim(background: image.getpoint(0, 0)[0..2])

        # Calculate new dimensions to restore aspect ratio
        trimmed_width = width
        trimmed_height = height
        # Compute the desired height we want for this trimmed width
        desired_trimmed_height = (trimmed_width.to_f / original_aspect_ratio).round
        if desired_trimmed_height > trimmed_height
          # Add rows (expand vertically)
          top = [top - (desired_trimmed_height - trimmed_height) / 2, 0].max
          height = desired_trimmed_height
        else
          desired_trimmed_width = (trimmed_height.to_f * original_aspect_ratio).round
          if desired_trimmed_width > trimmed_width
            # Add columns (expand horizontally)
            left = [left - (desired_trimmed_width - trimmed_width) / 2, 0].max
            width = desired_trimmed_width
          end
        end

        # Crop to the adjusted bounding box
        if left != 0 || top != 0 || width != image.width || height != image.height
          image.crop(left, top, width, height)
        else
          image
        end
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
          rotated_image = image.rotate(rng.rand(-rot_angle..rot_angle), background: image.getpoint(0, 0))
          (rotated_image.width != image.width || rotated_image.height != image.height) ? rotated_image.crop((rotated_image.width - image.width) / 2, (rotated_image.height - image.height) / 2, image.width, image.height) : rotated_image
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
          (image + ::Vips::Image.bandjoin(image.bands.times.map { |band_idx| ::Vips::Image.new_from_array(numo_rng.normal(shape: [image.height, image.width], loc: 0.0, scale: noise_intensity * 255).to_a) })).clamp(min: 0, max: 255)
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
        top_left = image.getpoint(0, 0)
        # Invert if intensity is in lower half range (normalized 0-1)
        top_left.inject(:+).to_f / top_left.size < 128 ? (-image + 255).scaleimage : image
      end

      # Apply min-max normalization to the image
      #
      # Parameters::
      # * *image* (Vips::Image): Input image to normalize
      # Result::
      # * Vips::Image: Image with each band normalized to 0-255 range
      def self.minmax_normalize(image)
        ::Vips::Image.bandjoin(
          # Normalize each band to 0-255 range and combine using bandjoin
          image.bands.times.map do |i|
            band = image[i]
            min_val = band.min
            max_val = band.max
            # If all pixels are the same, set to 127.5 (midpoint)
            max_val > min_val ? ((band - min_val) * 255) / (max_val - min_val) : band * 0 + 127
          end
        ).scaleimage
      end

      # Remove alpha channel from Vips image
      #
      # Parameters::
      # * *image* (Vips::Image): Input image that may have alpha channel
      # Result::
      # * Vips::Image: Image with alpha channel removed if present (4 bands -> 3 bands)
      def self.remove_alpha(image)
        # Remove alpha channel if present (4 bands -> 3 bands)
        image.bands == 4 ? image.extract_band(0, n: 3) : image
      end

    end
  end
end
