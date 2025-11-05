require 'ruby_neural_nets/data_loader'
require 'ruby_neural_nets/datasets/labeled_files'
require 'ruby_neural_nets/datasets/labeled_data_partitioner'
require 'ruby_neural_nets/datasets/file_to_vips'
require 'ruby_neural_nets/datasets/vips_remove_alpha'
require 'ruby_neural_nets/datasets/vips_grayscale'
require 'ruby_neural_nets/datasets/vips_adaptive_invert'
require 'ruby_neural_nets/datasets/vips_minmax_normalize'
require 'ruby_neural_nets/datasets/vips_trim'
require 'ruby_neural_nets/datasets/vips_normalize'
require 'ruby_neural_nets/datasets/vips_resize'
require 'ruby_neural_nets/datasets/vips_rotate'
require 'ruby_neural_nets/datasets/vips_crop'
require 'ruby_neural_nets/datasets/vips_noise'
require 'ruby_neural_nets/datasets/clone'
require 'ruby_neural_nets/datasets/one_hot_encoder'
require 'ruby_neural_nets/datasets/cache_memory'
require 'ruby_neural_nets/datasets/epoch_shuffler'
require 'ruby_neural_nets/datasets/minibatch'

module RubyNeuralNets

  module DataLoaders

    class VipsNumo < DataLoader

      # Constructor
      #
      # Parameters::
      # * *dataset* (String): The dataset name
      # * *max_minibatch_size* (Integer): Max size each minibatch should have
      # * *dataset_seed* (Integer): Random number generator seed for dataset shuffling and data order
      # * *nbr_clones* (Integer): Number of times each element should be cloned
      # * *rot_angle* (Float): Maximum rotation angle in degrees for random image transformations
      # * *grayscale* (bool): Convert images to grayscale, reducing channels from 3 to 1
      # * *adaptive_invert* (bool): Apply adaptive color inversion based on top left pixel intensity
      # * *trim* (bool): Trim images to remove borders and restore original aspect ratio
      # * *resize* (Array): Resize dimensions [width, height] for image transformations
      # * *noise_intensity* (Float): Intensity of Gaussian noise for image transformations
      # * *minmax_normalize* (bool): Scale image data to always be within the range 0 to 1
      def initialize(dataset:, max_minibatch_size:, dataset_seed:, nbr_clones:, rot_angle:, grayscale:, adaptive_invert:, trim:, resize:, noise_intensity:, minmax_normalize:)
        @nbr_clones = nbr_clones
        @rot_angle = rot_angle
        @grayscale = grayscale
        @adaptive_invert = adaptive_invert
        @trim = trim
        @resize = resize
        @noise_intensity = noise_intensity
        @minmax_normalize = minmax_normalize
        super(dataset:, max_minibatch_size:, dataset_seed:)
      end

      # Instantiate a partitioned dataset.
      #
      # Parameters::
      # * *name* (String): Dataset name containing real data
      # * *rng* (Random): The random number generator to be used
      # * *numo_rng* (Numo::Random::Generator): The Numo random number generator to be used
      # Result::
      # * LabeledDataPartitioner: The partitioned dataset.
      def new_partitioned_dataset(name:, rng:, numo_rng:)
        Datasets::LabeledDataPartitioner.new(
          Datasets::LabeledFiles.new(name:),
          rng:
        )
      end

      # Return a preprocessing dataset for this data loader.
      #
      # Parameters::
      # * *dataset* (Dataset): The partitioned dataset
      # Result::
      # * Dataset: The dataset with preprocessing applied
      def new_preprocessing_dataset(dataset)
        base_dataset = Datasets::VipsRemoveAlpha.new(
          Datasets::FileToVips.new(
            Datasets::OneHotEncoder.new(dataset)
          )
        )
        resized_dataset = Datasets::VipsResize.new(@trim ? Datasets::VipsTrim.new(base_dataset) : base_dataset, resize: @resize)

        # Apply preprocessing layers
        processed_dataset = resized_dataset
        processed_dataset = Datasets::VipsGrayscale.new(processed_dataset) if @grayscale
        processed_dataset = Datasets::VipsMinmaxNormalize.new(processed_dataset) if @minmax_normalize
        processed_dataset = Datasets::VipsAdaptiveInvert.new(processed_dataset) if @adaptive_invert

        Datasets::CacheMemory.new(processed_dataset)
      end

      # Return an augmentation dataset for this data loader.
      # This is only used for the training dataset.
      #
      # Parameters::
      # * *preprocessed_dataset* (Dataset): The preprocessed dataset
      # * *rng* (Random): The random number generator to be used
      # * *numo_rng* (Numo::Random::Generator): The Numo random number generator to be used
      # Result::
      # * Dataset: The dataset with augmentation applied
      def new_augmentation_dataset(preprocessed_dataset, rng:, numo_rng:)
        Datasets::VipsNoise.new(
          Datasets::VipsRotate.new(
            Datasets::Clone.new(
              preprocessed_dataset,
              nbr_clones: @nbr_clones
            ),
            rot_angle: @rot_angle,
            rng:
          ),
          numo_rng:,
          noise_intensity: @noise_intensity
        )
      end

      # Return a batching dataset for this data loader.
      #
      # Parameters::
      # * *augmented_dataset* (Dataset): The augmented dataset
      # * *rng* (Random): The random number generator to be used
      # * *numo_rng* (Numo::Random::Generator): The Numo random number generator to be used
      # * *max_minibatch_size* (Integer): The required minibatch size
      # Result::
      # * Dataset: The dataset with batching applied
      def new_batching_dataset(augmented_dataset, rng:, numo_rng:, max_minibatch_size:)
        Datasets::Minibatch.new(
          Datasets::EpochShuffler.new(
            Datasets::VipsNormalize.new(augmented_dataset),
            rng:
          ),
          max_minibatch_size:
        )
      end

      # Display a sample image from a dataset
      #
      # Parameters::
      # * *dataset_type* (Symbol): Dataset type from which the sample should be taken
      # * *label* (String): Label from which the sample should be taken
      def display_sample(dataset_type, label)
        # Get the minibatch dataset for the specified type
        elements_dataset = @partition_datasets[dataset_type].elements_dataset

        # Get the one-hot vector for the requested label
        target_one_hot = elements_dataset.one_hot_labels[label]

        # Find a sample with the matching one-hot vector
        found_x, _found_y = elements_dataset.find { |_select_x, select_y| select_y == target_one_hot }
        # Get image stats for dimensions
        stats = image_stats
        rows = stats[:rows]
        cols = stats[:cols]

        # Create ImageMagick image from pixel data using appropriate format
        image = Magick::Image.new(cols, rows)
        image.import_pixels(
          0,
          0,
          cols,
          rows,
          case image_stats[:channels]
          when 1
            'I'
          when 3
            'RGB'
          else
            raise "Unsupported number of channels for display: #{image_stats[:channels]}"
          end,
          # Convert normalized pixel data back to ImageMagick format
          # The data is in 0-1 range, need to convert back to 0-65535 range
          found_x.flatten.map { |pixel_value| (pixel_value * 65535).round }
        )

        log "Display sample image of label #{label} from #{dataset_type} dataset"
        Helpers.display_image(image)
      end

    end

  end

end
