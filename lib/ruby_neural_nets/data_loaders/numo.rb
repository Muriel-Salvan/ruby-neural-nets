require 'ruby_neural_nets/data_loader'
require 'ruby_neural_nets/datasets/labeled_files'
require 'ruby_neural_nets/datasets/labeled_data_partitioner'
require 'ruby_neural_nets/datasets/images_from_files'
require 'ruby_neural_nets/datasets/image_normalize'
require 'ruby_neural_nets/datasets/image_transform'
require 'ruby_neural_nets/datasets/clone'
require 'ruby_neural_nets/datasets/one_hot_encoder'
require 'ruby_neural_nets/datasets/cache_memory'
require 'ruby_neural_nets/datasets/epoch_shuffler'
require 'ruby_neural_nets/datasets/minibatch'

module RubyNeuralNets
  
  module DataLoaders
        
    class Numo < DataLoader

      # Constructor
      #
      # Parameters::
      # * *dataset* (String): The dataset name
      # * *max_minibatch_size* (Integer): Max size each minibatch should have
      # * *dataset_seed* (Integer): Random number generator seed for dataset shuffling and data order
      # * *nbr_clones* (Integer): Number of times each element should be cloned
      # * *rot_angle* (Float): Maximum rotation angle in degrees for random image transformations
      def initialize(dataset:, max_minibatch_size:, dataset_seed:, nbr_clones:, rot_angle:)
        @nbr_clones = nbr_clones
        @rot_angle = rot_angle
        super(dataset:, max_minibatch_size:, dataset_seed:)
      end

      # Instantiate a partitioned dataset.
      #
      # Parameters::
      # * *name* (String): Dataset name containing real data
      # * *rng* (Random): The random number generator to be used
      # Result::
      # * LabeledDataPartitioner: The partitioned dataset.
      def new_partitioned_dataset(name:, rng:)
        Datasets::LabeledDataPartitioner.new(
          Datasets::LabeledFiles.new(name:),
          rng:
        )
      end

      # Return a minibatch dataset for this data loader, from a dataset that has already been partitioned.
      #
      # Parameters::
      # * *dataset* (Dataset): The partitioned dataset serving data for the minibatches
      # * *rng* (Random): The random number generator to be used
      # * *max_minibatch_size* (Integer): The required minibatch size
      # Result::
      # * Dataset: The dataset that will serve data as minibatches
      def new_minibatch_dataset(dataset:, rng:, max_minibatch_size:)
        Datasets::Minibatch.new(
          Datasets::EpochShuffler.new(
            Datasets::CacheMemory.new(
              Datasets::OneHotEncoder.new(
                Datasets::ImageNormalize.new(
                  Datasets::ImageTransform.new(
                    Datasets::Clone.new(
                      Datasets::ImagesFromFiles.new(dataset),
                      nbr_clones: @nbr_clones
                    ),
                    rng: rng,
                    rot_angle: @rot_angle
                  )
                )
              )
            ),
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

        # Convert normalized pixel data back to ImageMagick format
        # The data is in 0-1 range, need to convert back to 0-255 range
        pixel_data = found_x.flatten.map { |pixel_value| (pixel_value * 255).round }

        # Create ImageMagick image from pixel data
        image = Magick::Image.new(cols, rows)
        image.import_pixels(0, 0, cols, rows, 'RGB', pixel_data.pack('C*'))

        log "Display sample image of label #{label} from #{dataset_type} dataset"
        Helpers.display_image(image)
      end

    end

  end

end
