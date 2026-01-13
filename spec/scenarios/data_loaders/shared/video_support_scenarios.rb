RSpec.shared_examples 'video support scenarios' do |options|

  describe 'video support' do

    it 'handles MP4 files correctly with video_slices_sec parameter' do
      with_test_dir(
        'test_dataset/class_0/test_video_0.mp4' => mp4(2, 2, 2.5, { color: [32768] })
      ) do |datasets_path|
        samples = new_data_loader(datasets_path:, video_slices_sec: 1.0, partitions: { training: 1.0 }, resize: [2, 2]).dataset(:training).first.each_element.to_a
        # Should have 3 slices from the 2.5 second video with 1.0 second slices, with 2x2 gray images in RGB (12 values)
        expect_array_within(samples.map { |sample| sample.input.to_a }, [[0.5] * 12] * 3)
        # All of them should belong to the class indexed 0
        expect(samples.map { |sample| options[:label_from].call(sample.target) }).to eq [0] * 3
      end
    end

    it 'shuffles video frames correctly across partitions' do
      with_test_dir(
        # Target index 0
        # 4 images (pixel value doubled = 0)
        'test_dataset/class_0/test_video_0.mp4' => mp4(2, 2, 3.5, { color: [0] }),
        # Target index 1
        # 5 images (pixel value doubled = 1)
        'test_dataset/class_1/test_video_1.mp4' => mp4(2, 2, 4.5, { color: [32768] }),
        # 3 images (pixel value doubled = 2)
        'test_dataset/class_1/test_video_2.mp4' => mp4(2, 2, 2.5, { color: [65535] })
      ) do |datasets_path|
        data_loader = new_data_loader(datasets_path:, video_slices_sec: 1.0, partitions: { training: 0.75, dev: 0.125, test: 0.125 }, resize: [2, 2])
        # For each partition, check expected number of images for a given color (on an Integer scale between 0 and 2), per target index
        {
          #         { target => { pixel_value_doubled => nbr_images } }
          training: { 0 => { 0 => 3 }, 1 => { 1 => 3, 2 => 3 } },
          dev:      { 0 => { 0 => 1 }, 1 => { 1 => 1         } },
          test:     {                  1 => { 1 => 1         } }
        }.each do |partition, images_per_target|
          # Count the number of samples, grouped by class and channel value (round the value multiplied by 2)
          expect(
            data_loader.dataset(partition).first.each_element.map do |sample|
              [
                options[:label_from].call(sample.target),
                (sample.input.to_a.first * 2).round
              ]
            end.
              group_by { |(target, _value)| target }.
              to_h do |target, target_values_list|
                [
                  target,
                  target_values_list.group_by { |(_target, value)| value }.to_h { |value, values_list| [value, values_list.size] }
                ]
              end
          ).to eq(images_per_target)
        end
      end
    end

    it 'handles mixed PNG and MP4 files correctly' do
      with_test_dir(
        # Force RGB in the PNG by not having true 0 values
        'test_dataset/class_0/test_image_0.png' => png(2, 2, { color: [0, 1, 0] }),
        'test_dataset/class_0/test_video_0.mp4' => mp4(2, 2, 1.5, { color: [32768] })
      ) do |datasets_path|
        # Should have 1 image of doubled value 0 and 2 images of doubled value 1
        expect(
          new_data_loader(datasets_path:, video_slices_sec: 1.0, partitions: { training: 1.0 }, resize: [2, 2]).
            dataset(:training).
            first.
            each_element.
            map { |sample| [options[:label_from].call(sample.target), (sample.input.to_a.first * 2).round] }.
            sort
        ).to eq [
          [0, 0],
          [0, 1],
          [0, 1]
        ]
      end
    end

    it 'raises error for unsupported file extensions' do
      with_test_dir(
        'test_dataset/class_0/test_file.txt' => 'This is not an image'
      ) do |datasets_path|
        expect {
          new_data_loader(datasets_path:)
        }.to raise_error(StandardError, /Unsupported file extension: \.txt/)
      end
    end

    it 'handles different video_slices_sec values correctly' do
      with_test_dir(
        'test_dataset/class_0/test_video_0.mp4' => mp4(2, 2, 3.9, { color: [32768] })
      ) do |datasets_path|
        # With 0.5 second slices, should get 8 slices from a nearly 4 second video
        expect(new_data_loader(datasets_path:, video_slices_sec: 0.5, partitions: { training: 1.0 }, resize: [2, 2]).dataset(:training).first.size).to eq(8)
        
        # With 2.0 second slices, should get 2 slices from a nearly 4 second video
        expect(new_data_loader(datasets_path:, video_slices_sec: 2.0, partitions: { training: 1.0 }, resize: [2, 2]).dataset(:training).first.size).to eq(2)
      end
    end

    describe 'image stats with videos' do

      it 'returns correct image stats for video-derived images' do
        with_test_dir(
          'test_dataset/class_0/test_video_0.mp4' => mp4(10, 20, 1.0, { color: [0, 10, 20] })
        ) do |datasets_path|
          expect(new_data_loader(datasets_path:, resize: [10, 20]).image_stats).to eq({
            rows: 20,
            cols: 10,
            channels: 3,
            depth: 8
          })
        end
      end
    end

  end

end
