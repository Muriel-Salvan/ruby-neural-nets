require 'fileutils'
require 'tmpdir'
require 'rmagick'

module RubyNeuralNetsTest
  module Helpers

    # Helper method to generate ONNX file content for a OneLayer model with Gemm as a string.
    #
    # This method creates an ONNX model protobuf for a single layer neural network
    # with the specified dimensions using Gemm (Dense layer), and returns the encoded content as a string.
    #
    # Parameters::
    # * *rows* (Integer): Number of rows per image
    # * *cols* (Integer): Number of columns per image  
    # * *channels* (Integer): Number of channels per image
    # * *nbr_classes* (Integer): Number of classes to identify
    # Result::
    # * String: The ONNX model content as a binary string
    def onnx_one_layer_gemm(rows, cols, channels, nbr_classes)
      require 'google/protobuf'
      require 'ruby_neural_nets/onnx/onnx_pb'
      require 'ruby_neural_nets/models/one_layer'

      # Instantiate a model with the specified dimensions
      model = RubyNeuralNets::Models::OneLayer.new(rows, cols, channels, nbr_classes)
      params_w = model.parameters(name: 'W').first
      params_b = model.parameters(name: 'B').first

      # Helper to create a TensorProto
      def float_tensor(name:, dims:, values:)
        Onnx::TensorProto.new(
          name: name,
          dims: dims,
          data_type: Onnx::TensorProto::DataType::FLOAT,
          float_data: values
        )
      end

      # 1️⃣ Create model
      protobuf_model = Onnx::ModelProto.new(
        ir_version: 8,                 # Onnx IR version
        opset_import: [Onnx::OperatorSetIdProto.new(domain: "", version: 13)],
        producer_name: 'Ruby example',
        graph: Onnx::GraphProto.new(
          name: "OneLayerGemmModel"
        )
      )

      protobuf_graph = protobuf_model.graph

      # 2️⃣ Define inputs
      protobuf_graph.input << Onnx::ValueInfoProto.new(
        name: 'input',
        type: Onnx::TypeProto.new(
          tensor_type: Onnx::TypeProto::Tensor.new(
            elem_type: Onnx::TensorProto::DataType::FLOAT,
            shape: Onnx::TensorShapeProto.new(
              dim: [
                Onnx::TensorShapeProto::Dimension.new(dim_value: 1), # batch
                Onnx::TensorShapeProto::Dimension.new(dim_value: params_w.shape[1])  # feature size
              ]
            )
          )
        )
      )

      # 3️⃣ Create weights and bias for Dense (Gemm)
      protobuf_graph.initializer << float_tensor(name: 'dense_weight', dims: params_w.shape, values: params_w.values.to_a.flatten)
      protobuf_graph.initializer << float_tensor(name: 'dense_bias', dims: [params_b.shape[0]], values: params_b.values.to_a.flatten)

      # 4️⃣ Create Gemm node (Dense)
      protobuf_graph.node << Onnx::NodeProto.new(
        op_type: 'Gemm',
        input: %w[input dense_weight dense_bias],
        output: %w[dense_out],
        name: 'Dense1'
      )

      # 5️⃣ Create Softmax node
      protobuf_graph.node << Onnx::NodeProto.new(
        op_type: 'Softmax',
        input: %w[dense_out],
        output: %w[output],
        name: 'Softmax1',
        attribute: [Onnx::AttributeProto.new(name: 'axis', i: 1, type: :INT)]
      )

      # 6️⃣ Define output
      protobuf_graph.output << Onnx::ValueInfoProto.new(
        name: 'output',
        type: Onnx::TypeProto.new(
          tensor_type: Onnx::TypeProto::Tensor.new(
            elem_type: Onnx::TensorProto::DataType::FLOAT,
            shape: Onnx::TensorShapeProto.new(
              dim: [
                Onnx::TensorShapeProto::Dimension.new(dim_value: 1),
                Onnx::TensorShapeProto::Dimension.new(dim_value: params_w.shape[0])
              ]
            )
          )
        )
      )

      # 7️⃣ Return model as encoded string
      Onnx::ModelProto.encode(protobuf_model)
    end

    # Helper method to generate ONNX file content for a OneLayer model with Conv as a string.
    #
    # This method creates an ONNX model protobuf for a simple convolutional neural network
    # with the specified dimensions using Conv operation, and returns the encoded content as a string.
    #
    # Parameters::
    # * *rows* (Integer): Number of rows per image
    # * *cols* (Integer): Number of columns per image  
    # * *channels* (Integer): Number of channels per image
    # * *nbr_classes* (Integer): Number of classes to identify
    # Result::
    # * String: The ONNX model content as a binary string
    def onnx_one_layer_conv(rows, cols, channels, nbr_classes)
      require 'google/protobuf'
      require 'ruby_neural_nets/onnx/onnx_pb'

      # Helper to create a TensorProto
      def float_tensor(name:, dims:, values:)
        Onnx::TensorProto.new(
          name: name,
          dims: dims,
          data_type: Onnx::TensorProto::DataType::FLOAT,
          float_data: values
        )
      end

      # Define convolution parameters
      out_channels = nbr_classes
      kernel_size = 3
      in_channels = channels

      # Create random weights and bias for simplicity
      conv_weight_size = out_channels * in_channels * kernel_size * kernel_size
      conv_weight_values = Array.new(conv_weight_size) { rand(-0.1..0.1) }
      conv_bias_values = Array.new(out_channels) { rand(-0.1..0.1) }

      # 1️⃣ Create model
      protobuf_model = Onnx::ModelProto.new(
        ir_version: 8,                 # Onnx IR version
        opset_import: [Onnx::OperatorSetIdProto.new(domain: "", version: 13)],
        producer_name: 'Ruby example',
        graph: Onnx::GraphProto.new(
          name: "OneLayerConvModel"
        )
      )

      protobuf_graph = protobuf_model.graph

      # 2️⃣ Define inputs - Input format: [batch, channels, height, width]
      protobuf_graph.input << Onnx::ValueInfoProto.new(
        name: 'input',
        type: Onnx::TypeProto.new(
          tensor_type: Onnx::TypeProto::Tensor.new(
            elem_type: Onnx::TensorProto::DataType::FLOAT,
            shape: Onnx::TensorShapeProto.new(
              dim: [
                Onnx::TensorShapeProto::Dimension.new(dim_value: 1), # batch
                Onnx::TensorShapeProto::Dimension.new(dim_value: in_channels), # channels
                Onnx::TensorShapeProto::Dimension.new(dim_value: rows), # height
                Onnx::TensorShapeProto::Dimension.new(dim_value: cols)  # width
              ]
            )
          )
        )
      )

      # 3️⃣ Create weights and bias for Conv
      protobuf_graph.initializer << float_tensor(
        name: 'conv_weight', 
        dims: [out_channels, in_channels, kernel_size, kernel_size], 
        values: conv_weight_values
      )
      protobuf_graph.initializer << float_tensor(
        name: 'conv_bias', 
        dims: [out_channels], 
        values: conv_bias_values
      )

      # 4️⃣ Create Conv node
      protobuf_graph.node << Onnx::NodeProto.new(
        op_type: 'Conv',
        input: %w[input conv_weight conv_bias],
        output: %w[conv_out],
        name: 'Conv1',
        attribute: [
          Onnx::AttributeProto.new(name: 'kernel_shape', ints: [kernel_size, kernel_size], type: :INTS),
          Onnx::AttributeProto.new(name: 'strides', ints: [1, 1], type: :INTS),
          Onnx::AttributeProto.new(name: 'pads', ints: [1, 1, 1, 1], type: :INTS) # padding to preserve size
        ]
      )

      # 5️⃣ Create Global Average Pooling to reduce spatial dimensions
      protobuf_graph.node << Onnx::NodeProto.new(
        op_type: 'GlobalAveragePool',
        input: %w[conv_out],
        output: %w[pool_out],
        name: 'GlobalPool1'
      )

      # 6️⃣ Create Softmax node
      protobuf_graph.node << Onnx::NodeProto.new(
        op_type: 'Softmax',
        input: %w[pool_out],
        output: %w[output],
        name: 'Softmax1',
        attribute: [Onnx::AttributeProto.new(name: 'axis', i: 1, type: :INT)]
      )

      # 7️⃣ Define output
      protobuf_graph.output << Onnx::ValueInfoProto.new(
        name: 'output',
        type: Onnx::TypeProto.new(
          tensor_type: Onnx::TypeProto::Tensor.new(
            elem_type: Onnx::TensorProto::DataType::FLOAT,
            shape: Onnx::TensorShapeProto.new(
              dim: [
                Onnx::TensorShapeProto::Dimension.new(dim_value: 1),
                Onnx::TensorShapeProto::Dimension.new(dim_value: out_channels)
              ]
            )
          )
        )
      )

      # 8️⃣ Return model as encoded string
      Onnx::ModelProto.encode(protobuf_model)
    end

    # Helper method to generate PNG file content as a string.
    #
    # This method creates a PNG image with the specified dimensions and fills it with pixel values.
    # The pixels parameter can be an array of pixel values or a hash with color key for uniform color.
    # The bit depth of the pixels is always 16 bits (between 0 and 65535).
    #
    # Parameters::
    # * *width* (Integer): The width of the image in pixels
    # * *height* (Integer): The height of the image in pixels
    # * *pixels* (Array<Integer> or Hash): Array of pixel values or Hash to describe the image's features.
    #   When used as a Hash, the following properties are possible:
    #   * *color* (Array<Integer>): A color description (as many values as desired channels) that fills the whole image.
    # Result::
    # * String: The PNG file content as a binary string
    def png(width, height, pixels)
      # Create a new image
      image = Magick::Image.new(width, height) do |img|
        img.format = 'PNG'
      end

      pixel_map, pixel_values =
        if pixels.is_a?(Hash)
          # Uniform color
          color = pixels[:color]
          case color.size
          when 1
            # Grayscale
            ['I', Array.new(width * height, color[0])]
          when 3
            # RGB
            ['RGB', color * (width * height)]
          else
            raise ArgumentError, "Unsupported color array length: #{color.size}"
          end
        else
          # Pixel values array or uniform color
          case pixels.size
          when width * height
            # Grayscale pixels
            ['I', pixels]
          when width * height * 3
            # RGB pixels
            ['RGB', pixels]
          else
            raise ArgumentError, "Unsupported pixels array length: #{pixels.size} for an image of size #{width}x#{height}"
          end
        end
      image.import_pixels(0, 0, width, height, pixel_map, pixel_values)

      image.to_blob
    end

    # Setup a temporary directory with some content
    #
    # Parameters::
    # * *content* (Hash<String, String>): A hash mapping relative file paths to their contents (as strings, e.g., PNG data blobs)
    # * Proc: Code block to execute with the test directory setup
    #   * Parameters::
    #     * *dir* (String): The temporary directory created
    def with_test_dir(content)
      Dir.mktmpdir do |dir|
        content.each do |path, content|
          full_path = "#{dir}/#{path}"
          FileUtils.mkdir_p(File.dirname(full_path))
          File.binwrite(full_path, content)
        end
        yield dir
      end
    end

    # Compute distance between 2 arrays, element-wise
    #
    # Parameters::
    # * *array_1* (Array or Numo::DFloat): First array
    # * *array_2* (Array or Numo::DFloat): Second array
    # Result::
    # * Array: The distance
    def array_distance(array_1, array_2)
      array_1.to_a.zip(array_2.to_a).map { |(e_1, e_2)| (e_1 - e_2).abs }
    end

    # Expect a given array to have all its elements within the range of another array.
    # Handle recursively nested arrays.
    #
    # Parameters::
    # * *array_1* (Array or Numo::DFloat): First array
    # * *array_2* (Array or Numo::DFloat): Second array
    # * *threshold* (Float): Threshold range [default: 0.01]
    def expect_array_within(array_1, array_2, threshold = 0.01)
      if array_1.is_a?(Array) && array_1.first.is_a?(Array)
        array_1.zip(array_2).each do |(sub_array_1, sub_array_2)|
          expect_array_within(sub_array_1, sub_array_2, threshold = threshold)
        end
      else
        max_distance = array_distance(array_1, array_2).max
        expect(max_distance).to be_within(threshold).of(0), "expected distance #{max_distance} to be smaller than #{threshold} between #{array_1.to_a} and expected #{array_2.to_a}"
      end
    end

    # Expect a given array not to have all its elements within the range of anothr array
    #
    # Parameters::
    # * *array_1* (Array or Numo::DFloat): First array
    # * *array_2* (Array or Numo::DFloat): Second array
    # * *threshold* (Float): Threshold range [default: 0.01]
    def expect_array_not_within(array_1, array_2, threshold = 0.01)
      expect(array_distance(array_1, array_2).max).not_to be_within(threshold).of(0)
    end

  end

end
