require 'google/protobuf'
require 'ruby_neural_nets/onnx/onnx_pb'
require 'ruby_neural_nets/logger'
require "ruby_neural_nets/models/torch.#{RUBY_PLATFORM}"

module RubyNeuralNets

  module Models

    class OnnxTorch < Torch

      class OnnxNet < ::Torch::NN::Module
        include Logger

        # Constructor
        #
        # Parameters::
        # * *onnx_file* (String): ONNX file path
        def initialize(onnx_file)
          super()
          @model = Onnx::ModelProto.decode(File.binread(onnx_file))
          
          # Register ONNX initializers as Torch parameters so they can be accessed by named_parameters
          @model.graph.initializer.each do |initializer|
            torch_tensor = tensor_proto_to_torch(initializer)
            # Ensure the tensor requires gradients for backpropagation
            torch_tensor.requires_grad!(true)
            # Register as a parameter - Torch register_parameter expects (name, tensor)
            register_parameter(initializer.name, torch_tensor)
          end
        end

        # Forward pass through the ONNX model
        #
        # Parameters::
        # * *x* (Torch::Tensor): Input tensor
        # Result::
        # * Torch::Tensor: Output tensor
        def forward(x)
          # Execute the ONNX model graph
          graph = @model.graph
          # Create a mapping of tensor names to their values
          tensor_values = {}
          # Set the input tensor and ensure it requires gradients for backpropagation
          x.requires_grad!(true) unless x.requires_grad
          tensor_values[graph.input.first.name] = x
          # Execute each node in the graph
          graph.node.each do |node|
            execute_node(node, tensor_values)
          end
          # Return the output tensor
          tensor_values[graph.output.first.name]
        end

        private

        PROTOBUF_TO_TORCH_MAPPING = {
          Onnx::TensorProto::DataType::FLOAT => {
            data_method: :float_data,
            unpack: 'f*'
          },
          Onnx::TensorProto::DataType::DOUBLE => {
            data_method: :double_data,
            unpack: 'd*'
          },
          Onnx::TensorProto::DataType::INT32 => {
            data_method: :int32_data,
            unpack: 'l*'
          },
          Onnx::TensorProto::DataType::INT64 => {
            data_method: :int64_data,
            unpack: 'q*'
          }
        }

        # Convert an ONNX TensorProto to a Torch tensor
        #
        # Parameters::
        # * *tensor_proto* (Onnx::TensorProto): The ONNX tensor proto
        # Result::
        # * Torch::Tensor: The corresponding Torch tensor
        def tensor_proto_to_torch(tensor_proto)
          raise "Unsupported tensor data type: #{tensor_proto.data_type}" unless PROTOBUF_TO_TORCH_MAPPING.key?(tensor_proto.data_type)

          # Use raw_data if the typed data is empty
          protobuf_mapping = PROTOBUF_TO_TORCH_MAPPING[tensor_proto.data_type]
          protobuf_data = tensor_proto.send(protobuf_mapping[:data_method])
          ::Torch.tensor(protobuf_data.empty? ? tensor_proto.raw_data.unpack(protobuf_mapping[:unpack]) : protobuf_data.to_a, dtype: :double).reshape(tensor_proto.dims.to_a)
        end

        NODE_TYPE_TO_METHOD = {
          # General Matrix Multiplication (Dense layer)
          'Gemm' => :execute_gemm,
          # Convolution operation
          'Conv' => :execute_conv,
          # Max Pooling operation
          'MaxPool' => :execute_max_pool,
          # Global Average Pooling operation
          'GlobalAveragePool' => :execute_global_average_pool,
          # Softmax activation
          'Softmax' => :execute_softmax,
          # ReLU activation
          'Relu' => :execute_relu,
          # Element-wise addition
          'Add' => :execute_add,
          # Matrix multiplication
          'MatMul' => :execute_matmul,
          # Reshape operation
          'Reshape' => :execute_reshape,
          # Transpose operation
          'Transpose' => :execute_transpose,
          # Concat operation
          'Concat' => :execute_concat,
          # Shape operation
          'Shape' => :execute_shape,
          # Constant operation
          'Constant' => :execute_constant,
          # Gather operation
          'Gather' => :execute_gather
        }

        # Execute a single ONNX node
        #
        # Parameters::
        # * *node* (Onnx::NodeProto): The ONNX node to execute
        # * *tensor_values* (Hash<String, Torch::Tensor>): Mapping of tensor names to values
        def execute_node(node, tensor_values)
          input_tensors = node.input.map do |input_name|
            tensor_value = tensor_values[input_name]
            if tensor_value.nil?
              # Try to get it from the registered parameters (initializers)
              param = named_parameters[input_name]
              raise "Tensor '#{input_name}' not found in tensor_values or registered parameters" if param.nil?

              param
            else
              tensor_value
            end
          end
          
          raise "Unsupported ONNX operation: #{node.op_type}" unless NODE_TYPE_TO_METHOD.key?(node.op_type)
          
          # Store the output tensor
          tensor_values[node.output.first] = self.send(NODE_TYPE_TO_METHOD[node.op_type], input_tensors, node.attribute)
          
          debug { "Executed #{node.op_type}: #{node.input} -> #{node.output}" }
        end

        # Execute Gemm operation (Dense layer)
        #
        # Parameters::
        # * *input_tensors* (Array<Torch::Tensor>): [input, weight, bias?]
        # * *attributes* (Array<Onnx::AttributeProto>): Node attributes
        # Result::
        # * Torch::Tensor: The output tensor
        def execute_gemm(input_tensors, attributes)
          input = input_tensors[0]
          bias = input_tensors[2]
          
          # Extract attributes
          alpha = find_attribute(attributes, 'alpha')&.f || 1.0
          beta = find_attribute(attributes, 'beta')&.f || 1.0
          trans_a = find_attribute(attributes, 'transA')&.i || 0
          
          # Transpose if needed
          input = input.t if trans_a == 1
          
          # Compute: alpha * (input @ weight) + beta * bias
          output = alpha * (input.matmul(input_tensors[1].t))
          output = output + beta * bias if bias
          
          output
        end

        # Execute Conv operation
        #
        # Parameters::
        # * *input_tensors* (Array<Torch::Tensor>): [input, weight, bias?]
        # * *attributes* (Array<Onnx::AttributeProto>): Node attributes
        # Result::
        # * Torch::Tensor: The output tensor
        def execute_conv(input_tensors, attributes)
          # Extract padding attributes
          pad_top, pad_left, pad_bottom, pad_right = find_attribute(attributes, 'pads')&.ints&.to_a || [0, 0, 0, 0]
          
          # Perform convolution using Torch's functional interface
          # ONNX Conv weight format: [out_channels, in_channels, kernel_height, kernel_width]
          # Torch conv2d expects the same format
          ::Torch::NN::Functional.conv2d(
            # Apply padding if needed
            (pad_top + pad_left + pad_bottom + pad_right) > 0 ? ::Torch::NN::Functional.pad(input_tensors[0], [pad_left, pad_right, pad_top, pad_bottom]) : input_tensors[0],
            input_tensors[1],
            input_tensors[2],
            stride: find_attribute(attributes, 'strides')&.ints&.to_a || [1, 1],
            padding: 0, # We already applied padding manually
            dilation: find_attribute(attributes, 'dilations')&.ints&.to_a || [1, 1],
            groups: find_attribute(attributes, 'group')&.i || 1
          )
        end

        # Execute MaxPool operation
        #
        # Parameters::
        # * *input_tensors* (Array<Torch::Tensor>): [input]
        # * *attributes* (Array<Onnx::AttributeProto>): Node attributes
        # Result::
        # * Torch::Tensor: The output tensor
        def execute_max_pool(input_tensors, attributes)
          # Extract attributes
          kernel_shape = find_attribute(attributes, 'kernel_shape')&.ints&.to_a || [2, 2]
          strides = find_attribute(attributes, 'strides')&.ints&.to_a || [1, 1]
          pads = find_attribute(attributes, 'pads')&.ints&.to_a || [0, 0, 0, 0]
          dilations = find_attribute(attributes, 'dilations')&.ints&.to_a || [1, 1]
          ceil_mode = find_attribute(attributes, 'ceil_mode')&.i || 0
          
          # Extract padding values
          pad_top, pad_left, pad_bottom, pad_right = pads
          
          # Apply padding if needed
          input = (pad_top + pad_left + pad_bottom + pad_right) > 0 ? 
                   ::Torch::NN::Functional.pad(input_tensors.first, [pad_left, pad_right, pad_top, pad_bottom]) : 
                   input_tensors.first
          
          # Perform max pooling using Torch's functional interface
          ::Torch::NN::Functional.max_pool2d(
            input,
            kernel_size: kernel_shape,
            stride: strides,
            padding: 0, # We already applied padding manually
            dilation: dilations,
            ceil_mode: ceil_mode == 1
          )
        end

        # Execute Global Average Pooling operation
        #
        # Parameters::
        # * *input_tensors* (Array<Torch::Tensor>): [input]
        # * *attributes* (Array<Onnx::AttributeProto>): Node attributes
        # Result::
        # * Torch::Tensor: The output tensor
        def execute_global_average_pool(input_tensors, attributes)
          # Global average pooling computes the average over the entire spatial dimensions
          # For 4D input [batch, channels, height, width], it reduces to [batch, channels]
          ::Torch::NN::Functional.adaptive_avg_pool2d(input_tensors.first, [1, 1]).squeeze(dim: 2).squeeze(dim: 2)
        end

        # Execute Softmax operation
        #
        # Parameters::
        # * *input_tensors* (Array<Torch::Tensor>): [input]
        # * *attributes* (Array<Onnx::AttributeProto>): Node attributes
        # Result::
        # * Torch::Tensor: The output tensor
        def execute_softmax(input_tensors, attributes)
          ::Torch::NN::Functional.softmax(input_tensors.first, dim: (find_attribute(attributes, 'axis')&.i || -1))
        end

        # Execute ReLU operation
        #
        # Parameters::
        # * *input_tensors* (Array<Torch::Tensor>): [input]
        # * *attributes* (Array<Onnx::AttributeProto>): Node attributes
        # Result::
        # * Torch::Tensor: The output tensor
        def execute_relu(input_tensors, attributes)
          ::Torch::NN::Functional.relu(input_tensors.first)
        end

        # Execute Add operation
        #
        # Parameters::
        # * *input_tensors* (Array<Torch::Tensor>): [input1, input2]
        # * *attributes* (Array<Onnx::AttributeProto>): Node attributes
        # Result::
        # * Torch::Tensor: The output tensor
        def execute_add(input_tensors, attributes)
          input_tensors[0] + input_tensors[1]
        end

        # Execute MatMul operation
        #
        # Parameters::
        # * *input_tensors* (Array<Torch::Tensor>): [input1, input2]
        # * *attributes* (Array<Onnx::AttributeProto>): Node attributes
        # Result::
        # * Torch::Tensor: The output tensor
        def execute_matmul(input_tensors, attributes)
          input_tensors[0].matmul(input_tensors[1])
        end

        # Execute Reshape operation
        #
        # Parameters::
        # * *input_tensors* (Array<Torch::Tensor>): [input, shape]
        # * *attributes* (Array<Onnx::AttributeProto>): Node attributes
        # Result::
        # * Torch::Tensor: The output tensor
        def execute_reshape(input_tensors, attributes)
          input_tensors.first.reshape(input_tensors[1].to_a)
        end

        # Execute Transpose operation
        #
        # Parameters::
        # * *input_tensors* (Array<Torch::Tensor>): [input]
        # * *attributes* (Array<Onnx::AttributeProto>): Node attributes
        # Result::
        # * Torch::Tensor: The output tensor
        def execute_transpose(input_tensors, attributes)
          input = input_tensors.first
          perm = find_attribute(attributes, 'perm')&.ints
          
          if perm
            input.permute(perm.to_a)
          else
            input.t
          end
        end

        # Execute Concat operation
        #
        # Parameters::
        # * *input_tensors* (Array<Torch::Tensor>): List of input tensors to concatenate
        # * *attributes* (Array<Onnx::AttributeProto>): Node attributes
        # Result::
        # * Torch::Tensor: The concatenated output tensor
        def execute_concat(input_tensors, attributes)
          # Extract the axis attribute (default to 0 if not specified)
          axis = find_attribute(attributes, 'axis')&.i || 0
          
          # Concatenate all input tensors along the specified axis
          ::Torch.cat(input_tensors, dim: axis)
        end

        # Execute Shape operation
        #
        # Parameters::
        # * *input_tensors* (Array<Torch::Tensor>): [input]
        # * *attributes* (Array<Onnx::AttributeProto>): Node attributes
        # Result::
        # * Torch::Tensor: The shape of the input tensor as a 1D tensor
        def execute_shape(input_tensors, attributes)
          # Extract the start and end attributes (optional)
          attr_start = find_attribute(attributes, 'start')&.i || 0
          attr_end = find_attribute(attributes, 'end')&.i

          # Get the shape of the input tensor
          shape = input_tensors.first.shape

          # Convert to Ruby array and apply start/end slicing if specified
          shape_array = shape.to_a
          if attr_end
            shape_array = shape_array[attr_start...attr_end]
          else
            shape_array = shape_array[attr_start..-1]
          end

          # Return as a 1D tensor with int64 dtype (standard for ONNX Shape)
          ::Torch.tensor(shape_array, dtype: :int64)
        end

        # Execute Constant operation
        #
        # Parameters::
        # * *input_tensors* (Array<Torch::Tensor>): Empty array for Constant operation
        # * *attributes* (Array<Onnx::AttributeProto>): Node attributes
        # Result::
        # * Torch::Tensor: The constant tensor
        def execute_constant(input_tensors, attributes)
          # Extract the value attribute which contains the constant tensor
          value_attribute = find_attribute(attributes, 'value')
          raise "Constant operation requires a 'value' attribute" if value_attribute.nil?
          
          # Convert the TensorProto to a Torch tensor
          tensor_proto_to_torch(value_attribute.t)
        end

        # Execute Gather operation
        #
        # Parameters::
        # * *input_tensors* (Array<Torch::Tensor>): [data, indices]
        # * *attributes* (Array<Onnx::AttributeProto>): Node attributes
        # Result::
        # * Torch::Tensor: The gathered tensor
        def execute_gather(input_tensors, attributes)
          data = input_tensors[0]
          indices = input_tensors[1]
          axis = find_attribute(attributes, 'axis')&.i || 0
          
          # Ensure axis is within valid range
          rank = data.dim
          axis = axis < 0 ? axis + rank : axis
          raise "Axis #{axis} is out of bounds for tensor with rank #{rank}" if axis < 0 || axis >= rank
          
          # Convert indices to int64 dtype if needed (Torch's gather requires integer indices)
          indices = indices.to(:int64) unless indices.dtype == :int32 || indices.dtype == :int64
          
          # Use Torch's gather function
          data.gather(axis, indices)
        end

        # Find an attribute by name
        #
        # Parameters::
        # * *attributes* (Array<Onnx::AttributeProto>): List of attributes
        # * *name* (String): Attribute name to find
        # Result::
        # * Onnx::AttributeProto: The found attribute or nil
        def find_attribute(attributes, name)
          attributes.find { |attr| attr.name == name }
        end

      end

      # Define a model for processing images and outputting a given number of classes
      #
      # Parameters::
      # * *rows* (Integer): Number of rows per image
      # * *cols* (Integer): Number of columns per image
      # * *channels* (Integer): Number of channels per image
      # * *nbr_classes* (Integer): Number of classes to identify
      # * *onnx_model* (String): Name of the ONNX model to load
      # * *models_path* (String): Path to model files
      def initialize(rows, cols, channels, nbr_classes, onnx_model:, models_path:)
        super(rows, cols, channels, nbr_classes, torch_net: OnnxNet.new("#{models_path}/#{onnx_model}.onnx"))
      end

    end

  end

end
