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
          input_name = graph.input.first.name
          x.requires_grad!(true) unless x.requires_grad
          tensor_values[input_name] = x
          
          # Execute each node in the graph
          graph.node.each do |node|
            execute_node(node, tensor_values)
          end
          
          # Return the output tensor
          output_name = graph.output.first.name
          tensor_values[output_name]
        end

        private

        # Convert an ONNX TensorProto to a Torch tensor
        #
        # Parameters::
        # * *tensor_proto* (Onnx::TensorProto): The ONNX tensor proto
        # Result::
        # * Torch::Tensor: The corresponding Torch tensor
        def tensor_proto_to_torch(tensor_proto)
          # Convert Google::Protobuf::RepeatedField to Ruby array
          dims = tensor_proto.dims.to_a
          
          case tensor_proto.data_type
          when Onnx::TensorProto::DataType::FLOAT
            if tensor_proto.float_data.empty?
              # Use raw_data if float_data is empty
              raw_data = tensor_proto.raw_data.unpack('f*')
              ::Torch.tensor(raw_data, dtype: :double).reshape(dims)
            else
              ::Torch.tensor(tensor_proto.float_data.to_a, dtype: :double).reshape(dims)
            end
          when Onnx::TensorProto::DataType::DOUBLE
            if tensor_proto.double_data.empty?
              raw_data = tensor_proto.raw_data.unpack('d*')
              ::Torch.tensor(raw_data, dtype: :double).reshape(dims)
            else
              ::Torch.tensor(tensor_proto.double_data.to_a, dtype: :double).reshape(dims)
            end
          when Onnx::TensorProto::DataType::INT32
            if tensor_proto.int32_data.empty?
              raw_data = tensor_proto.raw_data.unpack('l*')
              ::Torch.tensor(raw_data, dtype: :double).reshape(dims)
            else
              ::Torch.tensor(tensor_proto.int32_data.to_a, dtype: :double).reshape(dims)
            end
          when Onnx::TensorProto::DataType::INT64
            if tensor_proto.int64_data.empty?
              raw_data = tensor_proto.raw_data.unpack('q*')
              ::Torch.tensor(raw_data, dtype: :double).reshape(dims)
            else
              ::Torch.tensor(tensor_proto.int64_data.to_a, dtype: :double).reshape(dims)
            end
          else
            raise "Unsupported tensor data type: #{tensor_proto.data_type}"
          end
        end

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
              if param
                param
              else
                raise "Tensor '#{input_name}' not found in tensor_values or registered parameters"
              end
            else
              tensor_value
            end
          end
          
          case node.op_type
          when 'Gemm'
            # General Matrix Multiplication (Dense layer)
            # Input format: [input, weight, bias] where bias is optional
            output = execute_gemm(input_tensors, node.attribute)
          when 'Softmax'
            # Softmax activation
            output = execute_softmax(input_tensors, node.attribute)
          when 'Relu'
            # ReLU activation
            output = execute_relu(input_tensors)
          when 'Add'
            # Element-wise addition
            output = execute_add(input_tensors)
          when 'MatMul'
            # Matrix multiplication
            output = execute_matmul(input_tensors)
          when 'Reshape'
            # Reshape operation
            output = execute_reshape(input_tensors, node.attribute)
          when 'Transpose'
            # Transpose operation
            output = execute_transpose(input_tensors, node.attribute)
          else
            raise "Unsupported ONNX operation: #{node.op_type}"
          end
          
          # Store the output tensor
          tensor_values[node.output.first] = output
          
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
          input, weight = input_tensors[0], input_tensors[1]
          bias = input_tensors[2]
          
          # Extract attributes
          alpha = find_attribute(attributes, 'alpha')&.f || 1.0
          beta = find_attribute(attributes, 'beta')&.f || 1.0
          trans_a = find_attribute(attributes, 'transA')&.i || 0
          trans_b = find_attribute(attributes, 'transB')&.i || 0
          
          # Transpose if needed
          weight = weight.t if trans_b == 1
          input = input.t if trans_a == 1
          
          # For ONNX Dense layer: weight is stored as [nbr_classes, features] but needs to be [features, nbr_classes]
          # for matrix multiplication with input [batch_size, features]
          unless trans_b == 1
            weight = weight.t
          end
          
          # Compute: alpha * (input @ weight) + beta * bias
          output = alpha * (input.matmul(weight))
          output = output + beta * bias if bias
          
          output
        end

        # Execute Softmax operation
        #
        # Parameters::
        # * *input_tensors* (Array<Torch::Tensor>): [input]
        # * *attributes* (Array<Onnx::AttributeProto>): Node attributes
        # Result::
        # * Torch::Tensor: The output tensor
        def execute_softmax(input_tensors, attributes)
          input = input_tensors.first
          axis = find_attribute(attributes, 'axis')&.i || -1
          
          ::Torch::NN::Functional.softmax(input, dim: axis)
        end

        # Execute ReLU operation
        #
        # Parameters::
        # * *input_tensors* (Array<Torch::Tensor>): [input]
        # Result::
        # * Torch::Tensor: The output tensor
        def execute_relu(input_tensors)
          input = input_tensors.first
          ::Torch::NN::Functional.relu(input)
        end

        # Execute Add operation
        #
        # Parameters::
        # * *input_tensors* (Array<Torch::Tensor>): [input1, input2]
        # Result::
        # * Torch::Tensor: The output tensor
        def execute_add(input_tensors)
          input1, input2 = input_tensors
          input1 + input2
        end

        # Execute MatMul operation
        #
        # Parameters::
        # * *input_tensors* (Array<Torch::Tensor>): [input1, input2]
        # Result::
        # * Torch::Tensor: The output tensor
        def execute_matmul(input_tensors)
          input1, input2 = input_tensors
          input1.matmul(input2)
        end

        # Execute Reshape operation
        #
        # Parameters::
        # * *input_tensors* (Array<Torch::Tensor>): [input, shape]
        # * *attributes* (Array<Onnx::AttributeProto>): Node attributes
        # Result::
        # * Torch::Tensor: The output tensor
        def execute_reshape(input_tensors, attributes)
          input = input_tensors.first
          shape_tensor = input_tensors[1]
          
          # Convert shape tensor to Ruby array
          shape = shape_tensor.to_a
          input.reshape(shape)
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
