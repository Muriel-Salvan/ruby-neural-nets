require 'ruby_neural_nets/models/one_layer'
require 'ruby_neural_nets/optimizers/constant'

describe RubyNeuralNets::Models::OneLayer do

  describe '#stats' do

    it 'returns the right model statistics' do
      # Setup test data: create a model with specific dimensions (2x2x1 images on 3 classes)
      expect(described_class.new(2, 2, 1, 3).stats).to eq({
        parameters: {
          'W' => { size: 12 }, # 3 classes * 4 features (2*2*1)
          'B' => { size: 3 }   # 3 classes * 1 bias per class
        }
      })
    end

  end

  describe '#forward_propagate' do

    it 'produces expected output values for given input using initialized model parameters' do
      # Setup test data: create a model with specific dimensions (2x2x1 images on 3 classes)
      model = described_class.new(2, 2, 1, 3)

      # Input: simple flattened images, shape [n_x, minibatch_size] = [4, 2] for 2 samples
      x = Numo::DFloat[[1.0, 0.0],
                       [0.0, 1.0],
                       [0.0, 0.0],
                       [0.0, 0.0]]

      # Call forward_propagate
      output = model.forward_propagate(x)

      # Assert output matches expected (with small tolerance for floating point)
      # Compute expected output manually: z_1 = w_values.dot(x) + b_values, then softmax
      z_1 = model.parameters(name: 'W').first.values.dot(x) + model.parameters(name: 'B').first.values
      expect(output.shape).to eq([3, 2])
      expect_array_within(
        output.to_a,
        (0...2).map do |sample_idx|
          exp_values = z_1[true, sample_idx].map { |val| Math.exp(val) }
          sum_exp = exp_values.sum
          exp_values.map { |val| val / sum_exp }.to_a
        end.transpose
      )

      # Additional assertions: each column should sum to 1 (softmax property)
      (0...2).each do |j|
        expect((0...3).inject(0.0) { |s, i| s + output[i, j] }).to be_within(1e-6).of(1.0)
      end
    end

  end

  describe '#gradient_descent' do

    it 'updates parameters correctly during gradient descent' do
      # Setup test data: create a model with specific dimensions (2x2x1 images on 3 classes)
      model = described_class.new(2, 2, 1, 3)

      # Input: flattened images, shape [n_x, minibatch_size] = [4, 2]
      x = Numo::DFloat[[1.0, 0.0],
                       [0.0, 1.0],
                       [0.0, 0.0],
                       [0.0, 0.0]]

      # Labels: one-hot encoded, shape [nbr_classes, minibatch_size] = [3, 2]
      y = Numo::DFloat[[1.0, 0.0],
                       [0.0, 1.0],
                       [0.0, 0.0]]

      # Create minibatch
      minibatch = RubyNeuralNets::Minibatches::Numo.new(x, y)

      # Setup optimizer and link to parameters (use Constant for predictable updates)
      optimizer = RubyNeuralNets::Optimizers::Constant.new(learning_rate: 0.1, weight_decay: 0.0)
      optimizer.teach_parameters(model.parameters)

      # Store original parameter values
      original_w = model.parameters(name: 'W').first.values.dup
      original_b = model.parameters(name: 'B').first.values.dup

      # Forward and backward propagate
      optimizer.start_minibatch(0, minibatch.size)
      model.initialize_back_propagation_cache
      a = model.forward_propagate(x, train: true)
      # da is not used in the method, but we pass it as per signature
      model.gradient_descent(Numo::DFloat.zeros(3, 2), a, minibatch, 1.0)

      # Assert the parameter updates match expected gradient descent computation
      m = minibatch.size.to_f
      dz_1 = a - y
      expected_dw_1 = dz_1.dot(x.transpose) / m
      expected_db_1 = dz_1.sum(axis: 1, keepdims: true) / m
      learning_rate = 0.1
      expect_array_within(model.parameters(name: 'W').first.values.flatten, (original_w - learning_rate * expected_dw_1).flatten, 1e-6)
      expect_array_within(model.parameters(name: 'B').first.values.flatten, (original_b - learning_rate * expected_db_1).flatten, 1e-6)
    end

  end

end
