require "ruby_neural_nets/models/n_layers_torch.#{RUBY_PLATFORM}"

describe RubyNeuralNets::Models::NLayersTorch do

  describe '#stats' do

    it 'returns the right model statistics' do
      # Setup test data: create a model with specific dimensions (2x2x1 images on 3 classes, one hidden layer with 5 units)
      expect(described_class.new(2, 2, 1, 3, layers: [5]).stats).to eq(
        {
          parameters: {
            'l0_linear.weight' => { size: 20 }, # 5 units * 4 features
            'l0_batch_norm1d.weight' => { size: 5 },
            'l0_batch_norm1d.bias' => { size: 5 },
            'l1_linear.weight' => { size: 15 }, # 3 classes * 5 units
            'l1_batch_norm1d.weight' => { size: 3 },
            'l1_batch_norm1d.bias' => { size: 3 }
          }
        }
      )
    end

  end

  describe '#forward_propagate' do

    it 'produces expected output values for given input using initialized model parameters' do
      # Setup test data: create a model with specific dimensions (2x2x1 images on 3 classes, one hidden layer with 5 units)
      model = described_class.new(2, 2, 1, 3, layers: [5])

      # Input: simple flattened images, shape [batch_size, n_x] = [2, 4] for 2 samples
      x = ::Torch.tensor([[1.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0]], dtype: :double)

      # Call forward_propagate
      output = model.forward_propagate(x)

      # Assert output shape
      expect(output.shape).to eq([2, 3])
      # Values should be reasonable (logits, can be any real numbers)
      expect(output.numo.to_a.flatten.all? { |v| v.is_a?(Float) }).to be true

      # Compute expected output manually by propagating through layers
      # Layer 0: Linear (5 units, no bias)
      w0 = model.parameters.find { |p| p.name == 'l0_linear.weight' }.torch_parameter
      a0 = x.mm(w0.t)

      # Layer 1: BatchNorm1d
      gamma0 = model.parameters.find { |p| p.name == 'l0_batch_norm1d.weight' }.torch_parameter
      beta0 = model.parameters.find { |p| p.name == 'l0_batch_norm1d.bias' }.torch_parameter
      mean0 = a0.mean(dim: 0)
      var0 = a0.var(dim: 0, unbiased: false)
      eps = 1e-5
      x_hat0 = (a0 - mean0) / (var0 + eps).sqrt
      a1 = gamma0 * x_hat0 + beta0

      # Layer 2: LeakyReLU
      a1 = ::Torch::NN::LeakyReLU.new(negative_slope: 0.01).call(a1)

      # Layer 3: Linear (3 units, no bias)
      w1 = model.parameters.find { |p| p.name == 'l1_linear.weight' }.torch_parameter
      a2 = a1.mm(w1.t)

      # Layer 4: BatchNorm1d
      gamma1 = model.parameters.find { |p| p.name == 'l1_batch_norm1d.weight' }.torch_parameter
      beta1 = model.parameters.find { |p| p.name == 'l1_batch_norm1d.bias' }.torch_parameter
      mean1 = a2.mean(dim: 0)
      var1 = a2.var(dim: 0, unbiased: false)
      x_hat1 = (a2 - mean1) / (var1 + eps).sqrt
      expected = gamma1 * x_hat1 + beta1

      expect_array_within(output.numo.flatten.to_a, expected.numo.flatten.to_a)
    end

  end

  describe '#gradient_descent' do

    it 'updates parameters correctly during gradient descent' do
      # Set random seed for reproducible parameter initialization
      srand(42)
      # Setup test data: create a model with specific dimensions (2x2x1 images on 3 classes, one hidden layer with 5 units)
      model = described_class.new(2, 2, 1, 3, layers: [5])

      # Input: flattened images, shape [batch_size, n_x] = [2, 4]
      x = ::Torch.tensor([[1.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0]], dtype: :double)

      # Labels: class indices for CrossEntropyLoss
      y = ::Torch.tensor([0, 1], dtype: :long)

      # Setup optimizer
      optimizer = ::Torch::Optim::Adam.new(model.parameters.map(&:torch_parameter), lr: 0.1)

      # Store original parameter values
      original_params = model.parameters.map { |p| [p.name, p.torch_parameter.numo.dup] }.to_h

      # Forward propagate
      model.initialize_back_propagation_cache
      a = model.forward_propagate(x, train: true)

      # Compute loss
      criterion = ::Torch::NN::CrossEntropyLoss.new
      loss = criterion.call(a, y)

      # Backward propagate
      model.gradient_descent(nil, a, nil, loss)

      # Check that gradients were computed by autograd
      model.parameters.each do |param|
        expect(param.torch_parameter.grad).not_to be_nil
        expect(param.torch_parameter.grad.numo.to_a.flatten.any? { |g| g != 0 }).to be true
      end

      # Update parameters
      optimizer.step

      # Assert that parameters have been updated
      model.parameters.each do |param|
        expect(param.torch_parameter.numo).not_to eq(original_params[param.name])
      end
    end

  end

end
