require 'ruby_neural_nets/models/n_layers'
require 'ruby_neural_nets/optimizers/constant'
require 'ruby_neural_nets/losses/cross_entropy'

describe RubyNeuralNets::Models::NLayers do

  describe '#stats' do

    it 'returns the right model statistics' do
      # Setup test data: create a model with specific dimensions (2x2x1 images on 3 classes, one hidden layer with 5 units)
      expect(described_class.new(2, 2, 1, 3, layers: [5], dropout_rate: 0.0).stats).to eq(
        {
          parameters: {
            'L0_Dense_W' => { size: 20 }, # 5 units * 4 features
            'L2_BatchNormalization_Gamma' => { size: 5 },
            'L2_BatchNormalization_Beta' => { size: 5 },
            'L4_Dense_W' => { size: 15 }, # 3 classes * 5 units
            'L6_BatchNormalization_Gamma' => { size: 3 },
            'L6_BatchNormalization_Beta' => { size: 3 }
          }
        }
      )
    end

  end

  describe '#forward_propagate' do

    it 'produces expected output values for given input using initialized model parameters' do
      # Set random seed for reproducible parameter initialization
      srand(42)
      # Setup test data: create a model with specific dimensions (2x2x1 images on 3 classes, one hidden layer with 5 units)
      model = described_class.new(2, 2, 1, 3, layers: [5], dropout_rate: 0.0)

      # Input: simple flattened images, shape [n_x, minibatch_size] = [4, 2] for 2 samples
      x = Numo::DFloat[[1.0, 0.0],
                       [0.0, 1.0],
                       [0.0, 0.0],
                       [0.0, 0.0]]

      # Initialize back propagation cache
      model.initialize_back_propagation_cache

      # Call forward_propagate
      output = model.forward_propagate(x)

      # Assert output shape and properties
      expect(output.shape).to eq([3, 2])
      # Each column should sum to 1 (softmax property)
      (0...2).each do |j|
        expect((0...3).inject(0.0) { |s, i| s + output[i, j] }).to be_within(1e-6).of(1.0)
      end

      # Compute expected output manually by propagating through layers
      # Layer 0: Dense (5 units, no bias)
      w0 = model.parameters(name: 'L0_Dense_W').first.values
      a0 = w0.dot(x)

      # Layer 2: BatchNormalization
      gamma0 = model.parameters(name: 'L2_BatchNormalization_Gamma').first.values
      beta0 = model.parameters(name: 'L2_BatchNormalization_Beta').first.values
      mean0 = a0.mean(axis: 1, keepdims: true)
      var0 = ((a0 - mean0)**2).mean(axis: 1, keepdims: true)
      sqrt_var_eps0 = Numo::NMath.sqrt(var0 + 1e-5)
      x_hat0 = (a0 - mean0) / sqrt_var_eps0
      a1 = gamma0 * x_hat0 + beta0

      # Layer 3: LeakyRelu
      mask = Numo::DFloat.cast(a1.gt(0))
      a1 = mask * a1 + (1 - mask) * a1 * 0.01

      # Layer 4: Dense (3 units, no bias)
      w1 = model.parameters(name: 'L4_Dense_W').first.values
      a2 = w1.dot(a1)

      # Layer 6: BatchNormalization
      gamma1 = model.parameters(name: 'L6_BatchNormalization_Gamma').first.values
      beta1 = model.parameters(name: 'L6_BatchNormalization_Beta').first.values
      mean1 = a2.mean(axis: 1, keepdims: true)
      var1 = ((a2 - mean1)**2).mean(axis: 1, keepdims: true)
      sqrt_var_eps1 = Numo::NMath.sqrt(var1 + 1e-5)
      x_hat1 = (a2 - mean1) / sqrt_var_eps1
      a3 = gamma1 * x_hat1 + beta1

      # Layer 7: Softmax
      expect_array_within(
        output.to_a,
        (0...2).map do |sample_idx|
          exp_values = a3[true, sample_idx].to_a.map { |val| Math.exp(val) }
          sum_exp = exp_values.sum
          exp_values.map { |val| val / sum_exp }
        end.transpose
      )
    end

  end

  describe '#gradient_descent' do

    it 'updates parameters correctly during gradient descent' do
      # Set random seed for reproducible parameter initialization
      srand(42)
      # Setup test data: create a model with specific dimensions (2x2x1 images on 3 classes, one hidden layer with 5 units)
      model = described_class.new(2, 2, 1, 3, layers: [5], dropout_rate: 0.0)

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

      # Setup loss function
      loss = RubyNeuralNets::Losses::CrossEntropy.new

      # Setup optimizer and link to parameters (use Constant for predictable updates)
      optimizer = RubyNeuralNets::Optimizers::Constant.new(learning_rate: 0.1, weight_decay: 0.0)
      optimizer.teach_parameters(model.parameters)

      # Store original parameter values
      original_params = model.parameters.map { |p| [p.name, p.values.dup] }.to_h

      # Forward and backward propagate
      optimizer.start_minibatch(0, minibatch.size)
      model.initialize_back_propagation_cache
      a = model.forward_propagate(x, train: true)
      # da is computed using the loss function
      da = loss.compute_loss_gradient(a, y, model)
      model.gradient_descent(da, a, minibatch, 1.0)

      # Assert that parameters have been updated
      model.parameters.each do |param|
        expect(param.values).not_to eq(original_params[param.name])
      end
    end

  end

end
