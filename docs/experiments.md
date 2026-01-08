# Findings and experiments

## Table of Contents

- [One layer model on colors dataset](#one-layer-model-on-colors-dataset)
- [One layer model on numbers dataset](#one-layer-model-on-numbers-dataset)
- [N layer model on colors dataset](#n-layer-model-on-colors-dataset)
- [N layer model on numbers dataset](#n-layer-model-on-numbers-dataset)
- [N layer model using PyTorch](#n-layer-model-using-pytorch)
  - [Experiment [A]: Same parameters as with Numo implementation](#experiment-a-same-parameters-as-with-numo-implementation)
  - [Experiment [B]: Measuring model randomness effect](#experiment-b-measuring-model-randomness-effect)
  - [Experiment [C]: Measuring dataset randomness effect](#experiment-c-measuring-dataset-randomness-effect)
- [One layer model using ONNX and PyTorch](#one-layer-model-using-onnx-and-pytorch)
- [Performance benchmarks](#performance-benchmarks)
- [Inference on image segmentation](#inference-on-image-segmentation)

## One layer model on colors dataset

```bash
bundle exec ruby ./bin/train --dataset=colors --data-loader=NumoImageMagick --accuracy=ClassesNumo --model=OneLayer --optimizer=Constant
```

The one-layer model provides:
* 1 dense layer reducing the dimensions from the input down to the number of classes used for classification,
* 1 softmax activation layer.
It expects a cross-entropy loss function to be used by the trainer, as its gradient computation will use directly dJ/dz = a - y simplification. This simplification only works when softmax activation is combined with the cross-entropy loss.

With just this model, we can already validate a lot of the framework's capabilities and various techniques:
* Normal processing validates that cost is reducing while accuracy is increasing.
* Using minibatches of small sizes validate that cost decreases in average while increasing in small steps.
* Numerical instability can be seen when the Adam optimizer is used, but not when the constant learning rate is used. Those instabilities can be solved by using the N-Layer model that also includes a Batch-normalization layer.
* Numerical instability can be seen when initialization of the parameters is done randomly instead of using Xavier Glorot's algorithm.
* Invalid gradient computations lead to gradient checking issuing a lot of errors, which validates the gradients checking itself. Those errors usually are visible in the first 5 epochs.
* Once gradient checking, loss function, forward and backward propagations are fixed, we see that gradient checking is nearly constant (around 1e-7) for all kinds of datasets, optimizers, minibatches sizes and models being used.
* Adding gradient checking severely impacts performance, as forward propagation is run an additional 2 * nbr_gradient_checks_samples * nbr_parameters times.
* Adding numerical instability checks severely impacts performance as well.

Performance:
* After 7 epochs, accuracy for both training and dev sets stay at 100%.

## One layer model on numbers dataset

```bash
bundle exec ruby ./bin/train --dataset=numbers --data-loader=NumoImageMagick --accuracy=ClassesNumo --model=OneLayer --optimizer=Constant
```

Using the one-layer model on the numbers dataset validates the following:
* When using the Adam optimizer with a bigger learning rate (0.002) we see that accuracy and cost keep increasing and decreasing. That means the model can't learn anymore certainly due to gradient descent overshooting constantly over the minimum.
* When using the Adam optimizer with a bigger learning rate (0.003) we see numerical instability starting the third epoch.
* When using the N-layer model with 0 hidden layers (`--model=NLayers --layers=`), without BatchNormalization layers and with bias in Dense layers, we see the exact same behavior, which validates the computation of softmax gradients without the dz = a - y shortcut done in the OneLayer model.
* We see that removing gradient checking is not modifying any result, proving that gradient checking does not leak in computations.

Performance:
* The learning is quite slow using the constant optimizer, but still gets better and better up to an accuracy of 20% around epoch 80 that stagnates afterwards. Variance is about 4%, meaning the model has difficulty to learn but tends to overfit a bit the training set.
* The learning is more noisy (cost function is doing bounces) but much faster with the Adam optimizer, up to an accuracy of 55% at epoch 100. However we see variance increasing a lot starting epoch 60, as dev accuracy stays around 20%. This confirms the overfitting that is also visible with the constant optimizer.

## N layer model on colors dataset

```bash
bundle exec ruby ./bin/train --dataset=colors --data-loader=NumoImageMagick --accuracy=ClassesNumo --model=NLayers --optimizer=Adam
```

Observations:
* We see that using BatchNormalization layers allow the Adam optimizer to be used without numerical instability.

Performance:
* We see that the Adam optimizer converges more slowly on simple datasets like the colors one (both dev and training sets have 100% accuracy on epoch 59 instead of 7), but gets better results than the Constant optimizer on complex datasets like the numbers one.

## N layer model on numbers dataset

See [n_layer_model_on_numbers.md](n_layer_model_on_numbers.md) for details.

## N layer model using PyTorch

### Experiment [A]: Same parameters as with Numo implementation

```bash
bundle exec ruby ./bin/train -dataset=numbers --accuracy=ClassesTorch --model=NLayersTorch --optimizer=AdamTorch --loss=CrossEntropyTorch --gradient-checks=off --data-loader=TorchVips --instability-checks=off --grayscale=true --minmax-normalize=true --adaptive-invert=true --trim=true --resize=16,16 --noise-intensity=0 --rot-angle=0 --nbr-clones=1 --layers=16 --max-minibatch-size=100000 --learning-rate=1e-2 --dropout=0 --weight-decay=0.00 --display-samples=4 --track-layer=l0_linear.weight,8
```

![A](torch_numbers/a.png)

* We see the same behaviour as with the Numo implementation, with a smaller dev accuracy (94%), resulting in 6% variance.
* After using 64 bit floats in Torch, the accuracy goes up much faster.
* We can check that using ::Torch::NN::LogSoftmax layer with ::Torch::NN::NLLLoss loss is equivalent (but slower) than not using this last layer with ::Torch::NN::CrossEntropyLoss.
* We see that normalizing inputs between -1 and 1 with mean 0 instead of 0 and 1 with mean 0.5 does not change the performance of the training.

### Experiment [B]: Measuring model randomness effect

```bash
bundle exec ruby bin/train --dataset=numbers --accuracy=ClassesTorch --model=NLayersTorch --optimizer=AdamTorch --loss=CrossEntropyTorch --gradient-checks=off --data-loader=TorchVips --instability-checks=off --grayscale=true --minmax-normalize=true --adaptive-invert=true --trim=true --resize=16,16 --layers=16 --max-minibatch-size=100000 --learning-rate=1e-2 --training-times=5
```

![B](torch_numbers/b.png)

* Different seeds produce a variance of around 3% on dev accuracy, and less than 1% on training accuracy.

### Experiment [C]: Measuring dataset randomness effect

```bash
bundle exec ruby bin/train --instability-checks=off --dataset=numbers --accuracy=ClassesTorch --model=NLayersTorch --optimizer=AdamTorch --loss=CrossEntropyTorch --gradient-checks=off --data-loader=TorchVips --grayscale=true --minmax-normalize=true --adaptive-invert=true --trim=true --resize=16,16 --layers=16 --max-minibatch-size=100000 --learning-rate=1e-2 --experiment --dataset=numbers --accuracy=ClassesTorch --model=NLayersTorch --optimizer=AdamTorch --loss=CrossEntropyTorch --gradient-checks=off --data-loader=TorchVips --grayscale=true --minmax-normalize=true --adaptive-invert=true --trim=true --resize=16,16 --layers=16 --max-minibatch-size=100000 --learning-rate=1e-2 --experiment --dataset=numbers --accuracy=ClassesTorch --model=NLayersTorch --optimizer=AdamTorch --loss=CrossEntropyTorch --gradient-checks=off --data-loader=TorchVips --grayscale=true --minmax-normalize=true --adaptive-invert=true --trim=true --resize=16,16 --layers=16 --max-minibatch-size=100000 --learning-rate=1e-2 --experiment --dataset=numbers --accuracy=ClassesTorch --model=NLayersTorch --optimizer=AdamTorch --loss=CrossEntropyTorch --gradient-checks=off --data-loader=TorchVips --grayscale=true --minmax-normalize=true --adaptive-invert=true --trim=true --resize=16,16 --layers=16 --max-minibatch-size=100000 --learning-rate=1e-2
```

![C](torch_numbers/c.png)

* Different seeds produce a variance of around 3% on dev accuracy, and less than 1% on training accuracy.

## One layer model using ONNX and PyTorch

```bash
bundle exec ruby ./bin/train --dataset=colors --data-loader=TorchImageMagick --accuracy=ClassesTorch --model=OnnxTorch --onnx-model=one_layer --optimizer=AdamTorch --loss=CrossEntropyTorch --gradient-checks=off
```

* We see a similar behavior as the OneLayer model (`bundle exec ruby ./bin/train --dataset=colors --data-loader=NumoImageMagick --accuracy=ClassesNumo --model=OneLayer --optimizer=Adam --loss=CrossEntropy --gradient-checks=off`), which validates the serialization and deserialization of the ONNX models.

## Performance benchmarks

The benchmarks are made on CPU, under VirtualBox kubuntu, using 100 epochs on training on numbers dataset (caching all data preparation in memory), with 1 layer of 100 units, without minibatches.
Absolute values are meaningless as this setup is far from being optimal. However relative values give some comparison ideas between frameworks and algorithms, on the training part.

Here are the command lines used:

```bash
# Numo using Vips
bundle exec ruby ./bin/train --dataset=numbers --gradient-checks=off --instability-checks=off --grayscale=true --minmax-normalize=true --trim=true --adaptive-invert=true --resize=110,110 --data-loader=NumoVips --accuracy=ClassesNumo --model=NLayers --optimizer=Adam --loss=CrossEntropy --display-graphs=false
# Numo using ImageMagick
bundle exec ruby ./bin/train --dataset=numbers --gradient-checks=off --instability-checks=off --grayscale=true --minmax-normalize=true --trim=true --adaptive-invert=true --resize=110,110 --data-loader=NumoImageMagick --accuracy=ClassesNumo --model=NLayers --optimizer=Adam --loss=CrossEntropy --display-graphs=false
# Torch using Vips
bundle exec ruby ./bin/train --dataset=numbers --gradient-checks=off --instability-checks=off --grayscale=true --minmax-normalize=true --trim=true --adaptive-invert=true --resize=110,110 --data-loader=TorchVips --accuracy=ClassesTorch --model=NLayersTorch --optimizer=AdamTorch --loss=CrossEntropyTorch --display-graphs=false
# Torch using ImageMagick
bundle exec ruby ./bin/train --dataset=numbers --gradient-checks=off --instability-checks=off --grayscale=true --minmax-normalize=true --trim=true --adaptive-invert=true --resize=110,110 --data-loader=TorchImageMagick --accuracy=ClassesTorch --model=NLayersTorch --optimizer=AdamTorch --loss=CrossEntropyTorch --display-graphs=false
```

| Experiment              | Elapsed time | Memory consumption (MB) | Final dev accuracy |
| ----------------------- | ------------ | ----------------------- | ------------------ |
| Torch using ImageMagick | 5m 23s       | 760                     | 98%                |
| Numo using ImageMagick  | 5m 23s       | 755                     | 98%                |
| Numo using Vips         | 6m 07s       | 862                     | 97%                |
| Torch using Vips        | 6m 08s       | 765                     | 97%                |

Analysis: Overall performance is consistent between experiments:
* ImageMagick processing is more efficient than Vips (big factor), and results in better accuracy.
* Torch and Numo Ruby implementation have very similar performance.

## Inference on image segmentation

```bash
bundle exec ruby bin/infer --dataset=car --data-loader=TorchImageMagick --resize=320,320 --flatten=false --model=OnnxTorch --onnx-model=u2net
```

Using the ONNX model [U2-Net from Hugging Face](https://huggingface.co/BritishWerewolf/U-2-Net) on its [example image](https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png), we have the following image segmentation.

![Source image](u2net/car.png)

![Segmented image](u2net/car_segmented.png)
