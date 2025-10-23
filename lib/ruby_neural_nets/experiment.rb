module RubyNeuralNets
  
  # Represent 1 experiment to be run.
  # An experiment is comprised of:
  # * a model,
  # * a data loader,
  # * a training or evaluation configuration (profiling, checks...),
  # * hyperparameters.
  Experiment = Struct.new(*%i[
      exp_id
      dataset
      accuracy
      data_loader
      loss
      model
      optimizer
      nbr_epochs
      gradient_checker
      profiler
      training_mode
      display_units
    ],
    keyword_init: true
  )

end
