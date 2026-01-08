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
      display_samples
      dev_experiment
      early_stopping_patience
      dump_minibatches
    ],
    keyword_init: true
  )

  # Return a unique experiment id, taking into account previously instantiated experiments
  #
  # Parameters::
  # * *experiment_id* (String): Wishful experiment id
  # * *experiments* (Array<Experiment>): Existing experiments
  # Result::
  # * String: Unique experiment id
  def self.unique_experiment_id(experiment_id, experiments)
    exp_id_candidate = experiment_id
    exp_id_idx = 0
    while experiments.find { |select_exp| select_exp.exp_id == exp_id_candidate }
      exp_id_candidate = "#{experiment_id}_#{exp_id_idx}"
      exp_id_idx += 1
    end
    exp_id_candidate
  end

end
