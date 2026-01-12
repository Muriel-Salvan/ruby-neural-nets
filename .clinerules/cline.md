# General rules, applicable to the whole code base

1. Every class should be defined in a single file that follows the module and class name in its path, using snake_case for files and paths.
2. Classes always define their public methods first, followed by their private ones below. The keyword private is separating both parts.
3. Avoid defining local variables that are used only once. Try to use the variable value directly where it is needed.
4. Avoid catching cases of missing or unknown data explicitely: if the data is not in the expected format then a normal exception should be raised, without having to add extra code to support it. For example when accessing a hash's value that is supposed to exist, don't test for its presence (no need for "next if hash[key].nil?").
5. Any code or test modification should be validated by running all the tests (not only the modified ones) using `bundle exec rspec`. It is expected that it runs without any error, with the mention "0 failures" in the output.
6. Any code or test modification should trigger a verification of the README.md file content, and make sure that all sections still have up-to-date content regarding the change.
7. Text files should always end with a carriage return character.
8. Module names should not be added to class names, constants and class method calls when used instead the module instead. For example, no need to add RubyNeuralNets:: in front of any class that is used inside the RubyNeuralNets module.

# Documentation rules

1. README.md file should always be kept up-to-date with the new options or architecture changes that are made. Any section of the README should be checked if the content needs to be adapted with any code or test change.
2. Each method should have a header documenting its parameters and result, using the following template (example given for a method accepting 2 parameters and returning 2 result values):
  # Main method purpose and behaviour description.
  #
  # Parameters::
  # * *param1_name* (Param1Type): Description of the parameter 1
  # * *param2_name* (Param2Type): Description of the parameter 2 [default: DefaultValue2]
  # Result::
  # * Result1Type: Description of the result element 1
  # * Result2Type: Description of the result element 2

# Test rules, applicable to the test code base (inside the ./spec directory)

1. Platform-agnostic unit test scenarios are all defined under the spec/scenarios directory.
2. Platform-specific unit test scenarios are all defined under the spec/scenarios.#{RUBY_PLATFORM} directory.
3. Unit test framework and helpers are all defined under the spec/ruby_neural_nets_test directory.
4. Unit test scenarios are grouped per kind of interface being tested. Only the following kinds are tested:
  * Under spec/scenarios/data_loaders: All unit tests testing data loaders.
  * Under spec/scenarios/models: All unit tests testing models.
  * Under spec/scenarios/optimizers: All unit tests testing optimizers.
  * Under spec/scenarios/trainers: All unit tests testing trainer.
5. Unit test spec file name should follow the template spec/scenarios/<interface_kind>/<class_name>_spec.rb .
6. In the case too many scenarios have to be tested for a single class, then the scenarios will be split among different files all part of the same directory, following this template: spec/scenarios/<interface_kind>/<class_name>/<scenarios_group_meaning>_spec.rb .
7. Each unit test scenario should have the same structure:
  1. Setup test data and mocks.
  2. Call only the public interface of the class to be tested.
  3. Write simple assertions on results of the public interface of the class to be tested.
8. Each unit test scenario should run in an isolated way: running scenarios in whatever order or group should never change the result of the test. If side effects are found while running test scenarios that are impacting other test scenarios, those side effects should be removed to make sure each test scenario runs in isolation.

# Cline specific rules

1. Always try to group all edits for a given task in the same file in 1 step.
2. For big files, replace_in_file does not work properly. Use write_in_file when you see that there are no edits in the proposed changes.
3. Cline should run unit tests only with WSL, using the following command line: `wsl --distribution Ubuntu -- bash -i -c '\"$(ls /mnt/*/Env/Linux/Programs/bash/bash)\" -i -c \"bundle exec rspec\"'`. Don't forget to include the antislashes before the double quotes exactly as it is written, otherwise escaping won't work.
