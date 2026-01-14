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
  def my_method(param1_name, param2_name = DefaultValue2)
