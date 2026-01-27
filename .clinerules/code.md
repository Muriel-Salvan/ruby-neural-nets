# General rules, applicable to the whole code base

1. Every class should be defined in a single file that follows the module and class name in its path, using snake_case for files and paths.
2. Classes always define their public methods first, followed by their private ones below. The keyword private is separating both parts.
3. Text files should always end with an empty line.
4. Avoid defining local variables that are used only once. Try to use the variable value directly where it is needed (chained calls are preferred to explicit variables definitions).
5. Avoid catching cases of missing or unknown data explicitely: if the data is not in the expected format then a normal exception should be raised, without having to add extra code to support it. For example when accessing a hash's value that is supposed to exist, don't test for its presence (no need for "next if hash[key].nil?").
6. Module names should not be added to class names, constants and class method calls when used inside the module itself. For example, no need to add MyModule:: in front of any class that is used inside the MyModule module.
7. Avoid using global variables, and always try to limit the scope of variables and methods. Don't use instance variables when local variables are enough. Don't make methods public if they are not supposed to be used outside of the class.
8. Each method should have a header documenting its parameters and result, using the following template (example given for a method accepting 2 parameters and returning 2 result values):
  # Main method purpose and behaviour description.
  #
  # Parameters::
  # * *param1_name* (Param1Type): Description of the parameter 1
  # * *param2_name* (Param2Type): Description of the parameter 2 [default: DefaultValue2]
  # Result::
  # * Result1Type: Description of the result element 1
  # * Result2Type: Description of the result element 2
  def my_method(param1_name, param2_name = DefaultValue2)
9. When editing big files, replace_in_file does not work properly. Always check that the file is containing the edits you expect. Use write_in_file when you see that there are no edits in the proposed changes.
