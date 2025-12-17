source 'https://rubygems.org'

gem 'fiddle'
gem 'get_process_mem'
gem 'irb'
gem 'numo-gnuplot'
gem 'numo-random'
gem 'reline'
gem 'rmagick'
gem 'rspec'
gem 'ruby-prof'
gem 'ruby-vips'
gem 'sys-proctable'
gem 'terminal-table'
case RUBY_PLATFORM
when 'x86_64-linux'
  gem 'numo-linalg-alt'
  gem 'numo-narray-alt'
  gem 'iruby'
  gem 'matplotlib'
  gem 'torch-rb'
  gem 'torchvision', path: "#{__dir__}/../torchvision-ruby"
when 'x64-mingw-ucrt'
  gem 'numo-linalg'
  gem 'numo-narray'
  # This is used by sys-proctable on Windows
  gem 'win32ole'
end

gem 'byebug'
