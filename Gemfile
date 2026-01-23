source 'https://rubygems.org'

gem 'fiddle'
gem 'get_process_mem'
gem 'google-protobuf', '~> 3.21'
gem 'irb'
gem 'numo-gnuplot'
gem 'numo-random'
# Rake is used for google-protobuf
gem 'rake'
gem 'reline'
gem 'rmagick'
gem 'rspec'
gem 'ruby-prof'
gem 'ruby-vips'
gem 'streamio-ffmpeg'
gem 'sys-proctable'
gem 'terminal-table'
case RUBY_PLATFORM
when 'x86_64-linux'
  gem 'numo-linalg-alt'
  gem 'numo-narray-alt'
  gem 'iruby'
  gem 'matplotlib'
  # Rice 4.8.0 is not compatible with torch.rb
  gem 'rice', '4.7.1'
  gem 'torch-rb'
  gem 'torchvision', path: (
    proc do
      # Discover automatically torchvision-ruby path, so that it works also in git worktrees
      current_dir = __dir__
      current_dir = File.dirname(current_dir) while current_dir != '/' && !File.exist?("#{current_dir}/torchvision-ruby")
      raise 'Can\'t find torchvision-ruby next to parent directories' if current_dir == '/'
      "#{current_dir}/torchvision-ruby"
    end.call
  )
when 'x64-mingw-ucrt'
  gem 'numo-linalg'
  gem 'numo-narray'
  # This is used by sys-proctable on Windows
  gem 'win32ole'
end

# Development dependencies
gem 'byebug'

# Tools dependencies
gem 'sqlite3'
