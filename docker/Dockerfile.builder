# Dockerfile for Ruby Neural Nets project
# Ubuntu 25.10 base image with all dependencies
FROM ubuntu:25.10

ARG ruby_version=3.4
ARG ruby_version_patch=8
ARG bundler_version=4.0.4
ARG libtorch_version=2.9.1

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Basic tools
    curl \
    git \
    wget \
    build-essential \
    software-properties-common \
    ca-certificates \
    gnupg \
    lsb-release \
    unzip \
    \
    # Ruby build dependencies
    autoconf \
    bison \
    libssl-dev \
    libyaml-dev \
    libreadline-dev \
    zlib1g-dev \
    libncurses5-dev \
    libffi-dev \
    libgdbm-dev \
    libgdbm-compat-dev \
    libdb-dev \
    \
    # Image processing
    imagemagick \
    libmagickwand-dev \
    ffmpeg \
    libvips42 \
    libvips-dev \
    \
    # Other dependencies
    xdg-utils \
    gnuplot \
    x11-xserver-utils \
    protobuf-compiler \
    libsqlite3-dev \
    \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Install Ruby from source
RUN cd /tmp && \
    wget https://cache.ruby-lang.org/pub/ruby/${ruby_version}/ruby-${ruby_version}.${ruby_version_patch}.tar.gz && \
    tar -xzf ruby-${ruby_version}.${ruby_version_patch}.tar.gz && \
    cd ruby-${ruby_version}.${ruby_version_patch} && \
    ./configure --disable-install-doc && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/ruby-${ruby_version}.${ruby_version_patch}*

# Install Bundler
RUN gem install bundler -v ${bundler_version}

# Download and install libTorch
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${libtorch_version}%2Bcpu.zip && \
    unzip libtorch-shared-with-deps-${libtorch_version}+cpu.zip -d /opt/libtorch && \
    rm libtorch-shared-with-deps-${libtorch_version}+cpu.zip

# Clone and modify torchvision-ruby repository
RUN git clone https://github.com/ankane/torchvision-ruby.git /opt/torchvision-ruby && \
    sed -i 's/numo-narray/numo-narray-alt/' /opt/torchvision-ruby/torchvision.gemspec

# Temporarily get the Gemfile to cache dependencies
COPY Gemfile /opt/ruby-neural-nets/Gemfile

# Configure bundler globally so that it finds all dependencies, and install them already
RUN bundle config set --global path /opt/vendor/bundle && \
    bundle config set --global build.torch-rb "--with-torch-dir=/opt/libtorch/libtorch" && \
    cd /opt/ruby-neural-nets && \
    bundle install && \
    cd .. && \
    rm -rf /opt/ruby-neural-nets
