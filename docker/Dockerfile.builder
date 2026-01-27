# Dockerfile for Ruby Neural Nets project
# Ubuntu 25.10 base image with all dependencies

FROM ubuntu:25.10

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
    \
    # Ruby dependencies
    ruby-full \
    ruby-dev \
    bundler \
    \
    # Image processing
    imagemagick \
    libmagickwand-dev \
    ffmpeg \
    libvips42 \
    libvips-dev \
    \
    # Other dependencies
    libyaml-dev \
    xdg-utils \
    gnuplot \
    x11-xserver-utils \
    protobuf-compiler \
    libsqlite3-dev \
    \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Install additional Ruby dependencies
RUN gem install bundler

# Download and install libTorch
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.9.1%2Bcpu.zip && \
    unzip libtorch-shared-with-deps-2.9.1+cpu.zip -d /opt/libtorch && \
    rm libtorch-shared-with-deps-2.9.1+cpu.zip && \
    echo "LIBTORCH_PATH=/opt/libtorch/libtorch" >> /etc/environment

# Clone and modify torchvision-ruby repository
RUN git clone https://github.com/ankane/torchvision-ruby.git /opt/torchvision-ruby && \
    sed -i 's/gem "numo-narray"/gem "numo-narray-alt"/' /opt/torchvision-ruby/Gemfile
