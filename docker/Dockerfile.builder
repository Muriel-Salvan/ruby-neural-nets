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
