# rsviz as a docker image: for any Linux with docker, whatever its distribution, with no ROS and no Rust on the host.
#   docker build -t rsviz .          then          ./rsviz-docker.sh [arguments of rsviz]
# Two stages: rsviz is built with Rust and clang on ROS 2 Jazzy, then only the binary goes into a ROS 2 Jazzy image
# that has what it needs to draw (Vulkan and OpenGL through Mesa, or the NVIDIA driver given by the NVIDIA runtime).
FROM ros:jazzy-ros-base AS build
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates clang libclang-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
COPY . /rsviz
RUN . /opt/ros/jazzy/setup.sh && . /root/.cargo/env && cd /rsviz && cargo build --release

FROM ros:jazzy-ros-base
RUN apt-get update && apt-get install -y --no-install-recommends \
      libvulkan1 mesa-vulkan-drivers libgl1 libgl1-mesa-dri libegl1 libgles2 \
      libx11-6 libx11-xcb1 libxcb1 libxcursor1 libxi6 libxrandr2 libxkbcommon0 libxkbcommon-x11-0 \
      libwayland-client0 libwayland-cursor0 libwayland-egl1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /rsviz/target/release/rsviz /usr/local/bin/rsviz
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/jazzy/setup.bash && exec rsviz \"$@\"", "rsviz"]
