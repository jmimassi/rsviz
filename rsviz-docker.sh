#!/usr/bin/env bash
# rsviz from its docker image, on the screen of this machine: any Linux with docker, no ROS and no Rust on the host.
#   ./rsviz-docker.sh [arguments of rsviz]          (the image is built the first time: a few minutes, needs internet)
# The container shares the network and the IPC of the host, so it sees the ROS 2 topics of this machine and of the
# containers started the same way (--network host --ipc host); the home folder is mounted read only, to open bags by
# their path. It draws with the NVIDIA GPU when docker has the NVIDIA runtime, else with Mesa (the GPU of /dev/dri, or
# on the CPU). ROS_DOMAIN_ID, ROS_AUTOMATIC_DISCOVERY_RANGE, RMW_IMPLEMENTATION and the RSVIZ_* / WGPU_* variables of the
# caller go into the container. RSVIZ_IMAGE: the image (default rsviz).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMAGE=${RSVIZ_IMAGE:-rsviz}
docker image inspect "$IMAGE" > /dev/null 2>&1 || docker build -t "$IMAGE" "$HERE" || exit 1
OPT=()
if docker info 2> /dev/null | grep -q 'Runtimes:.*nvidia' || command -v nvidia-container-cli > /dev/null; then
  OPT+=(--gpus all)
elif [ -d /dev/dri ]; then
  OPT+=(--device /dev/dri)
  for g in video render; do getent group $g > /dev/null && OPT+=(--group-add "$(getent group $g | cut -d: -f3)"); done
fi
[ -n "${DISPLAY:-}" ] && OPT+=(-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix)
[ -n "${XAUTHORITY:-}" ] && OPT+=(-e XAUTHORITY -v "$XAUTHORITY:$XAUTHORITY:ro")
for v in $(compgen -e | grep -E '^(ROS_DOMAIN_ID|ROS_AUTOMATIC_DISCOVERY_RANGE|RMW_IMPLEMENTATION|RSVIZ_.*|WGPU_.*)$' | grep -v '^RSVIZ_IMAGE$'); do
  OPT+=(-e "$v")
done
TTY=; [ -t 0 ] && TTY=-it
exec docker run --rm $TTY --init --network host --ipc host --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$HOME:$HOME:ro" -w "$PWD" "${OPT[@]}" "$IMAGE" "$@"
