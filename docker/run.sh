#!/bin/env bash

docker run --rm -it \
    -v .:/workspace \
    -v /dev:/dev \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --runtime=nvidia \
    --gpus all \
    --net=host \
    --ipc=host \
    --pid=host \
    --env="DISPLAY=$DISPLAY" \
    --env="SDL_JOYSTICK_HIDAPI=0" \
    --privileged \
    ros-tb3
