#!/usr/bin/env sh

xhost +

docker run -it \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume $(realpath ~):${HOME} \
    --workdir ${PWD} \
    $USER/screenpulse/fft_scipy_opencv \
    bash

xhost -
