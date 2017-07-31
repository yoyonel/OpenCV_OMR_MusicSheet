#!/usr/bin/env sh

xhost +

DOCKER_IMAGE=omr_opencv3_zsh:python2
DOCKER_SHELL=bash

docker run \
	-it --rm \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume $(realpath ~):${HOME} \
    --workdir ${PWD} \
    --volume /etc/inputrc:/etc/inputrc \
    $DOCKER_IMAGE \
    $DOCKER_SHELL

xhost -