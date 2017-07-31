#!/bin/bash

# url: https://stackoverflow.com/questions/29274638/opencv-libdc1394-error-failed-to-initialize-libdc1394
ln /dev/null /dev/raw1394

exec "$@"