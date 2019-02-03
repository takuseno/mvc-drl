#!/bin/bash -eux


sudo docker run -it --rm --runtime nvidia -v ${PWD}:/home/app --name mvc-drl takuseno/mvc-drl bash
