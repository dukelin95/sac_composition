#!/bin/bash

if [ $1 = "prim" ]
then
  docker run \
         --rm \
         -p 5902:5900 \
         -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
         -v /dev/shm:/dev/shm \
         -v /home/duke/sac_composition/mjkey/mjkey.txt:/root/.mujoco/mjkey.txt \
         -v /home/duke/sac_composition/code/:/root/code/ \
         -v /home/duke/sac_composition/robosuite/:/root/robosuite/ \
         -v /home/duke/sac_composition/jacob_sac/:/root/jacob_sac/ \
         -v /home/duke/sac_composition/rllab/:/root/rllab/ \
         -v /home/owner/mujocoFiles/mujoco:/root/rllab/vendor/mujoco \
         --env DISPLAY=$DISPLAY \
         --env CUDA_VISIBLE_DEVICES=0 \
         --runtime nvidia \
         crl_vnc_sawyer
elif [ $1 = "comp" ]
then
  docker run \
           --rm \
           -p 5902:5900 \
           -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
           -v /dev/shm:/dev/shm \
           -v /home/duke/sac_composition/mjkey/mjkey.txt:/root/.mujoco/mjkey.txt \
           -v /home/duke/sac_composition/code/:/root/code/ \
           -v /home/duke/sac_composition/robosuite/:/root/robosuite/ \
           -v /home/duke/sac_composition/crl_code/:/root/crl_code/ \
           -v /home/duke/sac_composition/rllab/:/root/rllab/ \
           -v /home/owner/mujocoFiles/mujoco:/root/rllab/vendor/mujoco \
           --env DISPLAY=$DISPLAY \
           --env CUDA_VISIBLE_DEVICES=0 \
           --runtime nvidia \
           crl_vnc_sawyer
else
  echo need arg: comp or prim
fi
