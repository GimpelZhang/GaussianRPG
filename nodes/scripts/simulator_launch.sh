#!/bin/bash
CC=/usr/bin/gcc
CXX=/usr/bin/g++

# Setup the paths
relative_path=$0
absolute_path=$(readlink -f "$relative_path")
scripts_dir=$(dirname "$absolute_path")
nodes_dir=$(dirname "$scripts_dir")
nerf_dir=$(dirname "$nodes_dir")

source $nodes_dir/install/setup.bash

# Launch the rviz2 window:
rviz2 -d $nodes_dir/rviz2/gaussian_rpg.rviz &

# Launch the dummy controller:
ros2 run dummy_controllers aeb_controller &

# Launch the simulator:
ros2 run simulator evaluation 12.00 2.0 & # args: simulation_time collision_threshold
ros2 run simulator ground_truth ../output/waymo_full_exp/waymo_train_002_1cam/trajectory/ours_50000/cams_tape.json ../data/waymo/training/002/track/track_info.txt 98 6 &
ros2 run simulator simulator --config $nerf_dir/configs/example/waymo_train_002_1cam.yaml &

sleep 3

# Monitor the simulation process:
while true; do
    if ! pgrep -x "evaluation" > /dev/null; then
        echo "Simulation process ends."
        killall -9 aeb_controller
        killall -9 ground_truth
        killall -9 simulator
        break
    fi
    sleep 1
done

# Kill the rviz2 gui manually if you need.
sleep 1