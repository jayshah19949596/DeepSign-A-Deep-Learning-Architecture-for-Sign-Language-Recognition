#!/bin/bash

cd "/usr/local/openpose"

if [$4 == 0]; then
	echo "No hand tracking";
	./build/examples/openpose/openpose.bin --video $1 --write_keypoint_json $3
    # ./build/examples/openpose/openpose.bin --video $1 --write_video $2 --write_keypoint_json $3
else
	echo "Yes hand tracking";
	./build/examples/openpose/openpose.bin --video $1 --hand --hand_tracking --write_keypoint_json $3
    # ./build/examples/openpose/openpose.bin --video $1 --hand --hand_tracking --write_video $2 --write_keypoint_json $3
fi
