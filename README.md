# 'detect_unstable_object': A ROS package for unstable object detection

The `detect_unstable_object` ROS package provides recognition for detecting unstable objects about to fall.

*   Maintainer: Takaaki Fukui ([tfukui@i.ci.ritsumei.ac.jp](mailto:tfukui@i.ci.ritsumei.ac.jp)).
*   Author: Takaaki Fukui ([tfukui@i.ci.ritsumei.ac.jp](mailto:tfukui@i.ci.ritsumei.ac.jp)).

**Content:**

*   [Topics](#topics)
*   [Action](#action)
*   [Usage](#usage)

## Topics

*   Subscriber: **`detect_unstable_object/image_sub`** ([sensor_msgs/Image])
*   Publisher: **`detect_unstable_object/result_pub`** ([sensor_msgs/Image])

## Action

*   Server: **`detect_unstable_object/action_server`** ([sensor_msgs/Image])

## Usage

0.   Prepare model weights and set configuration.
1.   Execute the launch file: `roslaunch detect_unstable_object detect_unstable_object.launch`
2.   Perform the detection action with a test image: `python cv_detect_unstable_object/scripts/detector_action_client.py test_image.png`
