# cv_detect_unstable_object
ROS package of detection unstable object

## Nodes
### Subscribed Topics
* **`detect_unstable_object/image_sub`** ([sensor_msgs/Image])

### Published Topics
* **`detect_unstable_object/result_pub`** ([sensor_msgs/Image])

### Action
* **`detect_unstable_object/action_server`** ([sensor_msgs/Image])

## Usage
0. Prepare model weight and fix config  

1. Exec launch file.   
`roslaunch detect_unstable_object detect_unstable_object.launch`

2. Send test image to acton.  
`python cv_detect_unstable_object/scripts/detector_action_client.py test_image.png`
