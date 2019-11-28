cmake_minimum_required(VERSION 2.8.3)
project(detect_unstable_object)

find_package(catkin REQUIRED COMPONENTS
  rospy
  actionlib_msgs
  std_msgs
  message_generation
  sensor_msgs
  message_generation
)

## Generate messages in the 'msg' folder
#add_message_files(
#  FILES
#  Message1.msg
#  Message2.msg
#)

## Generate services in the 'srv' folder
add_service_files(
  FILES
    DetectUnstableObject.srv
)

## Generate actions in the 'action' folder
add_action_files(
  FILES
    DetectUnstableObject.action
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  actionlib_msgs
)

catkin_package(
 CATKIN_DEPENDS 
 rospy 
 std_msgs 
 message_runtime 
 sensor_msgs 
 actionlib_msgs 
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

