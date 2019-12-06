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

add_service_files(
  FILES
    DetectUnstableObject.srv
)

add_action_files(
  FILES
    DetectUnstableObject.action
)

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
