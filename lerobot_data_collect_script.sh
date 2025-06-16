# install dependencies
pip install -e ".[feetech]"

# find motors bus port
python lerobot/scripts/find_motors_bus_port.py

# open ports
HF_USER=Venn
sudo chmod 666 /dev/ttyACM1
sudo chmod 666 /dev/ttyACM0

# find cameras
python lerobot/common/robot_devices/cameras/opencv.py \
  --images-dir outputs/images_from_opencv_cameras

# adjust cameras
python adjust_cameras.py

# teleoperate
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=teleoperate \
  --control.display_data=true\
  --control.fps=15
  
# collect data
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=record \
  --control.fps=15 \
  --control.single_task="Grip a straight scissor and put it in the box." \
  --control.repo_id=${HF_USER}/2_cameras_fps15_enhanced_gripper \
  --control.tags='["so101"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=60 \
  --control.reset_time_s=60 \
  --control.num_episodes=200 \
  --control.display_data=true \
  --control.push_to_hub=false \
  --control.resume=true

# Grip a straight scissor and put it in the box.
# Grip a tweezer and put it in the box.
