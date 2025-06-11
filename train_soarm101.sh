HF_USER=Venn
# conda install -c conda-forge ffmpeg=6.1.1 -y
pip install -e ".[feetech]"
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/so101_scissors \
  --policy.type=act \
  --output_dir=outputs/train/so101_scissors_act_save_5000 \
  --job_name=so101_scissors_act \
  --policy.device=cuda \
  --wandb.enable=false

# eval

sudo chmod 666 /dev/ttyACM1
sudo chmod 666 /dev/ttyACM0
python lerobot/scripts/find_motors_bus_port.py

python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grip a straight scissor and put it in the box." \
  --control.repo_id=${HF_USER}/eval_so101_scissors_last \
  --control.tags='["so101"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.push_to_hub=false \
  --control.policy.path=outputs/train/so101_scissors_act/checkpoints/last/pretrained_model
