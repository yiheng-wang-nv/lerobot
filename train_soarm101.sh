HF_USER=Venn
# conda install -c conda-forge ffmpeg=6.1.1 -y
pip install -e ".[feetech]"
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/so101_tweezer \
  --policy.type=act \
  --output_dir=outputs/train/so101_tweezer_act \
  --job_name=so101_tweezer_act \
  --policy.device=cuda \
  --wandb.enable=false

pip install -e ".[pi0]"
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/so101_pick_and_place_home_fix_video \
  --policy.type=pi0 \
  --output_dir=outputs/train/so101_pick_and_place_home_fix_video_pi0 \
  --job_name=so101_pick_and_place_home_fix_video_pi0 \
  --policy.device=cuda \
  --wandb.enable=false