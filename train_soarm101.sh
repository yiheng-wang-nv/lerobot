HF_USER=Venn
# conda install -c conda-forge ffmpeg=6.1.1 -y
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/so101_pick_and_place_home_fix_video \
  --policy.type=act \
  --output_dir=outputs/train/so101_pick_and_place_home_fix_video \
  --job_name=so101_pick_and_place_home_fix_video \
  --policy.device=cuda \
  --wandb.enable=false