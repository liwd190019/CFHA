python data_processing/yolo_to_json_cluster4_stage1.py  \
    --labels_dir /scratch/qingqu_root/qingqu1/wdli/domain_adaptation/CACTIF/data/synplay_detector/synplay_test_for_removal/train/labels \
    --masks_dir /scratch/qingqu_root/qingqu1/wdli/domain_adaptation/CACTIF/data/synplay_detector/synplay_test_for_removal/train/masks   \
    --images_dir /scratch/qingqu_root/qingqu1/wdli/domain_adaptation/CACTIF/data/synplay_detector/synplay_test_for_removal/train/resized_image  \
    --output_dir /scratch/qingqu_root/qingqu1/wdli/domain_adaptation/CACTIF/data/synplay_detector/synplay_test_for_removal/train/json_labels    \
    --ref /scratch/qingqu_root/qingqu1/wdli/domain_adaptation/CACTIF/learned_clip_text_embedding/ref_person_test.pt \
    --min_keep 5    \
    --max_keep  20  
