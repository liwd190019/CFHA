# base_dir_path="/scratch/qingqu_root/qingqu1/wdli/domain_adaptation/CACTIF/data/aro_datasets/sementic_transformed/train"
base_dir_path="/scratch/qingqu_root/qingqu1/wdli/domain_adaptation/CACTIF/data/manipal_update2_sequential/Part_3"

# # stage 1: determine the cluster, and calculate clip score for each person.
# python data_processing/yolo_to_json_cluster4_stage1_batch.py  \
#     --labels_dir $base_dir_path/labels \
#     --masks_dir $base_dir_path/masks   \
#     --images_dir $base_dir_path/resized_images  \
#     --output_dir $base_dir_path/json_labels_stage1    \
#     --ref /scratch/qingqu_root/qingqu1/wdli/domain_adaptation/CACTIF/data/manipal-20/ref_person_test_semantic.pt

# # stage 2: compute cluster likelihoods
# python data_processing/yolo_to_json_cluster4_stage2.py  \
#     --json_dir $base_dir_path/json_labels_stage1  \
#     --output_dir $base_dir_path/json_labels_stage2    \
#     --temperature 0.7

# # stage 3: select clusters based on the likelihood
# python data_processing/yolo_to_json_cluster4_stage3.py  \
#     --json_dir $base_dir_path/json_labels_stage2    \
#     --output_dir $base_dir_path/json_labels_stage3    \
#     --metric "prob"   \
#     --k 0-8  \
#     --select 'sample'

# stage 4: update masks
python data_processing/update_masks_from_json_stage4.py \
    --masks_dir $base_dir_path/masks   \
    --json_dir $base_dir_path/json_labels_stage3   \
    --output_dir $base_dir_path/masks_selected_update \
    --search_radius 3

# stage 5: binarize masks

# stage 6: paste person to image
python data_processing/paste_person_to_image_stage6.py \
  --orig_dir $base_dir_path/resized_images  \
  --removed_dir $base_dir_path/resized_images_after_AE2_new_removal \
  --mask_dir $base_dir_path/masks_selected_update \
  --output_dir $base_dir_path/pasted_image_update \
  --feather 1 --open_iter 0 --close_iter 0


# # stage 7: 
# python data_processing/json2yolo_kept_persons_stage7.py \
#     --json_dir $base_dir_path/json_labels_stage3   \
#     --labels_out $base_dir_path/labels_kept

# stage 8:
python data_processing/viz_yolo_boxes_stage8.py \
    --images_dir $base_dir_path/pasted_image_update   \
    --labels_dir $base_dir_path/labels_kept    \
    --output_dir $base_dir_path/images_for_checking_update
