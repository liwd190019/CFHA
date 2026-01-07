yolo task=detect    \
    mode=train \
    model=yolov8n.pt    \
    data=/scratch/qingqu_root/qingqu1/wdli/domain_adaptation/CACTIF/data/manipal_yolo_train/results-synplay-500-manipal-upscale2-styletransform-wo-lowlight/data.yaml \
    epochs=800   \
    imgsz=[720,1280]   \
    batch=24   \
    device=0,1  
    # resume=True

    # model=/scratch/qingqu_root/qingqu1/wdli/domain_adaptation/CACTIF/data/manipal_yolo_train/results-synplay-500-manipal-upscale2-styletransform-wo-lowlight/runs/train27/weights/last.pt    \