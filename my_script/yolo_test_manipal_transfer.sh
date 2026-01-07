yolo task=detect \
    mode=val \
    model=/scratch/qingqu_root/qingqu1/wdli/domain_adaptation/CACTIF/data/manipal_yolo_train/results-synplay-500-manipal-upscale2-styletransform-wo-lowlight/runs/train27/weights/last.pt \
    data=/scratch/qingqu_root/qingqu1/wdli/domain_adaptation/CACTIF/data/manipal_yolo_train/results-synplay-500-manipal-upscale2-styletransform-wo-lowlight/data.yaml \
    imgsz=[720,1280] \
    batch=28 \
    device=0,1 \
    split=test
