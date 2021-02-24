from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
# Load model
config_file = '/home/devteam/tabledetection/Config/cascade_mask_rcnn_hrnetv2p_w32_20e.py'
checkpoint_file = '/home/devteam/tabledetection/Config/epoch_36.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Test a single image 
img = "/home/devteam/tabledetection/Demo/demo.png"

# Run Inference
result = inference_detector(model, img)

# Visualization results
show_result_pyplot(img, result,('Bordered', 'cell', 'Borderless'), score_thr=0.85)