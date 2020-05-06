pip install -r requirements.txt
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name pedestrian-detection-adas-0002 -o models/ --precisions FP16
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name person-reidentification-retail-0031 -o models/ --precisions FP16
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name person-reidentification-retail-0248 -o models/ --precisions FP16
cd models
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar xvzf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
cd ssd_mobilenet_v2_coco_2018_03_29
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
