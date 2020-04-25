pip install -r requirements.txt
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name pedestrian-detection-adas-0002 -o models/ --precisions FP16
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name person-reidentification-retail-0031 -o models/ --precisions FP16