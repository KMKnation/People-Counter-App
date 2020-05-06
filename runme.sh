python3 -m venv venv
source /opt/intel/openvino/bin/setupvars.sh
source venv/bin/activate
pip install -r requirements.txt
mkdir models
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name pedestrian-detection-adas-0002 -o models/ --precisions FP16
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name person-reidentification-retail-0031 -o models/ --precisions FP16
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name person-reidentification-retail-0248 -o models/ --precisions FP16
cd models
mkdir tensorflow
cd tensorflow
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar xvzf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
cd ssd_mobilenet_v2_coco_2018_03_29
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
cd ../../../
cd webservice/server
npm install
cd ../ui
npm install
cd ../../
cd webservice/server
kill $(sudo lsof -t -i:3001)
node server.js &
cd ../ui
kill $(sudo lsof -t -i:3000)
npm run dev &
cd ../../
kill $(sudo lsof -t -i:3004)
ffserver -f ./ffmpeg/server.conf &
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/tensorflow/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -m2 models/intel/person-reidentification-retail-0031/FP16/person-reidentification-retail-0031.xml -d CPU -pt 0.10 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm


