import tensorflow as tf
from inference import Network
import cv2
import time


def pre_process(frame, net_input_shape):
    p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
    p_frame = p_frame.transpose(2, 0, 1)
    # p_frame = np.expand_dims(p_frame, axis=1)
    p_frame = p_frame.reshape(1, *p_frame.shape)
    return p_frame


def test_from_frozen_graph(pb_file, img_cv2):
    img_cv2 = cv2.resize(img_cv2, (224, 224))
    img = img_cv2[:, :, [2, 1, 0]]

    # Read the graph.
    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    with tf.Session() as sess:
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        inference_start_time = time.time()
        outputs = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                            sess.graph.get_tensor_by_name('detection_scores:0'),
                            sess.graph.get_tensor_by_name('detection_boxes:0'),
                            sess.graph.get_tensor_by_name('detection_classes:0')],
                           feed_dict={
                               'image_tensor:0': img.reshape(1,
                                                             img.shape[0],
                                                             img.shape[1], 3)})
        inference_end_time = time.time()
        total_inference_time = inference_end_time - inference_start_time
        confidence = outputs[1][0][0]
        detection = outputs[2][0][0]
        return str(round(total_inference_time * 1000, 3)) + "ms", confidence


def post_convertion(frame, model, cpu_extension, device):
    network = Network()
    network.load_model(model, cpu_extension, device)
    processed_frame = pre_process(frame, net_input_shape=network.get_input_shape())
    inference_start_time = time.time()
    network.exec_net(processed_frame)
    if network.wait() == 0:
        inference_end_time = time.time()
        total_inference_time = inference_end_time - inference_start_time
        result = network.get_all_output()
        output = result['DetectionOutput']
        detection = output[0][0][0]
        image_id, label, conf, x_min, y_min, x_max, y_max = detection
        return str(round(total_inference_time * 1000, 3)) + "ms", conf


image = cv2.imread('resources/test.png')

print("PERFORMANCE OF THE ORIGINAL MODEL")
print(test_from_frozen_graph('models/tensorflow/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb', image))

print("OPENVINO PERFORMANCE AFTER CONVERTING THE MODEL")
print(post_convertion(image, 'models/tensorflow/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml', None,
                      'CPU'))
