import tensorflow as tf

graph_def_file = "ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb"
input_arrays = ["image_tensor"]
output_arrays = ["detection_scores","detection_boxes","detection_classes","detection_masks"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file, input_arrays, output_arrays,input_shapes={"image_tensor":[1,600,600,3]})
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)