from imageai.Detection import ObjectDetection
import os

exec_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(
	exec_path, "resnet50_coco_best_v2.1.0.h5")
)
detector.loadModel()

list = detector.detectObjectsFromImage(
	input_image=os.path.join(exec_path, "image-3.jpeg"),
	output_image_path=os.path.join(exec_path, "new_objects-3.jpg"),
	minimum_percentage_probability=70,
	display_percentage_probability=True,
	display_object_name=True
)