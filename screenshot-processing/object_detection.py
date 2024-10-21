from inference import get_model
import supervision as sv
import os
import cv2

def detect_web_components(image):

    model = get_model(model_id="website-screenshots/1", api_key="lTOAOMecY7UN1KeKaclH")

    results = model.infer(image)[0]

    detections = sv.Detections.from_inference(results)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    sv.plot_image(annotated_image)
    cv2.imwrite("annotated_image.png", annotated_image)

if __name__ == '__main__':
    image_path = "./data/screenshots/image.png"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    detect_web_components(image)