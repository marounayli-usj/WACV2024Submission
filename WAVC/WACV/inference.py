from ultralytics import YOLO
from PIL import Image

from paddleocr import PaddleOCR
ocr = PaddleOCR(lang="en", use_angle_cls=True,show_log=False)

model = YOLO('runs/detect/train22/weights/best.pt')
import cv2

class WebElementList:
    def __init__(self, elements):
        self.elements = elements

    def by_class(self, element_class):
        filtered_elements = [element for element in self.elements if element.element_class == element_class]
        return WebElementList(filtered_elements)

    def by_text(self, text):
        filtered_elements = [element for element in self.elements if element.text == text]
        return WebElementList(filtered_elements)

    def get_elements(self):
        return self.elements


class WebElement:
    def __init__(self, bbox, element_class, text=None):
        self.bbox = bbox  # Tuple (x, y)
        self.element_class = element_class  # String
        self.text = text  # Nullable String



# Display model information (optional)
model.info()

names = model.names

image = cv2.imread("cropped/image-3.png")
color = (0, 255, 0)  # Green
thickness = 2

results = model.predict(source=image, save=False)
for elt in results:
    boxes = elt.boxes
    objects = [{} for _ in range(len(boxes))]
    for i, c in enumerate(boxes.cls):
        objects[i]["class"] = names[int(c)]

    box = elt.boxes.xyxy
    boxes_np = box.cpu().numpy()

    for i, row in enumerate(boxes_np):
        x1, y1, x2, y2 = row
        objects[i]["box"] = (int(x1), int(y1), int(x2), int(y2))

        # Crop the region of interest (ROI) from the original image
        roi = image[int(y1):int(y2), int(x1):int(x2)]

        # Convert the ROI to a format suitable for PaddleOCR
        roi_np = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)


        if len(roi_np) >0:
        # Apply OCR to the ROI
            ocr_results = ocr.ocr(roi_np)
            # print(ocr_results)
        # Process OCR results and store them in the 'objects' dictionary
        texts = []
        for line in ocr_results[0]:
            _, (text, confidence) = line
            texts.append({"text": text, "confidence": confidence})

        objects[i]["text"] = texts

print(objects)

# cv2.imwrite('cropped/inference.png', image) 