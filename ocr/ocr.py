import pytesseract
import cv2

from google.cloud import vision
client = vision.ImageAnnotatorClient()

def extract_text_from_image(image, config: str = "", lang_code: str = "eng") -> str:
    """
    * Function Name:    extract_text_from_image
    * Input:            image: np.ndarray, config: str, lang_code: str
    * Output:           str
    * Logic:            This function extracts text from the image using Google Vision API or Pytesseract.
    """


    # Google OCR using Vision API
    _, encoded_image = cv2.imencode('.png', image)
    content = encoded_image.tobytes()

    if content is None:
      print("Content is None")
      return
    
    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)
    texts = response.text_annotations

    if texts:
        return texts[0].description
    else:
        print("No text detected")
        return " "

    ## Use above OR uncomment below to use Pytesseract
    # text = pytesseract.image_to_string(image, lang= lang_code, config=config)
    # return text