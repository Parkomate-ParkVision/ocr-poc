# from easyocr.detection import get_detector, get_textbox
import easyocr

reader = easyocr.Reader(
    lang_list=["en"],
    detector=False,
)
# reader.get_detector, reader.get_textbox = get_detector, get_textbox
reader.detector = 'models/best_accuracy.pth'
print(reader.recognize('images/image.png'))
