import cv2
import pytesseract
from datetime import datetime

# Optional: Specify path to tesseract if not in PATH
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

def capture_image():
    cap = cv2.VideoCapture(0)  # 0 is usually the Pi camera
    if not cap.isOpened():
        raise IOError("Cannot open camera")

    print("Capturing image...")
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Failed to capture image from camera.")

    # Optionally save the image
    filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Image saved as {filename}")

    return frame

def extract_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Optional preprocessing
    # gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(gray)
    return text

def main():
    image = capture_image()
    text = extract_text(image)
    print("\n--- OCR Output ---")
    print(text)

if __name__ == "__main__":
    main()
