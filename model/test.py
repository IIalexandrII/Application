from ultralytics import YOLO

# Load the YOLO26 model
model = YOLO("yolo26s.pt")

# Export the model to TFLite format
model.export(format="tflite")  # creates 'yolo26n_float32.tflite'

# Load the exported TFLite model
tflite_model = YOLO("yolo26n_float32.tflite")
