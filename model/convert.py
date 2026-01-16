from ultralytics import YOLO
import sys

modelFilePath = sys.argv[1]
model = YOLO(modelFilePath)

print(model.names)

model.export(
    format="torchscript",
)

#PyTorch: starting from 'best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 14, 8400) (5.9 MB)
#{0: 'Light', 1: 'Trafic', 2: 'Person', 3: 'Scooter', 4: 'Open_Manhole', 5: 'Door', 6: 'Stairs', 7: 'Car', 8: 'Bus', 9: 'Bicycle'}