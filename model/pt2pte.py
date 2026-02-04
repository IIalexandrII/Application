from ultralytics import YOLO
import sys
import os

modelFilePath = sys.argv[1]
model = YOLO(modelFilePath)

classes = model.names

export_dir = os.path.splitext(os.path.basename(modelFilePath))[0] + "_executorch_model"
output_path = os.path.join(export_dir, 'classes.txt')

os.makedirs(export_dir, exist_ok=True)

with open(output_path, 'w') as f:
    values = [v for v in classes.values()]
    f.write(','.join(values))

print(f"Classes saved to: {output_path}")
print("Classes:")
for k, v in classes.items():
    print(" ", k, v)

model.export(format="executorch")