from ultralytics import YOLO

# Путь к твоей TorchScript-модели
model_path = "best.torchscript"

# Загружаем модель через Ultralytics
model = YOLO(model_path)

# Запускаем вебкамеру (source=0)
# Для реального времени можно использовать track, predict тоже работает
model.track(source=0, show=True, conf=0.3)
