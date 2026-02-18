#https://github.com/pytorch/executorch/issues/14644

import torch
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower
from ultralytics import YOLO

yolo_model = YOLO("best.pt")
yolo_model=yolo_model.model
yolo_model.eval()

device = 'cpu'
yolo_model = yolo_model.to(device)

sample_inputs = (torch.randn(1, 3, 640, 640).to(device),)

# Dry run
yolo_model(*sample_inputs)

exported=torch.export.export(yolo_model,args=sample_inputs)

et_program = to_edge_transform_and_lower(
    exported,
    partitioner=[VulkanPartitioner()],
).to_executorch()

with open("bestVulkan.pte", "wb") as file:
    et_program.write_to_file(file)