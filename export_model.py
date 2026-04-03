from ultralytics import YOLO

# Load model (downloads automatically)
model = YOLO("yolov8n.pt")

# Export to ONNX
model.export(format="onnx", opset=11)

print("✅ Export complete!")
