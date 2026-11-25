# Computer Vision Detection

A learning project (2025): object detection on uploaded images with a pre-trained YOLOv8 model in a Streamlit app.

**Live demo:** https://computer-vision-detection.streamlit.app

## What it does

- Loads the pre-trained YOLOv8 Nano model (Ultralytics, COCO, 80 classes); no custom training.
- Runs detection on an uploaded image with an adjustable confidence threshold.
- Draws bounding boxes and labels, shows inference time and a confidence distribution, and lets you download the annotated image.

## Stack

Python · Streamlit · Ultralytics YOLOv8 · OpenCV · Pillow · Matplotlib

## Run locally

```bash
git clone https://github.com/IrinaDragunow/computer-vision-detection.git
cd computer-vision-detection
pip install -r requirements.txt
streamlit run app.py
```

`packages.txt` lists the system libraries OpenCV needs on Streamlit Cloud.

## Limitations

- Uses the generic COCO classes; detecting domain-specific defects (e.g. in manufacturing quality control) would require a labelled dataset and fine-tuning.
- Single images only, no video stream.
- Prototype for learning, not a production system.
