# YOLO11_gradio_nuclei
Interactive nuclei detection in H&amp;E images using Gradio and YOLO11

**The source dataset** is [NuInsSeg](https://www.kaggle.com/datasets/ipateam/nuinsseg), a fully annotated dataset for nuclei instance segmentation in H&amp;E-stained images.

Mahbod, A., Polak, C., Feldmann, K. et al. NuInsSeg: A fully annotated dataset for nuclei instance segmentation in H&E-stained histological images. Sci Data 11, 295 (2024).
https://doi.org/10.1038/s41597-024-03117-2

The dataset (image::mask pairs) were converted to the YOLO format using scripts developed by [bnsreenu](https://github.com/bnsreenu)

***Bhattiprolu, S. (2023). python_for_microscopists. GitHub.*** [336-Nuclei-Instance-Detectron2.0_YOLOv8_code](https://github.com/bnsreenu/python_for_microscopists/tree/master/336-Nuclei-Instance-Detectron2.0_YOLOv8_code)


1. Create and activate a new virtual environment using conda:
```bash
  conda create -n yolov11_nuc python=3.10
  conda activate yolov11_nuc
```
2. Use pip to install the required libraries:
```bash
  pip install jupyterlab ultralytics tensorboard scikit-learn opencv-python gradio
```
3. Within yolo_v11_gradio_predict.ipynb, build a new YOLOv11 instance segmentation model and transfer weights from the pretrained model:
```bash
  model =  YOLO("yolo11n-seg.yaml")
  model = YOLO("yolo11n-seg.pt")
  model =  YOLO("yolo11n-seg.yaml").load("yolo11n-seg.pt")
```
4. Train the model while visualizing training progress via tensorboard.
5. Launch a Gradio interface to interactively upload and segment H&E images using the newly trained model:
```bash
  import gradio as gr
  from ultralytics import YOLO
  
  model = YOLO("yolo_dataset/results/5_epochs-/weights/last.pt")
  
  def predict_image(img, conf_threshold, iou_threshold):
      results = model.predict(
          source=img,
          conf=conf_threshold,
          iou=iou_threshold,
          show_labels=True,
          show_conf=True,
      )
      return results[0].plot() if results else None
  
  iface = gr.Interface(
      fn=predict_image,
      inputs=[
          gr.Image(type="pil", label="Upload Image"),
          gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
          gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
      ],
      outputs=gr.Image(type="pil", label="Result"),
      title="Gradio YOLO11 Nuclei Segmentation",
      description="Upload images of H&E nuclei for YOLO11 detection.",
  )
  iface.launch(share=True)
```
![Screenshot of the Gradio Web Interface](/assets/images/gradio_interface_screenshot.png)
