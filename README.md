A repository for testing out different face recognition stacks 

## Setup
- Create virtual environment
```bash
python -m venv .venv
```
- Enter into virtual environment
```bash
. ./.venv/Scripts/activate
```
- Install python packages
```bash
pip install -r requirements.txt
```

### Yunet + SFace
On the root directory of this project, run this commands
```bash
Invoke-WebRequest `
  -Uri "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" `
  -OutFile ".\models\face_detection_yunet_2023mar.onnx"

curl.exe -L `
  -o .\models\face_recognition_sface_2021dec.onnx `
  https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```