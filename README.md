# 3D Pose Reconstruction using GCN-MLP

Dự án xây dựng mô hình GraphMLP để ước lượng tư thế 3D (3D pose estimation) từ video, kết hợp Graph Convolutional Networks (GCN) và Multi-Layer Perceptron (MLP).

## ✨ Tính năng

- 🎯 Phát hiện tư thế 2D từ video
- 🏃‍♂️ Tái tạo tư thế 3D từ keypoints 2D
- 📊 Hỗ trợ huấn luyện trên dataset Human3.6M và MPI-INF-3DHP
- 🎨 Web demo với Streamlit để visualize kết quả
- 🔄 Tích hợp refinement model để cải thiện độ chính xác

## 📋 Yêu cầu

- Python 3.x
- PyTorch
- CUDA (khuyến nghị cho training)

## 🚀 Cài đặt

```bash
# Clone repository
git clone https://github.com/ThieuHuy43/3D-reconstruction-GCN-MLP.git
cd 3D-reconstruction-GCN-MLP

# Cài đặt dependencies
pip install -r requirements.txt
```

## 💻 Sử dụng

### Training

```bash
python main.py --dataset h36m --train --gpu 0
```

### Demo với Streamlit

```bash
streamlit run app.py
```

Mở trình duyệt và upload video để xem kết quả ước lượng tư thế 3D. 

### Demo GUI

```bash
python demo_gui.py
```

## 📁 Cấu trúc thư mục

```
├── model/          # Mô hình GraphMLP và các block
├── common/         # Utilities, dataset loaders, arguments
├── checkpoint/     # Saved model checkpoints
├── dataset/        # Dataset files
├── demo/           # Demo files và outputs
├── main.py         # Training script
├── app.py          # Streamlit web demo
└── demo_gui.py     # GUI demo
```

## 📊 Datasets

Hỗ trợ:
- Human3.6M
- MPI-INF-3DHP

## 🙏 Acknowledgments

Dựa trên nghiên cứu về GraphMLP cho bài toán 3D pose estimation. 
