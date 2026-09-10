# Tài liệu luận văn (docs)

| Tài liệu | Nội dung |
|---|---|
| [`datasets.md`](datasets.md) | Thống kê bộ dữ liệu (Harvard-GF + các dataset OCT công khai) & phân tích lựa chọn dataset |
| [`denoise-methods-comparison.md`](denoise-methods-comparison.md) | So sánh phương pháp khử nhiễu speckle (Gaussian/Median/Bilateral/TV/Wavelet/NLM/BM3D/DnCNN/SwinIR) — metric + ảnh mẫu |
| [`2d-feature-extraction-sweep.md`](2d-feature-extraction-sweep.md) | Sweep trích xuất đặc trưng 2D (Tiny2D/ConvNeXtV2/DeiT3/MaxViT × AIP/SlabAIP/SlabMIP) — nhận xét + bảng val/test/calibration cho báo cáo |
| [`fusion-ablation-sweep.md`](fusion-ablation-sweep.md) | Ablation 6 phép fusion 2D–3D (Concat/Add/Attention/CrossGate/Mamba/FiLM) — cảnh báo collapse + bảng val/test cho báo cáo |
| [`3d-backbone-sweep.md`](3d-backbone-sweep.md) | Sweep backbone 3D (17 run: ResNeXt3D/VNet/SegResNet/CNN/MONAI/MedicalNet/3DINO) — xếp hạng + cảnh báo collapse |
| [`final-4branch-design.md`](final-4branch-design.md) | Thiết kế **mô hình 4 nhánh (1×3D + 3×2D)** tổng hợp từ 3 sweep + quy trình huấn luyện chống collapse |
| [`view-redundancy-analysis.md`](view-redundancy-analysis.md) | Phân tích 3 view en-face có trùng lặp không → khuyến nghị số nhánh 2D (1/2/3) |
| [`model-architecture.md`](model-architecture.md) | Kiến trúc mô hình chính (2×2D MaxViT + 1×3D ResNeXt3D + CrossGate) — sơ đồ + params |
| [`multiview-model.md`](multiview-model.md) | Kiến trúc mô hình đa luồng 2D-projection + 3D (Group A) |
| [`train-platform-method.md`](train-platform-method.md) | So sánh nền tảng huấn luyện (local/Colab/vast.ai) & phương pháp tối ưu chi phí-thời gian |
| [`swinunetr-encoder-notes.md`](swinunetr-encoder-notes.md) | Ghi chú encoder Swin-UNETR |

Hình minh hoạ nằm ở thư mục gốc `figures/` (khi tài liệu trong `docs/` dùng đường dẫn `../figures/...`).
