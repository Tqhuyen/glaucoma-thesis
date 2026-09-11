# Huấn luyện CrossGate: W&B, checkpoint và khôi phục

## Preset hiện hành: fine-tune Bilateral 5 epoch

Notebook hiện chỉ chạy **`bilateral_s42`**, khởi tạo từ best model của giai đoạn raw mới nhất:

```text
/content/drive/MyDrive/MasterBKDN/Thesis/final_2x2d_3d_crossgate/
raw_s42_recovered_20260910/raw_s42/best_weights.pt
```

Đây là đường dẫn được tạo bởi cấu hình lần chạy trước, chưa được kiểm tra trực tiếp trên Drive trong phiên sửa code này. Notebook kiểm tra file tồn tại trước khi chuẩn bị dữ liệu; nếu thiếu thì báo lỗi, không dùng lại file emergency cũ và không train từ khởi tạo ngẫu nhiên.

- Nhóm mới: `bilateral_s42_finetune5_from_raw_recovered`, không ghi đè run raw.
- Thời lượng: 5 epoch bổ sung, LR `5e-5`, batch 2, tích lũy 8, seed 42. Patience 5 không cắt ngắn ngân sách 5 epoch thông thường.
- Cả Training, Validation và Test đều đọc `*_volumes_dn.npy`; cả Slab MIP và Full AIP đều được tạo từ chính dữ liệu Bilateral tương ứng.
- Tham số Bilateral: `sigma_color=0.10`, `sigma_spatial=4.0`. Cache hoàn tất đúng cấu hình được tái sử dụng; cache thiếu/dở phải xử lý trước training, không thay bằng raw.
- Best checkpoint được chọn bằng AUC validation Bilateral trong giai đoạn fine-tune. Sau train, fit temperature và threshold trên validation Bilateral rồi đánh giá test Bilateral.
- W&B, checkpoint và Drive giữ nguyên cơ chế bảo vệ. XAI tắt mặc định ở run thật (`RUN_XAI=False`) để không tốn thêm nhiều forward/backward ngoài yêu cầu train và đánh giá.
- Kết quả này phải mô tả là **fine-tune 5 epoch trên Bilateral từ model đã học trên raw**, không phải huấn luyện từ đầu hoàn toàn trên Bilateral.

Các mục dưới đây giải thích cơ chế chung và lưu lại hướng dẫn cho giai đoạn cứu hộ trước. Preset Bilateral ở trên thay thế preset raw cũ.

## 1. Phiên bản mới

Notebook: [3d_glaucoma_final_2x2d_3d_crossgate.ipynb](../notebooks/3d_glaucoma_final_2x2d_3d_crossgate.ipynb).

Logic train, checkpoint và báo cáo nằm trong [scripts/final_training.py](../scripts/final_training.py). Kiến trúc trong `scripts/final_model.py` không thay đổi. Khi chuyển sang Colab cần có cả notebook mới và script mới trong cùng phiên bản repo; mở notebook mới nhưng clone repo cũ chưa có script sẽ không chạy được.

Các thay đổi trên máy local không tự cập nhật kernel Colab đang mở. Sau khi bảo đảm file cứu hộ đã có trên Drive, nên dùng runtime mới và chạy các cell theo thứ tự. Không chạy lại các cell khởi tạo trong runtime cũ nếu vẫn cần cứu thêm optimizer hoặc đối tượng đang nằm trong bộ nhớ.

## 2. Tiếp tục từ file trọng số vừa cứu

Preset trước đây là `raw_s42_recovered_20260910`: chỉ raw, seed 42, warm-start từ file cứu hộ, tối đa 10 epoch bổ sung. Preset đó đã được thay bằng Bilateral 5 epoch ở đầu tài liệu. Không dùng lại ví dụ cứu hộ dưới đây cho giai đoạn Bilateral hiện tại.

File đã cứu là state dictionary, không có optimizer, scheduler, epoch hoặc RNG. Vì vậy đây là **warm-start**, không phải resume chính xác lần train cũ. Bộ trọng số được giữ lại; trạng thái huấn luyện và W&B bắt đầu mới.

Trong cell `Configuration`, sử dụng:

```python
RUN_GROUP = 'crossgate_recovered_v1'
RUN_TARGET = 'raw_s42'
RESUME = False
WARM_START_WEIGHTS = (
    '/content/drive/MyDrive/MasterBKDN/Thesis/'
    'final_2x2d_3d_crossgate/'
    'recovered_crossgate_20260910_220644_live_weights.pt'
)
WARM_START_TARGET = RUN_TARGET
CHECKPOINT_EVERY_STEPS = 25
```

**`raw_s42` chỉ là ví dụ.** Thay bằng dataset/seed mà bạn đã xác nhận từ log cũ. Không thể suy ra dataset hoặc seed từ trọng số. Nếu chưa xác nhận, không trình bày run warm-start như một lần tái lập chính xác thí nghiệm trước.

`RUN_TARGET` phải thuộc `DATASETS` và `SEEDS`. Khi chỉ chọn raw, bước chuẩn bị dữ liệu không chạy bilateral. Chọn một `RUN_GROUP` mới để không ghi đè thí nghiệm đã có. File emergency của phiên bản mới, chứa `weights` và `resumable=False`, cũng có thể được dùng làm warm-start.

Chạy lần lượt: setup → configuration → data/cache → train/evaluate → summary. Nếu dùng runtime đã import bản script cũ, khởi động runtime mới sau khi lưu an toàn để tránh dùng module cũ trong bộ nhớ.

## 3. Dừng an toàn

Trong cell training, bấm Stop/Interrupt **một lần**. Khi SIGINT được kernel chuyển đến chương trình, trainer ghi nhận yêu cầu dừng, hoàn thành cửa sổ gradient accumulation hiện tại, cập nhật optimizer/scaler/scheduler, xóa gradient và ghi checkpoint trước khi thoát.

Chờ thông báo `Stopped safely`. Không bấm Stop liên tục và không ngắt kết nối/xóa runtime trong lúc lưu. Nếu đang validation, yêu cầu dừng có thể phải chờ validation hoàn tất. Với volume 200³, thời gian chờ một cửa sổ accumulation có thể đáng kể.

Lần interrupt thứ hai hoặc một lỗi khác kích hoạt lưu emergency weights theo khả năng hiện có, không cố thay `last.pt` bằng trạng thái đang cập nhật dở. Không có phần mềm Python nào bảo đảm lưu thành công nếu tiến trình bị kill, runtime bị xóa, GPU lỗi nghiêm trọng hoặc hết dung lượng. Khi đó dùng checkpoint an toàn gần nhất đã ghi thành công.

`ACTIVE_MODEL` và `ACTIVE_TRAINER` được giữ để kiểm tra nếu có lỗi; không cần phụ thuộc vào traceback để tìm model như notebook cũ. Không tiếp tục bằng cách gọi lại trainer bị lỗi, vì trạng thái trong RAM có thể chưa nhất quán. Tạo lại trainer bằng quy trình resume.

## 4. Resume từ checkpoint mới

Giữ nguyên nhóm run và cấu hình đã lưu:

```python
RUN_GROUP = 'crossgate_recovered_v1'
RUN_TARGET = 'raw_s42'
RESUME = True
WARM_START_WEIGHTS = ''
WARM_START_TARGET = ''
```

Chạy lại các cell theo thứ tự. `last.pt` được nạp từ local nếu có, hoặc từ Drive khi local không có. Không thay đổi epoch budget, learning rate, batch size, số lần tích lũy hoặc các trường cấu hình khác khi resume; cấu hình và định danh dữ liệu được đối chiếu để tránh khôi phục sai thí nghiệm.

Với `RUN_TARGET = ''` và `RESUME = True`, sweep xử lý riêng từng tag: bỏ qua run đã hoàn tất đủ artifact, resume run có checkpoint, và bắt đầu run chưa từng chạy. Run mới chỉ có identity mà chưa có checkpoint được khởi tạo lại với identity đó. Nếu run này là warm-start, file trọng số gốc phải còn tồn tại.

Checkpoint lưu tại ranh giới cập nhật an toàn gồm model, optimizer, scheduler, AMP scaler, RNG, thứ tự mẫu, vị trí trong epoch, lịch sử, thống kê một phần epoch và best weights. Phần công việc sau checkpoint cuối có thể được chạy lại. Augmentation được xác định theo seed/epoch/index và loader dùng zero workers. Đây không phải cam kết khôi phục tại một lệnh bất kỳ hoặc tái lập bitwise trên mọi GPU/phiên bản thư viện.

## 5. Các file được lưu

Thư mục local:

```text
outputs/final_2x2d_3d_crossgate/<RUN_GROUP>/<tag>/
```

Thư mục Drive:

```text
/content/drive/MyDrive/MasterBKDN/Thesis/
  final_2x2d_3d_crossgate/<RUN_GROUP>/<tag>/
```

| File | Mục đích |
|---|---|
| `last.pt` | Checkpoint đầy đủ để resume, lưu trước batch đầu, định kỳ theo bước optimizer, mỗi epoch và khi dừng an toàn. |
| `best.pt` | Checkpoint đầy đủ khi validation AUC cải thiện. |
| `best_weights.pt` | State dictionary của best model, xuất trong giai đoạn đánh giá sau train. |
| `run_identity.pt` | W&B run ID và cấu hình để tiếp tục đúng run. |
| `emergency_weights_*.pt` | Trọng số cứu hộ khi lỗi/ngắt lần hai; không phải checkpoint resume đầy đủ. |
| `metrics.json`, `test_predictions.pt` | Báo cáo và xác suất/nhãn test. |
| `curves.png`, `test_report.png` | Đồ thị train/validation và đánh giá test. |
| Các file XAI | Grad-CAM, occlusion, integrated gradients và phân tích fusion. |
| `completed.pt` | Đánh dấu hoàn tất sau báo cáo, XAI và W&B finish. |

`CHECKPOINT_EVERY_STEPS=25` nghĩa là sau 25 bước cập nhật optimizer thành công, không phải 25 minibatch. Giảm giá trị này nếu muốn giảm lượng công việc có thể mất, đổi lại tăng I/O và tải lên Drive. Cấu hình này cũng được kiểm tra khi resume.

Local được ghi trước bằng file tạm và đổi tên. Copy Drive lỗi sẽ báo lỗi rõ ràng, giữ file local và tạo dấu `.sync-pending.pt`; không được coi đó là đã đồng bộ. Sau khi sửa kết nối/dung lượng, resume trên runtime còn file local để đẩy lại checkpoint. Nếu mất runtime trước khi đồng bộ, chỉ bản Drive gần nhất còn tồn tại.

Checkpoint đầy đủ có thể lớn hơn nhiều so với file 223 MiB chỉ chứa trọng số vì còn optimizer và best snapshot. Kiểm tra dung lượng local/Drive trước khi chạy sweep. Chỉ nạp full checkpoint do chính bạn tạo và tin cậy; nó chứa trạng thái Python, không chỉ tensors.

## 6. W&B và đánh giá

- `.env` được nạp trước `wandb.init`; Colab Secrets `WANDB_API_KEY` cũng được hỗ trợ.
- Run thật yêu cầu W&B online; lỗi xác thực không bị bỏ qua để âm thầm train không có logging.
- Train loss/accuracy/LR được log trực tiếp trong quá trình cập nhật; validation metrics log mỗi epoch; test chỉ đánh giá sau lựa chọn best checkpoint.
- Resume dùng lại W&B run ID. Phần chạy lại từ checkpoint có thể tạo các giá trị progress trùng nhau; không diễn giải như các bước dữ liệu độc lập.
- Validation AUC là tiêu chí chọn best. Temperature scaling được fit trên validation, sau đó chọn threshold trên xác suất validation đã calibration; test và bootstrap dùng cùng threshold/thang xác suất.
- Báo cáo và hình được đồng bộ ngay sau khi tạo. Summary của nhóm run được tổng hợp từ kết quả đã lưu, không chỉ run vừa chọn bằng `RUN_TARGET`.

## 7. Dữ liệu và denoise

Dữ liệu thật giữ độ phân giải lưu trữ 200³. File denoise dở được ghi dưới tên partial và không được đưa vào training. Marker hoàn tất chứa nguồn, phương pháp, tham số và định danh implementation. Cache không có marker từ notebook cũ sẽ không được tin cậy tự động.

Hai ảnh chiếu của run bilateral được tạo từ thể tích bilateral, thay vì tái sử dụng ảnh chiếu raw như notebook cũ. Đây là sửa đổi tiền xử lý quan trọng; khi warm-start từ run cũ cần ghi rõ sự thay đổi này nếu run cũ dùng dữ liệu denoised.

`DENOISE_LIMIT > 0` giới hạn số mẫu xử lý thêm mỗi split trong một lần chạy cell. Nếu chưa hoàn thành, chạy lại cell dữ liệu để tiếp tục. Khi thay tham số denoise, dùng `DATA_ROOT` và `RUN_GROUP` mới để giữ nguyên thí nghiệm cũ. Partial denoise chỉ được phục hồi local; dữ liệu denoise hoàn chỉnh và view cache được đồng bộ Drive trước training. Trên runtime mới, cần phục hồi/chuẩn bị lại dữ liệu trước khi resume model; notebook không tự tải lại toàn bộ data cache từ Drive.

## 8. Kiểm thử và giới hạn

Bộ kiểm thử gồm checkpoint round-trip, warm-start, config mismatch, lỗi Drive, resume sweep, bảo toàn bảng tổng hợp, cache denoise và SIGINT thật khi còn gradient tích lũy. Notebook được chạy tuần tự toàn bộ code cell bằng dữ liệu synthetic CPU và model nhỏ ở chế độ smoke, với W&B offline được bật tường minh.

Kết quả kiểm tra local: **35 tests passed**, gồm kiểm tra preset Bilateral 5 epoch và smoke thực sự đi qua bước lọc Bilateral, đọc ba split denoise và tạo view từ nguồn denoise. Có một cảnh báo backward hook từ Grad-CAM trong smoke. Chưa kiểm thử W&B online, Google Drive thật, CUDA AMP hoặc huấn luyện toàn bộ MaxViT/ResNeXt3D trên dữ liệu 200³. Smoke không chứng minh hội tụ hoặc hiệu năng chẩn đoán của mô hình thật.
