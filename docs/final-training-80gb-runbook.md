# Chạy mô hình cuối trên GPU 80 GB: quy trình và kiểm chứng

## 1. Phạm vi cố định

Notebook [3d_glaucoma_final_2x2d_3d_crossgate.ipynb](../notebooks/3d_glaucoma_final_2x2d_3d_crossgate.ipynb) là đối tượng chuẩn hóa trong lượt này. Notebook sweep là nguồn tham khảo bố cục, không bị sửa. Theo xác nhận của người dùng, các commit mới trên remote được hợp nhất: giữ notebook ba nhánh đã có, khả năng dò batch/worker, mở rộng epoch và bộ phân tích information theory. Preset model cuối vẫn giữ nguyên phạm vi Bilateral 5 epoch, không tự chạy notebook khác. “Lưu DB” được thực hiện bằng **W&B + các artifact có kiểm tra nội dung trên Google Drive**, không bổ sung SQLite hoặc máy chủ DB.

| Thành phần | Cấu hình |
|---|---|
| Giai đoạn | Fine-tune Bilateral 5 epoch từ model raw đã có |
| Run | `bilateral_s42` |
| Nhóm mới | `bilateral_s42_finetune5_80gb_v1` |
| Model gốc | `raw_s42_recovered_20260910/raw_s42/best_weights.pt` |
| Kiến trúc | ResNeXt3D + hai MaxViT-Tiny + CrossGate, không thay layer/stride/kích thước đặc trưng |
| Dữ liệu | Cả Training, Validation, Test và hai view đều dùng Bilateral |
| Kích thước | Volume 200³, hai view 224² |
| Batch / tích lũy | 2 / 8; effective batch 16 |
| LR / weight decay | `5e-5` / `1e-4` |
| Epoch / patience | 5 / 5 |
| Checkpoint | Mỗi 10 bước optimizer, mỗi epoch, yêu cầu dừng và sau khoảng 300 giây tại ranh giới an toàn tiếp theo |
| Precision | Chọn BF16 nếu GPU hỗ trợ, nếu không dùng FP16 với scaler; lưu lựa chọn vào cấu hình |
| Tối ưu truyền dữ liệu | Pinned memory, nonblocking transfer; zero workers để giữ quy trình resume xác định |
| XAI | Tắt mặc định, section riêng |

Không tự tăng batch, đổi độ phân giải, thay nhánh, bật compile hoặc checkpointing activation. Những thay đổi đó chưa được chứng minh có lợi trên phần cứng đích và có thể thay đổi hành vi huấn luyện. `FUSED_ADAMW=False` mặc định; nếu bật thì phải qua preflight với đúng backend đó. Đây là cấu hình ưu tiên tính kiểm soát và tính tương thích, **không phải tuyên bố đã tìm được tốc độ tối đa trên mọi GPU 80 GB**.

## 2. Việc phải làm trước khi thuê GPU

Chuẩn bị/checksum/download/khử nhiễu có thể tốn nhiều thời gian CPU và truyền dữ liệu. Không nên giữ GPU trả phí chỉ để chờ những bước này. Các section 1–6 không cần dựng model CUDA hoặc train; nên hoàn thành trên runtime CPU trước.

1. Kiểm tra model gốc trên Drive, gồm `best_weights.pt`, `last.pt`, `run_identity.pt` và `metrics.json` của run raw. Không thay bằng file emergency khi thiếu file.
2. Chạy section 1–4 để xác nhận code, W&B/Drive, model gốc và raw split. Hash dữ liệu raw/nhãn phải khớp sáu định danh đã lưu trong checkpoint cha.
3. Chọn cache Bilateral đã hoàn tất. Nếu dùng manifest portable của bộ trợ giúp mới, đặt `CACHE_MANIFEST` tới manifest cần khôi phục khi không thể tự chọn duy nhất.
4. Nếu đã có export từ công cụ CPU `prepare_bilateral_200.py`, đặt `CPU_EXPORT_ROOT` tới thư mục export có `manifest.json`. Bộ nhập kiểm tra revision nguồn, nhãn, hash đầu ra và công thức lọc; chỉ tạo view còn thiếu, không lọc lại volume đã xác minh.
5. Nếu thật sự chưa có Bilateral, bật `ALLOW_BUILD_DENOISED=True` ở runtime CPU để tạo. `DENOISE_LIMIT` giới hạn số mẫu xử lý thêm mỗi split; chạy lại section chuẩn bị khi còn partial.
6. Khi cần xuất cache mới lên Drive, bật riêng `PUBLISH_DATA_CACHE=True`. Việc tải lên có thể gồm hàng chục GB; cho phép build không tự đồng nghĩa cho phép upload. Cache đã có đúng nội dung được bỏ qua thay vì upload lại.
7. Chỉ chuyển sang GPU sau khi section 6 hoàn thành. Dữ liệu chưa hoàn tất hoặc chưa được xuất/khôi phục hợp lệ sẽ chặn training.

Nếu checkpoint gốc hoặc cache không khớp, dừng để giải quyết ở CPU. Không bỏ qua các kiểm tra bằng cách đổi tên raw thành denoised hoặc tự tạo marker hoàn tất.

## 3. Sections và cách chạy

| Section | Nội dung | Có dùng GPU không? |
|---|---|---|
| 0 | Hợp đồng thí nghiệm và giới hạn | Không |
| 1 | Setup, phiên bản, hash code trước import | Không cần GPU |
| 2 | Cấu hình, xác thực Drive và W&B | Không cần GPU |
| 3 | Xác minh checkpoint cha | CPU |
| 4 | Raw download, revision và hash so với checkpoint cha | CPU/network |
| 5 | Khôi phục hoặc chuẩn bị Bilateral | CPU/disk/network |
| 6 | Audit đầy đủ volume, nhãn, view, trục depth | CPU/disk |
| 7 | Preflight đầy đủ trên GPU đích | Có, chỉ khi bật |
| 8 | Train/resume Bilateral 5 epoch | Có, chỉ khi bật và preflight hợp lệ |
| 9 | Inference validation/test độc lập | Có nếu chưa có logits; chỉ khi bật |
| 10 | Calibration, metrics, bootstrap, plots và báo cáo từ logits | CPU, không train lại |
| 11 | XAI độc lập | Có ở run thật nếu bật |
| 12 | Information theory tùy chọn: thu thập embedding rồi phân tích theo từng bước | Cần model khi thu thập; các bước sau chạy CPU |
| 13 | Audit nội dung artifact và công bố tổng hợp | Không cần GPU |

Ở run thật, các cờ dưới đây mặc định là `False`:

```python
ENABLE_GPU_PREFLIGHT = False
ENABLE_TRAIN = False
ENABLE_EVAL = False
ALLOW_BUILD_DENOISED = False
PUBLISH_DATA_CACHE = False
RUN_XAI = False
```

Các cờ nằm trong cell Configuration; có thể đổi chúng bằng cell gán biến ngắn ngay trước section cần chạy, không phải chạy lại cả notebook. Không chạy lại Configuration chỉ để bật cờ nếu bạn vẫn cần giữ đối tượng đang train trong RAM.

Sau khi dữ liệu đã được chuẩn bị và khôi phục trên runtime GPU, bật và chạy section 7 trước:

```python
ENABLE_GPU_PREFLIGHT = True
```

Chỉ khi preflight thành công và đã xem bộ nhớ/thời gian đo, bật section 8:

```python
ENABLE_TRAIN = True
```

Sau khi section 8 lưu handoff và best weights thành công, bật section 9:

```python
ENABLE_EVAL = True
```

Sau section 9, chạy section 10 và 13. Không cần bật XAI hoặc information theory để có bảng kết quả phân loại đầy đủ. `Run all` với cấu hình mặc định không tự train, nhưng vẫn có thể thực hiện các bước CPU/network đã cho phép; nó không phải chế độ chỉ xem notebook.

## 4. Preflight thực sự kiểm tra gì?

Preflight yêu cầu CUDA, tổng VRAM tối thiểu 70 GiB và đủ bộ nhớ trống cho ngân sách được đặt. Nó gọi `ft.find_batch_size` với tập ứng viên chỉ gồm batch 2 cho preset này: xác minh batch đã chọn, không âm thầm đổi điều kiện thí nghiệm. Nó dùng **toàn model**, volume 200³, hai view 224², đúng batch/precision/backend. Hai cửa sổ optimizer tương ứng 16 microbatch được chạy trên model dùng thử, gồm cấp phát trạng thái Adam và backward khi gradient đã tồn tại.

Preflight không gọi `Trainer.fit`, không chạy epoch dữ liệu thật và không ghi đè checkpoint training. Nó có cập nhật optimizer của model dùng thử, sau đó giải phóng model đó và khôi phục RNG. Report ghi peak allocated/reserved, thông tin GPU/software, thời gian và định danh cấu hình/code/data/parent. Training chỉ được chạy với preflight khớp các định danh này.

Preflight cũng tiêu thời gian GPU. Thời gian ngoại suy từ nó không bao gồm đầy đủ download, denoise, đồng bộ Drive, biến thiên tốc độ hoặc các lỗi dịch vụ. Qua preflight không bảo đảm toàn bộ 5 epoch sẽ không gặp sự cố; nó là một chốt kiểm tra tải trước khi chạy dài.

Khi preflight thất bại, không tự giảm batch hay resolution để chạy tiếp. Cần kiểm tra lỗi và chốt lại cấu hình. Máy local chỉ có GPU 4 GB, nên **chưa có số đo 80 GB trong lần bàn giao này**.

## 5. Dừng và resume

Bấm Stop một lần: SIGINT yêu cầu trainer dừng ở ranh giới cập nhật an toàn rồi lưu checkpoint. SIGTERM cũng được xử lý theo cơ chế yêu cầu dừng. Không bấm liên tục hoặc xóa runtime trong lúc lưu. Lỗi nghiêm trọng hoặc lần interrupt thứ hai chỉ có thể lưu emergency weights theo khả năng; không cam kết cứu được khi tiến trình bị kill.

Sau lỗi, `fx.ACTIVE_MODEL` và `fx.ACTIVE_TRAINER` giữ trạng thái để kiểm tra. Sau khi chắc chắn đã có checkpoint an toàn, dùng runtime mới hoặc `fx.release_active()` trước khi dựng lại model. Hàm release không thay thế thao tác lưu; không gọi nó nếu vẫn cần cứu thêm trạng thái trong RAM.

Để resume, giữ đúng nhóm/config và đặt `RESUME=True`, rồi chạy lại các section cần thiết để khôi phục/audit dữ liệu và preflight trên GPU đích. Resume dùng checkpoint của **giai đoạn Bilateral này**, không dùng optimizer của run raw. Nếu chỉ còn identity do lỗi xảy ra trước checkpoint đầu tiên, chương trình kiểm tra identity và parent rồi khởi tạo lại đúng warm-start với cùng W&B ID. Resume vào nhóm hoàn toàn mới bị từ chối.

Checkpoint bao gồm model, optimizer, scheduler, scaler khi có, RNG, thứ tự mẫu/cursor, lịch sử và thống kê epoch dở. Phần công việc sau checkpoint cuối có thể được chạy lại. Không hứa tái lập bitwise giữa các GPU hoặc phiên bản phần mềm.

## 6. Lưu đầy đủ W&B và Drive

Thư mục Drive của nhóm:

```text
/content/drive/MyDrive/MasterBKDN/Thesis/
final_2x2d_3d_crossgate/bilateral_s42_finetune5_80gb_v1/
```

Các artifact được tổ chức theo stage thay vì giữ trong một biến notebook. Nhóm có context, code identity, parent identity, dữ liệu và cấu hình execution; các stage có W&B identity/session và marker audit/completion riêng.

| Stage/artifact | Dữ liệu được giữ |
|---|---|
| Parent và data manifest | Hash model nguồn, raw/labels/dn/views/depth; tham số lọc và producer provenance |
| Preflight | GPU/stack, peak memory, thời gian và binding với cấu hình thực thi |
| Training | `last.pt`, `best.pt`, `best_weights.pt`, lịch sử, raw step JSONL, telemetry, W&B ID và `handoff.pt` |
| Evaluation | Logits và labels validation/test, chỉ số mẫu theo split, checkpoint/source identity và receipt nội dung |
| Analysis | Xác suất trước/sau calibration, temperature, threshold, metrics, bootstrap CI, CSV, Markdown và các hình |
| XAI nếu bật | Attribution/figures và định danh mẫu validation/checkpoint |
| Completion | Danh sách artifact đã được kiểm tra bằng SHA và tổng hợp stage hoàn tất |

W&B log train/validation trực tiếp và các thông số hệ thống CPU/RAM/disk/VRAM, cùng kết quả cuối. Evaluation/report/XAI có run phân tích riêng liên kết với training ID. Không âm thầm bỏ W&B khi xác thực lỗi. Nội dung khoa học được đồng bộ và kiểm tra trước khi kết thúc W&B thành công; lỗi đồng bộ báo rõ, giữ bản local và không coi marker thiếu là hoàn tất.

Bảng cuối chỉ được công bố từ analysis đã xác minh, không lấy một `metrics.json` dở làm kết quả hoàn chỉnh. Không lưu trọng số, dữ liệu OCT hoặc secrets vào Git.

## 7. Sửa chỉ số mà không tốn thêm GPU

Phiên bản này dùng module báo cáo riêng cho model cuối. AP xử lý đúng các điểm số bằng nhau; ROC/PR curves dùng cùng quy tắc với chỉ số. ROC-AUC/AP dùng logit margin để tránh mất thứ tự do xác suất bão hòa. ECE và hình calibration cùng dùng xác suất lớp dương, 10 bin. Temperature và threshold được fit trên validation, không trên test; fallback temperature được ghi rõ khi fit không hợp lệ hoặc làm NLL xấu đi.

Do định nghĩa/calculation được chuẩn hóa, **không âm thầm thay các số AP/ECE cũ**. Các báo cáo mới lưu version của metric; đối chiếu raw–Bilateral cần cùng cách tính nếu có đủ logits/dự đoán cũ. Không có logits cũ thì ghi rõ khác biệt quy trình thay vì giả vờ đã tính lại.

Sau khi logits validation/test đã lưu, chỉ cần section 1–2, 10 và 13 để tạo lại báo cáo ở runtime CPU. Không cần dataset hoặc model và không gọi train. CI là bootstrap theo scan với temperature/threshold cố định, không phải patient-level CI hoặc bằng chứng về biến thiên qua seed.

Section 12 giữ phân tích information theory thành bảy bước có artifact trung gian: collection, probes, estimators, surrogates, interactions, information plane và reporting. Mặc định tắt ở run thật. Chỉ dùng tập con có giới hạn của training/validation, không dùng test để điều chỉnh estimator; các bước sau thu thập embedding không cần train lại encoder. Information plane hiện là ảnh chụp của best checkpoint, không phải quỹ đạo qua epoch. Các ước lượng MI/MIC và tương tác phải được mô tả như phân tích thăm dò, không phải bằng chứng nhân quả hay số đo thông tin tuyệt đối chính xác.

## 8. Kiểm thử và giới hạn bàn giao

Đã kiểm thử local toàn bộ notebook theo thứ tự ở chế độ synthetic CPU, lọc Bilateral thực sự trên dữ liệu nhỏ, W&B offline và Drive mô phỏng. Bộ test bao gồm metric ties, calibration, cache corruption/relocation, nhập CPU export, kiểm tra parent/raw hashes, resume, SIGINT, stage thư mục thật với W&B giả, artifact checksum và rerender không train lại.

Kiểm chứng bản triển khai: trên một Git worktree sạch, chỉ chứa các file đã commit, `python -m pytest tests -q` đạt **162 passed, 1 skipped**. Test bị bỏ qua chỉ đối chiếu mã nguồn của công cụ CPU export tùy chọn không nằm trong bản triển khai; các test nhập/kiểm tra format export bằng fixture vẫn chạy. Cả notebook cuối và notebook ba nhánh được giữ từ remote đều qua smoke CPU/offline. Ruff lint/format của các file được sửa đều đạt; có hai cảnh báo backward hook Grad-CAM trong smoke. Bộ test trong workspace làm việc có thêm test chưa commit của tác vụ khác nên số lượng khác; không dùng số lượng đó để mô tả bản GitHub.

Chưa trực tiếp xác minh file parent của bạn trên Drive, W&B online, Drive mount thật hoặc model 200³ trên GPU 80 GB. Không tuyên bố tốc độ tối đa hoặc độ chính xác phân loại được bảo đảm. Các section CPU và preflight phải xác nhận điều kiện thực tế trước khi bạn bật training.

Mã được triển khai theo một bộ hash nguồn đã review. Notebook cũ hoặc clone thiếu script mới sẽ bị từ chối trước khi chạy công việc nặng; không tự reset/pull đè thay đổi của người dùng. Dùng notebook và các helper từ cùng phiên bản GitHub, tốt nhất trên runtime mới.
