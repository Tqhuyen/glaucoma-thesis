# Phương pháp X-AI và Information Theory cho mô hình 3D + 2×2D CrossGate

Tài liệu này giải thích **từng phương pháp** đã dùng, **vì sao phải dùng**, **nó trả lời câu hỏi gì**, **ví dụ dễ hiểu**, và **hình ảnh nào nên gắn với phương pháp nào**, kèm **kết quả ở hình đó cho thấy điều gì**. Cuối cùng là bảng khuyến nghị giữ/bỏ ảnh.

Mẫu phân tích (toàn bộ hình trong tài liệu): **Validation row 0 = scan `data_2101`, label = 1 (glaucoma positive)**, mô hình bilateral seed 42, dataset revision `939a38876b7b9313162842ef2d44b7edc2b57020`. Hình nằm ở `figures/xai/final_bilateral_s42/`.

---

## 0. Vì sao cần hai nhóm công cụ

Mô hình là một "hộp đen": nó cho xác suất, nhưng không tự nói **nó nhìn vào đâu** và **nó dùng thông tin nào**. Hai câu hỏi khác nhau cần hai nhóm công cụ:

- **X-AI (explainable AI)** trả lời *Ở đâu / tại sao mẫu này*: mô hình dựa vào vùng nào của ảnh, nhánh nào, view nào. Kết quả là **bản đồ saliency/heatmap** và **mức quan trọng của nhánh**. Đây là mức **một mẫu** (local).
- **Information Theory** trả lời *Bao nhiêu thông tin / các nhánh bổ sung hay trùng nhau*: mỗi biểu diễn (3D, 2D-0, 2D-1, fused) chứa bao nhiêu thông tin về nhãn, và các nhánh kết hợp kiểu redundancy (trùng) hay synergy (bổ sung). Đây là mức **toàn tập** (population).

Ví dụ đời thường: X-AI giống như **chiếu đèn vào ảnh** xem bác sĩ (mô hình) đang nhìn vào đâu; Information Theory giống như **đo hàm lượng thông tin** trong từng nguồn tin và xem hai nguồn có nói cùng một chuyện hay ghép lại thành thông tin mới.

---

## 1. Nhóm X-AI

### 1.1. Grad-CAM — nền tảng chung

**Vì sao dùng.** Là phương pháp "nhìn vào đâu" phổ biến và rẻ nhất cho CNN. Nó biến đổi gradient của class score theo feature map thành một bản đồ nhiệt: chỗ nào đóng góp mạnh vào lớp dự đoán thì nóng.

**Giải thích dễ hiểu.** Tưởng tượng bạn hỏi: "Nếu thay đổi nhẹ đặc trưng ở ô này thì điểm của lớp bệnh đổi bao nhiêu?" Grad-CAM lấy trung bình gradient theo kênh để có **trọng số tầm quan trọng**, nhân với **activation** thực tế, rồi ReLU để giữ chỗ làm tăng điểm. Kết quả là bản đồ cùng kích thước ảnh, chuẩn hoá về [0, 1].

**Trong repo.** `scripts/final_model.py:370` — `grad_cam(module, model, x3d, views, target, is_3d)`:
1. gắn hook lấy activation + gradient ở module cuối (`_ActHook`);
2. backward cho logit của lớp mục tiêu;
3. `w = grad.mean(over spatial)`; `cam = ReLU(sum(w * act))`;
4. upsample về kích thước input; chuẩn hoá min–max.

**Lưu ý quan trọng.** Grad-CAM chỉ là tương quan cục bộ theo gradient, **không phải nhân quả**; vùng nóng có thể là biên ảnh, feature-map thô hoặc artefact upsampling. Vì vậy nó phải đi kèm bằng chứng định lượng.

### 1.2. Grad-CAM 3D (nhánh ResNeXt3D)

**Vì sao dùng.** Nhánh 3D là nguồn thông tin mạnh (probe AUC 0.8381). Cần biết mô hình 3D "nhìn" vào dải lớp nào của khối OCT.

**Giải thích dễ hiểu.** Giống 1.1 nhưng trên khối 200³ (model input 96³). Bản đồ 3D hiển thị ở lát giữa; vùng nóng bám cấu trúc thể tích.

**Ví dụ.** Nếu bản đồ nóng đúng dải lớp võng mạc trong (nơi lớp sợi thần kinh/ganglion nằm) thì đó là tín hiệu hợp lý về bệnh tăng nhãn áp.

**Hình.** `03_gradcam_3d_cam-only_sample-val0-data2101-lbl1.png` (CAM-only) và `13_gradcam_3d_overlay_sample-val0-data2101-lbl1.png` (overlay lên raw + bilateral).

**Kết quả cho thấy gì.** Hai cụm nóng gọn, bám theo hai cung dải lớp võng mạc (trên-phải và dưới-trái) — **đây là kết quả hợp lý nhất về giải phẫu** trong tất cả hình.

### 1.3. Grad-CAM 2D theo từng view (branch 0 = slab-mip, branch 1 = aip-full)

**Vì sao dùng.** Mô hình có 2 nhánh 2D; cần biết từng view đóng góp vào vùng nào để đối chiếu với probe từng nhánh.

**Giải thích dễ hiểu.** Mỗi view là một ảnh 2D (slab MIP = chiếu độ dày lớp trong; AIP full = chiếu toàn bộ chiều sâu). Grad-CAM 2D cho biết mô hình nhìn vào vùng nào của từng ảnh.

**Ví dụ.** Với ảnh AIP có đĩa thị rõ, nếu vùng nóng tập trung quanh đĩa thị thì mô hình dùng đĩa thị; nếu nóng rải rác toàn võng mạc thì mô hình dùng texture nền.

**Hình.** branch 0: `01_...cam-only` và `11_...overlay`; branch 1: `02_...cam-only` và `12_...overlay`.

**Kết quả cho thấy gì.**
- **Branch 0 (slab-mip):** khối nóng lớn, khuếch tán theo vùng sáng rộng của slab, không khu trú vào cấu trúc nhỏ. Đi kèm probe AUC 2D-0 chỉ **0.5188** (gần ngẫu nhiên) → nhánh này khó là nguồn thông tin chính.
- **Branch 1 (aip-full):** vùng nóng lan rộng và có **"lỗ" tối đúng vùng đĩa thị**; mô hình không dựa vào tâm đĩa thị ở view này mà dùng vùng võng mạc quanh đó (đây là **giả thuyết**, cần kiểm chứng, vì probe 2D-1 mạnh 0.8381).

### 1.4. Occlusion sensitivity (3D)

**Vì sao dùng.** Kiểm tra "che đi thì mất gì" — trực tiếp hơn Grad-CAM vì đo **thay đổi xác suất thật**, không qua gradient.

**Giải thích dễ hiểu.** Chia khối 3D thành lưới `n×n×n`. Lần lượt che (đặt 0) từng ô rồi đo xác suất lớp đúng giảm bao nhiêu. Ô nào che vào làm xác suất giảm mạnh là quan trọng. Giống như bịt từng vùng trên ảnh và xem bác sĩ có đổi ý không.

**Ví dụ.** Nếu che vùng dải lớp võng mạc làm xác suất rớt mạnh, còn che góc ảnh không đổi, thì mô hình thực sự dựa vào dải lớp.

**Trong repo.** `scripts/final_model.py:390` — lưới `n` (mặc định 6, XAI thật dùng `n=4` → **4×4×4**), fill = 0.

**Hình.** `05_occlusion_4x4x4_cam-only_...` và `15_occlusion_4x4x4_overlay_...`.

**Kết quả cho thấy gì.** Lưới 4×4×4 trên khối 96³ quá thô: kết quả là các khối vuông lớn, vùng sáng nằm ở góc dưới-trái. **Độ phân giải quá thấp để kết luận cấu trúc**; chỉ nên xem như định hướng, cần tăng `n` (ví dụ 8 hoặc 12) hoặc dùng occlusion theo dải lớp.

### 1.5. Integrated Gradients (IG)

**Vì sao dùng.** IG có nền tảng lý thuyết tốt hơn gradient thô: nó tích phân gradient dọc đường đi từ ảnh "nền" (đen) tới ảnh thật, nên giảm hiện tượng gradient nhiễu và có tính "completeness".

**Giải thích dễ hiểu.** Từ từ "hiện hình" ảnh từ đen đến ảnh thật theo `steps` bước; ở mỗi bước cộng dồn gradient. Điểm ảnh nào đóng góp dương đều đặn dọc đường đi thì quan trọng. Giống như mở dần độ sáng đèn và ghi lại chỗ nào làm mô hình đổi ý.

**Trong repo.** `scripts/final_model.py:407` — baseline = 0, `steps=16`; trả `|attr|`.

**Hình.** `04_integrated-gradients_cam-only_...` và `14_integrated-gradients_overlay_...`.

**Kết quả cho thấy gì.** Bản đồ nổi bật là **một đường thẳng đứng ở rìa trái** — dấu hiệu **artefact biên/padding**, không phải cấu trúc giải phẫu. **IG trong lần này không dùng được để diễn giải.**

### 1.6. CrossGate attention (attention của fusion)

**Vì sao dùng.** CrossGate là cơ chế fusion: 3D làm query, các token 2D làm key/value; trọng số attention cho biết fusion "chú ý" vào token nào. Đây là bằng chứng **bên trong kiến trúc**, không phải saliency ảnh.

**Giải thích dễ hiểu.** Giống như trong cuộc họp, nhánh 3D hỏi và tự quyết định lắng nghe nhánh 2D nào nhiều hơn. Attention 0.6/0.4 nghĩa là 3D "nghe" 2D-0 nhiều hơn.

**Trong repo.** `scripts/final_model.py:421` — đọc `model.fusion.last_w` và `last_gate`.

**Hình / log.** **Không có hình ảnh**; giá trị được log dạng scalar: `xai/gate=0.616`, `xai/attention_2D-0=0.595`, `xai/attention_2D-1=0.405`.

**Kết quả cho thấy gì.** Attention nghiêng về 2D-0, **nhưng** attention chỉ là trọng số mềm của mô hình, **không đảm bảo là nhân quả**.

### 1.7. Branch drop / leave-one-branch-out (ablation tại inference)

**Vì sao dùng.** Để trả lời dứt khoát "bỏ nhánh nào thì mô hình mất gì" — đây là phép đo **nhân quả ở mức dự đoán**, bổ trợ/đối chiếu cho attention.

**Giải thích dễ hiểu.** Lần lượt "rút phích" từng nhánh: đặt input nhánh đó về 0 rồi đo xác suất lớp đúng giảm bao nhiêu. Giảm nhiều = nhánh quan trọng.

**Trong repo.** `scripts/final_model.py:429` — zero toàn bộ 2D, rồi zero từng view; `names[0]` bị đổi thành `all_2d_inputs`.

**Hình / log.** **Không có hình ảnh**; giá trị: bỏ 2D-0 giảm **0.1052**, bỏ 2D-1 giảm **~0.000**, bỏ toàn bộ 2D giảm **0.0497**.

**Kết quả cho thấy gì.** **Mâu thuẫn có ích:** attention nói 2D-0 quan trọng nhất, nhưng branch-drop cho thấy bỏ 2D-1 gần như không đổi, bỏ 2D-0 mới giảm. Kết luận: **dùng branch-drop/probe để xét nhánh, không dùng attention đơn thuần.**

### 1.8. Overlay reconstruction (bước tái dựng để xem bằng mắt)

**Vì sao dùng.** Ảnh X-AI gốc là CAM-only, không có nền OCT, nên không thể biết vùng nóng có trùng cấu trúc không. Bước này ghép heatmap lên ảnh thật để kiểm tra bằng mắt.

**Giải thích dễ hiểu.** Lấy ảnh gốc, tô heatmap lên với độ trong suốt (alpha), viền contour top-20%. Input bilateral được tạo bằng `skimage.restoration.denoise_bilateral` (sigma_color=0.1, sigma_spatial=4) để tái lập đúng ảnh mô hình nhận.

**Trong repo / cấu hình.** Run `final_bilateral_s42_xai_overlay_reconstruction` (id `l4yahq6c`): `sample="Validation row 0"`, `label=1`, `overlay_alpha=0.42`, `dataset_revision=939a388…`, `heatmap_recovery="Recovered from rendered W&B PNG axes; approximate"`, `reconstruction_exact_preprocess=True`.

**Giới hạn.** **Heatmap phục hồi từ PNG (có sai số oxy hoá/vị trí), không phải tensor saliency gốc.** Vì vậy overlay chỉ để định hướng; tensor gốc cần được lưu (`scripts/xai_identity.py` đã làm việc này).

**Hình.** `00_contact-sheet_all-methods_overlay_...` (tổng hợp) + `11..15` (từng phương pháp).

---

## 2. Nhóm Information Theory

Điểm chung: dùng **embedding** (vector đặc trưng) thay vì ảnh. Với mô hình: `z` = fused, `e3d` = nhánh 3D, `e2d[:,0]`, `e2d[:,1]` = hai view 2D. Thu thập bằng `collect_embeddings` (`scripts/information_theory.py:477`).

Tất cả phương pháp nhóm này **không có hình ảnh**; theo quy ước của repo, chúng log bằng **bảng/số** (`report/probe_table`, `report/mi_estimators_table`, `report/surrogate_table`, `report/interaction_table`, `report/plane_table` + summary `entropy/*`, `mi/*`, `surrogate/*`, `interaction/*`), không vẽ đồ thị metrics.

### 2.1. Label entropy H(Y)

**Vì sao dùng.** Cần biết "trần" thông tin: nhãn vốn đã bất định bao nhiêu, để chuẩn hoá MI (NMI = MI / H(Y)).

**Giải thích dễ hiểu.** Nếu tỉ lệ hai lớp 50/50, bất định = 1 bit; nếu lệch mạnh, bất định thấp hơn. Ví dụ: nhãn cân bằng ~0.69 nats.

**Trong repo.** `label_entropy` (`:268`).

### 2.2. Linear probe + probe-based ablation

**Vì sao dùng.** Trả lời: "Chỉ với embedding của một nhánh, một bộ phân loại **đơn giản** (tuyến tính) đọc được bao nhiêu thông tin về nhãn?" Đây là cách đo khả năng biểu diễn tách biệt khỏi đầu phân loại phức tạp.

**Giải thích dễ hiểu.** Giống kiểm tra "trong đầu mô hình đã có sẵn câu trả lời chưa": nếu chỉ cần vẽ một đường thẳng đã phân loại tốt thì thông tin nằm rõ trong embedding.

**Trong repo.** `linear_probe` (`:357`, logistic hoặc MLP) và `probe_auc_cv` (`:403`). Ablation mức representation: từng nhánh, cặp 3D+view, concat tất cả, `z_fused`.

**Hình / log.** Không hình; `report/probe_table`.

**Kết quả cho thấy gì.** Probe AUC: 3D **0.8381**, 2D-1 **0.8381**, 2D-0 **0.5188**, fused **0.8640**, concat **0.8372**. → **3D và 2D-1 là nguồn thông tin chính; 2D-0 gần ngẫu nhiên; fusion học tốt hơn concat thô.**

### 2.3. MI estimators: DV, NWJ, InfoNCE (MINE)

**Vì sao dùng.** MI lý thuyết `I(X;Y)` không tính được trực tiếp với biến liên tục nhiều chiều; phải **xấp xỉ** bằng mạng neural (MINE) với các lower bound khác nhau. Dùng nhiều estimator để tránh kết luận phụ thuộc một phương pháp.

**Giải thích dễ hiểu.** MI = "biết X thì giảm được bao nhiêu bất định về Y". Ba bound là ba cách ước lượng khác nhau:
- **DV (Donsker–Varadhan):** dùng log-mean-exp trên cặp âm (marginal) → bound chặt nhưng phương sai cao.
- **NWJ (Nguyen–Wainwright–Yu):** dùng `exp(marginal-1)` → ổn định hơn nhưng có thể âm/nhiễu.
- **InfoNCE:** bài toán contrastive (positive vs negative) → giống "phân loại cặp đúng/sai".

**Ví dụ.** Nếu `I(z; y) ≈ 0` thì embedding gần như không liên quan nhãn; nếu dương lớn thì embedding chứa thông tin về bệnh.

**Trong repo.** `mine_mi` (`:117`), `info_nce_mi` (`:177`), `estimate_mi` (`:235` — chạy **nhiều seed**, báo median/mean/std/min/max và **negative_rate**). **Không clamp giá trị âm**: âm nghĩa là estimator chưa hội tụ/bias, không phải "MI âm".

**Hình / log.** Không hình; `report/mi_estimators_table`.

**Kết quả cho thấy gì.** Các ước lượng **chưa ổn định** (chỉ 5 seed, nhiều giá trị âm, negative-rate cao ở 2D-0 và concat) → **chỉ đọc thứ hạng và mức nhất quán với probe/MIC, không diễn giải trị tuyệt đối.**

### 2.4. KSG (k-NN MI)

**Vì sao dùng.** Một ước lượng MI **phi tham số, không cần huấn luyện** (`ksg_mi`, `:274`), dùng cho `dependence` trong interaction và `conditional_dependence` theo lớp. Bổ trợ MINE vì không có phương sai do tối ưu.

**Giải thích dễ hiểu.** Đếm hàng xóm gần nhất trong không gian chung so với hai không gian riêng; nếu hai biến gắn nhau, lân cận trùng nhau nhiều → MI tăng.

### 2.5. MIC (maximal information coefficient)

**Vì sao dùng.** MI/MINE chỉ nhạy với quan hệ mà mạng học được; **MIC phát hiện quan hệ phi tuyến bất kỳ** bằng cách quét nhiều cách chia bin. Dùng `max_mic_features` (`:348`) trên PCA của embedding để chống nhiễu chiều cao.

**Giải thích dễ hiểu.** Chia dữ liệu thành lưới theo nhiều kích thước, mỗi lưới tính MI chuẩn hoá, rồi lấy lưới tốt nhất — giống "thử nhiều cách vẽ lưới để tìm quy luật ẩn".

**Ví dụ.** Hai biến có quan hệ hình chữ U (không tuyến tính) có thể có Pearson ~0 nhưng MIC cao.

**Hình / log.** Không hình; giá trị `max_mic` trong `report/mi_estimators_table`.

**Kết quả cho thấy gì.** MIC fused **0.4020** (lớn nhất trong các representation), hỗ trợ kết luận fused chứa nhiều thông tin phi tuyến về nhãn nhất.

### 2.6. Surrogate permutation test (kiểm định p-value)

**Vì sao dùng.** Mọi ước lượng MI/MIC đều có thể dương do ngẫu nhiên. Cần **kiểm định thống kê**: so giá trị thật với phân phối null tạo bằng **hoán vị nhãn**.

**Giải thích dễ hiểu.** Xáo trộn ngẫu nhiên nhãn nhiều lần để xem "nếu không có quan hệ thật thì ước lượng đạt mức nào", rồi xem giá trị thật vượt null bao xa.

**Trong repo.** `surrogate_test` (`:519`): `p = (1 + #{null ≥ real}) / (n+1)`, z-score, ngưỡng **Bonferroni = 0.05/n_tests**. MIC dùng `IT_SURROGATES` (≥1000), MINE dùng `IT_MINE_SURROGATES` (ít hơn vì đắt).

**Hình / log.** Không hình; `report/surrogate_table`.

**Kết quả cho thấy gì.** fused/3D/2D-1 đều permutation **p=0.001** (đạt ý nghĩa); 2D-0 **p=0.035**, **không qua Bonferroni 0.00357** → 2D-0 không đủ bằng chứng.

### 2.7. Joint/conditional MI và Interaction Information (redundancy vs synergy)

**Vì sao dùng.** Biết hai nhánh đều chứa thông tin chưa đủ; còn phải biết **chúng trùng nhau hay bổ sung cho nhau**. Điều này quyết định có cần đa nhánh hay không.

**Giải thích dễ hiểu.** Interaction Information `II = I(Z1;Z2) − I(Z1;Z2|Y)`:
- **II > 0 (redundancy):** hai nhánh nói cùng một chuyện (thừa).
- **II < 0 (synergy):** biết cả hai mới có thông tin mà từng nhánh riêng không có (bổ sung).

**Ví dụ.** Hai ảnh chụp cùng một góc mắt → redundancy. Một ảnh 3D cấu trúc + một ảnh 2D mạch máu → có thể synergy.

**Trong repo.** `conditional_mi` (`:424`), `conditional_dependence` (`:436`), `interaction_information` (`:450`).

**Hình / log.** Không hình; `report/interaction_table`.

**Kết quả cho thấy gì.** `interaction` dương mạnh nhất ở cặp **3D+2D-1 (0.1634)**, rồi 3D+2D-0 (0.0982), 2D-0+2D-1 (0.0844). Dương nghĩa là **redundancy**, nhưng giá trị dựa trên estimator nhiễu → chỉ là **chỉ báo khám phá, chưa phải PID đầy đủ**.

### 2.8. Information plane I(X;Z) vs I(Z;Y)

**Vì sao dùng.** Xem **động lực học representation theo epoch**: embedding nén input (I(X;Z) giảm) mà vẫn giữ thông tin nhãn (I(Z;Y) cao) — ý tưởng information bottleneck.

**Giải thích dễ hiểu.** Vẽ đường cong theo epoch: trục tung là thông tin về nhãn, trục hoành là thông tin về input. Mô hình tốt thường giữ nhãn và giảm dần phụ thuộc input.

**Trong repo.** `information_plane` (`:536`), dùng embedding thu mỗi epoch khi bật `RUN_INFO`.

**Hình / log.** Không hình; `report/plane_table`.

**Kết quả cho thấy gì.** Hiện chỉ có **best checkpoint epoch 1**, chưa thu embedding qua nhiều epoch → **chưa thể kết luận động lực nén/mở rộng**.

### 2.9. Estimator validation (XOR / independent)

**Vì sao dùng.** Trước khi tin MI, phải kiểm tra estimator tự nó đúng: trên dữ liệu **XOR** (MI lý thuyết = ln2 ≈ 0.693) và hai biến **độc lập** (MI ≈ 0).

**Trong repo.** `validate_estimators` (`:574`).

**Hình / log.** Không hình; `estimator/xor_mi`, `estimator/independent_mi`.

---

## 3. Bảng: phương pháp nào dùng hình nào, giữ hay bỏ, và kết quả cho thấy gì

Tên hình rút gọn (đầy đủ xem Phụ lục). Tất cả đều của mẫu `data_2101`.

| Phương pháp | Hình nên gắn | Giữ/Bỏ | Vì sao | Kết quả cho thấy gì |
|---|---|---|---|---|
| Tổng hợp | `00_contact-sheet_...` | **Giữ** | Cái nhìn 5 phương pháp cạnh nhau, dễ so sánh | 3D bám dải lớp; 2D khuếch tán; IG là đường biên; occlusion thô |
| Grad-CAM 3D | `03_...cam-only` + `13_...overlay` | **Giữ (chính)** | Hợp lý nhất về giải phẫu, có overlay để đối chiếu | Hai cụm nóng bám cung dải lớp võng mạc |
| Grad-CAM 2D branch 0 (slab-mip) | `01_...cam-only` + `11_...overlay` | **Giữ (phụ)** | Cho thấy nhánh yếu nhìn vào đâu | Khối nóng lớn khuếch tán; khớp probe 2D-0 ≈ 0.52 |
| Grad-CAM 2D branch 1 (aip-full) | `02_...cam-only` + `12_...overlay` | **Giữ (phụ)** | Nhánh mạnh (0.8381) nhưng CAM lan rộng | Có "lỗ" tối ở đĩa thị → giả thuyết dùng võng mạc quanh đĩa |
| Occlusion 4×4×4 | `05_...cam-only` + `15_...overlay` | **Giữ tạm / nên hạn chế** | Lưới quá thô, dễ gây hiểu nhầm | Vùng nóng ở góc dưới-trái; không kết luận cấu trúc |
| Integrated Gradients | `04_...cam-only` + `14_...overlay` | **Nên bỏ khỏi phần kết quả chính** | Là artefact biên, không phải cấu trúc | Một đường thẳng đứng rìa trái |
| CrossGate attention | (không hình) | **Không có hình** | Là scalar nội bộ, không phải bản đồ ảnh | 2D-0/2D-1 = 0.595/0.405; gate 0.616 |
| Branch drop | (không hình) | **Không có hình** | Là scalar ablation | Bỏ 2D-0 −0.105, 2D-1 ~0, cả 2D −0.0497 |
| Linear probe (IT) | (không hình) | **Không có hình** | Quy ước: metrics log bảng/số | 3D 0.8381, 2D-1 0.8381, 2D-0 0.5188, fused 0.8640 |
| DV/NWJ/InfoNCE, MIC | (không hình) | **Không có hình** | Estimator nhiều seed, log bảng | MIC fused 0.4020; estimator chưa ổn định |
| Surrogate | (không hình) | **Không có hình** | Kiểm định p-value, log bảng | fused/3D/2D-1 p=0.001; 2D-0 p=0.035 |
| Interaction (II) | (không hình) | **Không có hình** | Proxy redundancy/synergy | 3D+2D-1 = 0.1634 (redundancy) |
| Information plane | (không hình) | **Không có hình** | Chỉ epoch 1, chưa đủ | Chưa kết luận động lực |

**Tóm tắt giữ/bỏ:**
- **Giữ và đưa vào kết quả chính:** `00`, `03`/`13` (Grad-CAM 3D), `11`/`12` (Grad-CAM 2D đối chiếu).
- **Giữ để minh hoạ nhưng ghi rõ hạn chế:** `05`/`15` (occlusion thô), `01`/`02` (CAM-only khi cần so sánh vùng).
- **Bỏ khỏi kết quả chính:** `04`/`14` (IG artefact) — chỉ nên đưa vào phần "hạn chế/phương pháp chưa hoạt động".
- **Không có hình (đúng quy ước):** toàn bộ Information Theory + CrossGate attention + branch drop.

---

## 4. Nhận xét tổng hợp

1. **Bằng chứng X-AI và IT hội tụ về cùng một kết luận:** 3D và 2D-1 là nguồn thông tin chính; 2D-0 yếu (probe 0.5188, surrogate không qua Bonferroni); fused tốt hơn concat (probe 0.8640 vs 0.8372; MIC 0.4020 cao nhất).
2. **Grad-CAM 3D là hình thuyết phục nhất**, bám dải lớp võng mạc. Đây là hình nên đưa lên phần kết quả X-AI chính.
3. **Attention có thể gây hiểu nhầm:** attention 2D-0 cao hơn, nhưng branch-drop cho thấy bỏ 2D-0 mới giảm xác suất. Luôn đối chiếu attention với ablation.
4. **IG và occlusion hiện chưa đủ chất lượng** để diễn giải (IG artefact biên; occlusion lưới thô). Nên loại khỏi kết luận hoặc đưa vào phần giới hạn phương pháp.
5. **Chưa thể kết luận lâm sàng:** một mẫu dương tính duy nhất, heatmap phục hồi từ PNG (không phải tensor gốc), thiếu TP/TN/FP/FN và các kiểm định định lượng. Vùng nóng có thể là cấu trúc, cũng có thể là biên/artefact.

## 5. Khuyến nghị để nhận xét đáng tin hơn

- Lưu **tensor saliency gốc + danh tính mẫu** bằng `scripts/xai_identity.py` (đã có, kèm test).
- Mở rộng cohort ≥12–20 ca gồm TP/TN/FP/FN, cả hai lớp; lấy thêm lát cắt có saliency cực đại cho 3D.
- Thêm kiểm định định lượng: deletion/insertion curve, randomization test, stability khi nhiễu/flip, tỉ lệ saliency ở viền, đồng thuận Grad-CAM ↔ IG ↔ occlusion.
- Tăng độ phân giải occlusion (n=8–12) hoặc occlusion theo dải lớp; tăng `steps` cho IG và dùng baseline phù hợp hơn (ví dụ ảnh bilateral trung bình) để tránh artefact biên.
- Chuẩn hoá color scale/overlay alpha giữa các ca.

---

## Phụ lục A. Danh sách ảnh đã đổi tên

Thư mục gốc: `figures/xai/final_bilateral_s42/`.

**Run `final_bilateral_s42_xai` (CAM-only, id `iow9ispv`)**

| Tên mới | Thành phần |
|---|---|
| `01_gradcam_2d-branch0-slab-mip_cam-only_sample-val0-data2101-lbl1.png` | Grad-CAM 2D, nhánh 0 (slab-mip), CAM-only |
| `02_gradcam_2d-branch1-aip-full_cam-only_sample-val0-data2101-lbl1.png` | Grad-CAM 2D, nhánh 1 (aip-full), CAM-only |
| `03_gradcam_3d_cam-only_sample-val0-data2101-lbl1.png` | Grad-CAM 3D, CAM-only |
| `04_integrated-gradients_cam-only_sample-val0-data2101-lbl1.png` | Integrated Gradients, CAM-only |
| `05_occlusion_4x4x4_cam-only_sample-val0-data2101-lbl1.png` | Occlusion 4×4×4, CAM-only |

**Run `final_bilateral_s42_xai_overlay_reconstruction` (id `l4yahq6c`)**

| Tên mới | Thành phần |
|---|---|
| `00_contact-sheet_all-methods_overlay_sample-val0-data2101-lbl1.png` | Contact sheet 5 phương pháp + overlay |
| `11_gradcam_2d-branch0-slab-mip_overlay_sample-val0-data2101-lbl1.png` | Grad-CAM 2D nhánh 0 overlay |
| `12_gradcam_2d-branch1-aip-full_overlay_sample-val0-data2101-lbl1.png` | Grad-CAM 2D nhánh 1 overlay |
| `13_gradcam_3d_overlay_sample-val0-data2101-lbl1.png` | Grad-CAM 3D overlay |
| `14_integrated-gradients_overlay_sample-val0-data2101-lbl1.png` | Integrated Gradients overlay |
| `15_occlusion_4x4x4_overlay_sample-val0-data2101-lbl1.png` | Occlusion 4×4×4 overlay |

Bảng đối chiếu tên gốc → tên mới: `figures/xai/final_bilateral_s42/names.json`. Manifest (id, config, summary, sha256): `figures/xai/final_bilateral_s42/manifest.json`.

## Phụ lục B. Danh tính mẫu và tái lập

- Mẫu: `Validation row 0` → `Validation_idx0_data_2101_label1`.
- Quy tắc index: Training `data_0001–2100`, Validation `data_2101–2400`, Test `data_2401–3300` (đã kiểm chứng khớp 100% nhãn).
- `scripts/xai_identity.py` — `resolve_stem`, `sample_identity`, `save_xai_identified` (lưu tensor + meta + tên file có danh tính).
- `scripts/fetch_wandb_xai.py` — tải lại ảnh W&B: `python scripts/fetch_wandb_xai.py --run final_bilateral_s42_xai --run final_bilateral_s42_xai_overlay_reconstruction --out figures/xai/final_bilateral_s42`.
