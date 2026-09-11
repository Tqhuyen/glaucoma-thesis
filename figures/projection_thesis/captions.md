# Thesis Projection Plates

Insert at 6.3 in (16.002 cm) wide without further reduction. PDF/SVG keep text as vector text; OCT images are raster.
Page size: 6.3 x 5.2 in (16.002 x 13.208 cm).
All axes and indices below refer to the source array and are zero-based. No physical pixel spacing is assumed.

## data_2404 (glaucoma)

**English.** Views of raw Harvard-GF Test volume data_2404 (glaucoma; 200 x 200 x 200 uint8). (a,b) Full-depth average-intensity (AIP) and maximum-intensity (MIP) en-face projections. (c,d) Corresponding projections of the peak-centered slab, depth pixels 25-57 inclusive on source axis 1, centered at pixel 41. (e) Selected B-scan at source axis 2, index i=122. (f) Selected orthogonal section at source axis 0, index i=140. Panels (a-d) combine depth samples, whereas (e,f) show individual sections. The footer identifies the display convention; shared processing and selection details appear in Methods Notes.

**Tiếng Việt (gợi ý).** Thể tích OCT thô Harvard-GF data_2404 (có glaucoma). (a,b) Ảnh chiếu en-face trung bình (AIP) và cực đại (MIP) toàn chiều sâu. (c,d) Các phép chiếu tương ứng trong dải pixel 25-57, gồm hai đầu, trên trục nguồn 1, tâm tại đỉnh 41. (e) B-scan được chọn: trục nguồn 2, chỉ số i=122. (f) Lát cắt trực giao: trục nguồn 0, chỉ số i=140. (a-d) gộp chiều sâu; (e,f) là lát cắt riêng. Quy ước hiển thị nằm dưới hình; phương pháp chung xem Methods Notes.

## data_3294 (glaucoma)

**English.** Views of raw Harvard-GF Test volume data_3294 (glaucoma; 200 x 200 x 200 uint8). (a,b) Full-depth average-intensity (AIP) and maximum-intensity (MIP) en-face projections. (c,d) Corresponding projections of the peak-centered slab, depth pixels 36-68 inclusive on source axis 1, centered at pixel 52. (e) Selected B-scan at source axis 2, index i=78. (f) Selected orthogonal section at source axis 0, index i=144. Panels (a-d) combine depth samples, whereas (e,f) show individual sections. The footer identifies the display convention; shared processing and selection details appear in Methods Notes.

**Tiếng Việt (gợi ý).** Thể tích OCT thô Harvard-GF data_3294 (có glaucoma). (a,b) Ảnh chiếu en-face trung bình (AIP) và cực đại (MIP) toàn chiều sâu. (c,d) Các phép chiếu tương ứng trong dải pixel 36-68, gồm hai đầu, trên trục nguồn 1, tâm tại đỉnh 52. (e) B-scan được chọn: trục nguồn 2, chỉ số i=78. (f) Lát cắt trực giao: trục nguồn 0, chỉ số i=144. (a-d) gộp chiều sâu; (e,f) là lát cắt riêng. Quy ước hiển thị nằm dưới hình; phương pháp chung xem Methods Notes.

## data_2615 (non-glaucoma)

**English.** Views of raw Harvard-GF Test volume data_2615 (non-glaucoma; 200 x 200 x 200 uint8). (a,b) Full-depth average-intensity (AIP) and maximum-intensity (MIP) en-face projections. (c,d) Corresponding projections of the peak-centered slab, depth pixels 47-79 inclusive on source axis 1, centered at pixel 63. (e) Selected B-scan at source axis 2, index i=62. (f) Selected orthogonal section at source axis 0, index i=145. Panels (a-d) combine depth samples, whereas (e,f) show individual sections. The footer identifies the display convention; shared processing and selection details appear in Methods Notes.

**Tiếng Việt (gợi ý).** Thể tích OCT thô Harvard-GF data_2615 (không glaucoma). (a,b) Ảnh chiếu en-face trung bình (AIP) và cực đại (MIP) toàn chiều sâu. (c,d) Các phép chiếu tương ứng trong dải pixel 47-79, gồm hai đầu, trên trục nguồn 1, tâm tại đỉnh 63. (e) B-scan được chọn: trục nguồn 2, chỉ số i=62. (f) Lát cắt trực giao: trục nguồn 0, chỉ số i=145. (a-d) gộp chiều sâu; (e,f) là lát cắt riêng. Quy ước hiển thị nằm dưới hình; phương pháp chung xem Methods Notes.

## Methods Notes

**English.** All plates use the same processing. Depth is the source axis with the largest standard deviation of its mean-intensity profile. Other axes retain source order. The peak-centered slab spans the global mean-profile maximum plus/minus 16 pixels, clipped to volume bounds (33 pixels in these volumes); it is not an anatomically segmented RNFL band. AIP retains floating means; MIP takes maxima. Sections maximize standard deviation after P1/P99 clipping over candidate indices 30-169 on each non-depth axis; ties retain the first index. They are not necessarily central or clinically representative. Source axes/indices are zero-based; section depth increases downward.

Each complete 200 x 200 panel uses its own linear P1/P99 display bounds, recorded in metadata; brightness is not directly comparable across panels. No denoising, retouching or spatial cropping was applied. Pixel aspect is 1:1 in array coordinates, not calibrated physical aspect. Labels come from figures/report_views_meta.json, previously checked against the Harvard-GF source CSV as supplied by the user; the CSV was not re-fetched. Non-glaucoma does not imply absence of other pathology. Source hashes, per-panel display bounds and full selection scores are in report_views_thesis_meta.json.

**Tiếng Việt.** Các hình dùng chung quy trình. Trục chiều sâu có độ lệch chuẩn lớn nhất của hồ sơ cường độ trung bình; các trục còn lại giữ thứ tự nguồn. Dải chiếu lấy đỉnh hồ sơ trung bình cộng/trừ 16 pixel, giới hạn trong thể tích (33 pixel ở đây), không phải lớp RNFL được phân đoạn giải phẫu. AIP giữ trung bình số thực; MIP lấy cực đại. Chọn lát cắt có độ lệch chuẩn lớn nhất sau giới hạn P1/P99 trong khoảng chỉ số 30-169 trên từng trục không phải chiều sâu; khi bằng nhau, lấy chỉ số đầu. Lát cắt không nhất thiết ở trung tâm hoặc đại diện lâm sàng. Trục/chỉ số nguồn bắt đầu từ 0; chiều sâu lát cắt tăng từ trên xuống.

Mỗi ô giữ đủ 200 x 200 pixel, dùng giới hạn hiển thị tuyến tính P1/P99 riêng ghi trong metadata; không so sánh trực tiếp độ sáng giữa các ô. Không khử nhiễu, chỉnh sửa hay cắt xén. Tỷ lệ pixel 1:1 theo mảng không phải tỷ lệ vật lý đã hiệu chuẩn. Nhãn từ figures/report_views_meta.json đã được đối chiếu CSV nguồn Harvard-GF trước đó theo thông tin người dùng; lần này không tải lại CSV. Không glaucoma không đồng nghĩa với không có bệnh lý khác. Hash nguồn, giới hạn hiển thị và điểm chọn lát cắt được lưu trong report_views_thesis_meta.json.

## Delivery

Drive sync: pending.
