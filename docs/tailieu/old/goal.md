1.1 Mục đích dự án

Dự án MiniBrain được xây dựng nhằm hỗ trợ nghiên cứu và phát triển alpha cho WorldQuant Brain một cách hiệu quả hơn, giảm sự phụ thuộc vào việc submit trực tiếp lên nền tảng.

Hiện tại quá trình nghiên cứu alpha gặp một số hạn chế:

Mỗi lần kiểm tra ý tưởng đều phải submit lên Brain.
Thời gian phản hồi của simulator tương đối chậm.
Khó đánh giá hàng trăm hoặc hàng nghìn ý tưởng alpha khác nhau.
Không thể phân tích sâu nguyên nhân alpha thất bại.
Tốn nhiều thời gian thử nghiệm thủ công.

Mục tiêu cuối cùng của hệ thống là tạo ra một môi trường nghiên cứu cục bộ có khả năng:

Tự động thu thập DataFields và Operators từ WorldQuant Brain.
Sinh ra hàng nghìn alpha mới mỗi ngày bằng Genetic Programming.
Backtest cục bộ để loại bỏ các alpha chất lượng thấp.
Đánh giá Sharpe, Turnover và Fitness trước khi submit.
Xếp hạng alpha theo nhiều tiêu chí khác nhau.
Phát hiện các alpha trùng lặp hoặc quá tương đồng.
Xây dựng cơ sở dữ liệu alpha phục vụ nghiên cứu dài hạn.

Thông qua hệ thống này, quy trình nghiên cứu sẽ chuyển từ:

Ý tưởng → Sim qua API → Chờ kết quả

thành:

Ý tưởng → Sinh alpha hàng loạt → Backtest local → Chọn alpha tốt nhất → Sim

Mục tiêu dài hạn là xây dựng một "AI Quant Research Assistant" có khả năng:

Tự động khám phá các mẫu alpha mới.
Tự động kết hợp các DataFields.
Tự động tối ưu cấu trúc biểu thức.
Học từ lịch sử alpha đã thử nghiệm.
Đề xuất các alpha có xác suất vượt simulator cao nhất.

Kỳ vọng của dự án không phải là tái tạo chính xác kết quả của WorldQuant Brain, mà là tạo ra một hệ thống nghiên cứu có độ tương quan đủ cao để:

Loại bỏ phần lớn alpha kém chất lượng.
Tăng tốc quá trình nghiên cứu.
Nâng cao tỷ lệ alpha đạt simulator.
Tạo lợi thế nghiên cứu lâu dài trong các cuộc thi WorldQuant Brain.