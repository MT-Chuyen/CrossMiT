# CrossMiT: Cross-Domain Transfer Framework for
Enhanced miRNA–Target Interaction Prediction via
Joint Learning

**Tóm tắt:** Chúng tôi cải thiện quy trình tái định vị thuốc (computational drug repositioning) bằng cách tích hợp nhiều mạng lưới tương đồng bệnh (disease similarity networks) thành các mạng $\textbf{multiplex}$ và $\textbf{multiplex-heterogeneous}$.

---

## 📂 Repo Structure  

* **`Data/`**: Chứa tất cả dữ liệu được sử dụng trong nghiên cứu.
* **`Code/`**: Chứa tất cả mã nguồn ($\textbf{source code}$) để tái tạo lại các kết quả của nghiên cứu.
 

---

## 🚀 How to Run  

### 1. Cài đặt các gói R cần thiết

Bạn cần cài đặt các gói ($\textbf{packages}$) sau trong môi trường R của mình:

* `RandomWalkRestartMH`
* `igraph`
* `foreach`
* `doParallel`
* `ROCR`
* `ggplot2`
* `Metrics`
* `hash`

### 2. Tải Repository

Tải xuống ($\textbf{Download}$) hoặc nhân bản ($\textbf{clone}$) $\textbf{repository}$ này.

### 3. Thực hiện theo hướng dẫn

Thực hiện theo các hướng dẫn chi tiết có trong thư mục **`Code`** để chạy chương trình và tái tạo các kết quả.

> **⚠️ Lưu ý về hiệu suất:** Đối với các mạng lưới lớn và phức tạp (ví dụ: mạng lưới bệnh $\textbf{multiplex}$, và mạng lưới thuốc và bệnh $\textbf{multiplex-heterogeneous}$), **khuyến nghị** nên chạy trên máy tính đa lõi ($\textbf{multi-core}$) với ít nhất **16 GB RAM**.

---

## 📚 Reference (Tham khảo)

Le, DH. Improving computational drug repositioning through multi-source disease similarity networks. Sci Rep 15, 30773 (2025).

[**DOI: 10.1038/s41598-025-04772-0**](https://doi.org/10.1038/s41598-025-04772-0)
