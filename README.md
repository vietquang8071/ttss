# Gaussian–Laplacian Zero-Crossing với C và OpenMP

Dự án chuyển ảnh đầu vào sang grayscale `float32`, rồi chạy pipeline Gaussian → Laplacian → zero-crossing bằng C/OpenMP. Gaussian separable 5 tap (mặc định σ=1.0) làm mượt ảnh; Laplacian 4 láng giềng dùng zero padding; zero-crossing xét 8 láng giềng và ngưỡng contrast mặc định 10.0 để tạo edge map nhị phân. Chương trình benchmark toàn bộ pipeline tuần tự và OpenMP, đồng thời kiểm tra đáp ứng Laplacian có dấu và edge map.

## Cấu trúc thư mục

- `data/raw/`: ảnh đầu vào.
- `scripts/preprocess.py`: cắt ảnh vuông ở giữa, chuyển grayscale và tạo dữ liệu cho ba kích thước 512×512, 1024×1024, 2048×2048.
- `data/processed/`: dữ liệu `.bin` được tạo bởi bước tiền xử lý.
- `laplacian_openmp/`: mã C, Makefile và các script chạy benchmark.
- `results/laplacian/`: CSV, ảnh xem trước và kết quả benchmark.

`data/processed/` và `results/laplacian/` là dữ liệu sinh ra nên có thể không được đưa lên GitHub. Người clone repo cần chạy bước tiền xử lý trước khi benchmark.

## Yêu cầu

Hướng dẫn dưới đây dành cho Linux, dùng GCC và OpenMP. Cần có:

- GCC, Make và hỗ trợ OpenMP (trên Ubuntu/Debian cài bằng `build-essential`).
- Python 3 và `venv`.
- Thư viện Python `numpy` và `Pillow`.
- Ảnh nguồn có chiều rộng và chiều cao tối thiểu 2048 pixel. Script hỗ trợ JPEG, PNG, BMP, WebP và TIFF.

## Cài đặt sau khi clone từ GitHub

```bash
git clone <URL-repository>
cd <ten-thu-muc-repository>

sudo apt update
sudo apt install build-essential python3 python3-venv

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pillow
```

Thay `<URL-repository>` bằng URL repo GitHub và `<ten-thu-muc-repository>` bằng tên thư mục được tạo sau khi clone.

## Chuẩn bị dữ liệu

Đặt ảnh nguồn trong `data/raw/`. Nếu thư mục này không có trong bản GitHub, hãy tải hoặc chép ảnh vào đó. Mặc định script kiểm tra có đúng 20 ảnh; mỗi ảnh cần ít nhất 2048×2048 pixel.

Chạy từ thư mục gốc của repo:

```bash
python3 scripts/preprocess.py \
  --input data/raw \
  --output data/processed \
  --expected 20 \
  --preview
```

Lệnh tạo 20 file `.bin` cho mỗi kích thước trong `data/processed/512`, `data/processed/1024` và `data/processed/2048`, cùng `data/processed/metadata.csv`. Tùy chọn `--preview` tạo thêm ảnh xem trước trong `data/processed/preview/`. Nếu bộ ảnh có số lượng khác 20, thay `20` bằng số ảnh thực tế hoặc dùng `--expected 0` để bỏ kiểm tra số lượng.

## Biên dịch và chạy

### Chạy thử một ảnh

```bash
./laplacian_openmp/test_one.sh 1024 image_1.bin
```

Thay `1024` bằng `512` hoặc `2048` để chạy kích thước khác. Tên file phải khớp với tên ảnh trong `data/raw/`; ví dụ `image_1.jpg` tạo ra `image_1.bin`. Script tự biên dịch chương trình nếu cần.

Kết quả được lưu trong `results/laplacian/single/<kích-thước>/`, gồm CSV đo thời gian, đáp ứng Laplacian có dấu, edge map `.bin` và ảnh xem trước PNG.

Mỗi lần chạy tạo đáp ứng Laplacian có dấu (`*_laplacian.bin`), edge map float32 gồm `0.0`/`1.0` (`*_edges.bin`) và preview đặt ảnh grayscale cạnh edge map nhị phân. Có thể điều chỉnh tham số trực tiếp khi gọi chương trình C:

```bash
make -C laplacian_openmp
./laplacian_openmp/laplacian_benchmark data/processed/1024/image_1.bin 1024 \
  --sigma 1.2 --threshold 14 \
  --output response.bin --edges-output edges.bin
```

`--sigma` phải lớn hơn 0; `--threshold` phải không âm. Gaussian dùng clamped-edge padding; Laplacian giữ zero padding. Zero-crossing chỉ đánh dấu cặp điểm khi cả hai nằm bên trong ảnh, có dấu trái nhau và `abs(p) + abs(q) >= threshold`.

### Chạy toàn bộ benchmark

```bash
./laplacian_openmp/run_full.sh
```

Script tự biên dịch và chạy tất cả file `.bin` ở ba kích thước. Kết quả nằm trong `results/laplacian/full/`:

- `raw_results.csv`: số đo cho từng ảnh và cấu hình luồng.
- `summary.csv`: số liệu tổng hợp theo kích thước, phiên bản và số luồng.
- `previews/`: ảnh grayscale đặt cạnh edge map zero-crossing nhị phân.

Mỗi ảnh có một dòng Sequential và năm dòng OpenMP (2/4/8/12/16 luồng) trong `raw_results.csv`.

Để chỉ xuất CSV, không tạo ảnh xem trước:

```bash
EXPORT_IMAGES=0 ./laplacian_openmp/run_full.sh
```

Các script mặc định đọc dữ liệu ở `data/processed/` và ghi kết quả vào `results/laplacian/`. Có thể truyền đường dẫn khác cho script chạy toàn bộ:

```bash
./laplacian_openmp/run_full.sh /duong-dan/data/processed /duong-dan/results
```

## Lưu ý khi đưa dự án lên GitHub

- Đưa ảnh cần thiết trong `data/raw/` lên GitHub, hoặc ghi rõ nơi tải ảnh và cách đặt chúng vào thư mục đó. Không có ảnh nguồn thì người clone không thể tạo dữ liệu để chạy benchmark.
- Các file dữ liệu đã xử lý và kết quả benchmark có thể được tạo lại bằng các lệnh phía trên; `.gitignore` của repo bỏ qua `data/processed/`, `results/laplacian/` và file thực thi đã biên dịch.
