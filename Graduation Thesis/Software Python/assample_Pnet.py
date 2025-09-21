import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import numpy.random as npr

# Nếu module assemble có tồn tại, sử dụng assemble_data trong module đó.
try:
    from assemble import assemble_data
except ImportError:
    def assemble_data(output_file, anno_file_list=[]):
        """
        Kết hợp các file annotation (pos, neg, part) thành một file duy nhất với chiến lược lấy mẫu.

        Args:
            output_file (str): Đường dẫn tới file lưu kết quả.
            anno_file_list (list): Danh sách đường dẫn các file annotation.

        Returns:
            int: Tổng số mẫu (dòng) được ghi vào file output.
        """
        if not anno_file_list:
            return 0

        # Nếu file output đã tồn tại, xoá bỏ nó trước khi ghi mới.
        if os.path.exists(output_file):
            os.remove(output_file)

        total_count = 0

        for anno_file in anno_file_list:
            if not os.path.exists(anno_file):
                print(f"Warning: File annotation {anno_file} không tồn tại, bỏ qua.")
                continue

            with open(anno_file, 'r') as f:
                print(f"Processing annotation file: {anno_file}")
                anno_lines = f.readlines()

            base_num = 250000

            # Chiến lược lấy mẫu:
            if len(anno_lines) > base_num * 3:
                idx_keep = npr.choice(len(anno_lines), size=base_num * 3, replace=True)
            elif len(anno_lines) > 100000:
                idx_keep = npr.choice(len(anno_lines), size=len(anno_lines), replace=True)
            else:
                idx_keep = np.arange(len(anno_lines))
                np.random.shuffle(idx_keep)

            # Ghi các dòng được chọn vào file output
            with open(output_file, 'a+') as f_out:
                for idx in idx_keep:
                    f_out.write(anno_lines[idx])
                    total_count += 1

        return total_count

#Định nghĩa các đường dẫn cho Kaggle (có thể thay đổi theo môi trường của bạn)
pnet_positive_file = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/pos_12_train.txt"
pnet_part_file = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/part_12_train.txt"
pnet_neg_file = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/neg_12_train.txt"
imglist_filename = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/imglist_anno_12_train.txt"

# pnet_positive_file = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/pos_12_val.txt"
# pnet_part_file = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/part_12_val.txt"
# pnet_neg_file = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/neg_12_val.txt"
# imglist_filename = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/imglist_anno_12_val.txt"

if __name__ == '__main__':
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs("C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store", exist_ok=True)
    
    # Danh sách các file annotation cần kết hợp
    anno_list = [
        pnet_positive_file,
        pnet_part_file,
        pnet_neg_file
    ]
    
    # Kiểm tra sự tồn tại của các file annotation
    for anno_file in anno_list:
        if not os.path.exists(anno_file):
            print(f"Error: File annotation {anno_file} không tồn tại.")
    
    # Thực hiện kết hợp annotations
    try:
        total_samples = assemble_data(imglist_filename, anno_list)
        print(f"PNet train annotation result file path: {imglist_filename}")
        print(f"Tổng số mẫu kết hợp: {total_samples}")
    except Exception as e:
        print(f"Error during assembly: {e}")