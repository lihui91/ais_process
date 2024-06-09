import os

def clear_directory(start_dir):
    files_to_delete = []
    current_file_name = os.path.basename(__file__)  # 获取当前脚本文件名

    for root, dirs, files in os.walk(start_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file != current_file_name:  # 检查文件名是否等于当前脚本文件名
                files_to_delete.append(file_path)

    for file_path in files_to_delete:
        os.remove(file_path)
        print(f"Removed: {file_path}")

# 在当前目录开始执行
clear_directory('.')
