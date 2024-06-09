import os

def rename_files(start_dir):
    for root, dirs, files in os.walk(start_dir):
        for file in files:
            if file == 'ct_test_offset.pkl':
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, 'ct_test.pkl')
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")
            elif file == 'test_labels_final_type_1.pkl':
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, 'test_labels.pkl')
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")

# 在当前目录开始执行
rename_files('.')
