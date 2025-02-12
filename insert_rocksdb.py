import os
import rocksdbpy

# 定义 ImageNet 训练集根目录（根据实际情况修改）
imagenet_root = '/mnt/data/imagenet/train'

# 配置 RocksDB 选项并创建数据库
db_path = '/home/s4695741/imagenet_train.db'
db = rocksdbpy.open_default(db_path)

# 遍历目录，将每张图像存入 RocksDB
# 这里 key 使用文件的相对路径，value 为图像二进制数据
for root, dirs, files in os.walk(imagenet_root):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(root, file)
            with open(file_path, 'rb') as f:
                img_data = f.read()
            # 使用相对路径作为 key（也可根据需要自定义 key）
            key = os.path.relpath(file_path, imagenet_root)
            
            # 注意 RocksDB 的 key 和 value 需要为 bytes 类型
            db.set(key.encode('utf-8'), img_data)
            print(f"Stored: {key}")

db.close()
print("ImageNet 数据存储完成！")