import os
import rocksdb
import tarfile
from tqdm import tqdm
from io import BytesIO
from urllib.request import urlopen

# Define the paths
tar_path_root = '/mnt/data/imagenet'
db_path_root = '/home/s4695741/imagenet'

if not os.path.exists(os.path.join(tar_path_root, "ILSVRC2012_img_train.tar")):
    print(f"{os.path.join(tar_path_root, 'ILSVRC2012_img_train.tar')} does not exist. Exit.")
if not os.path.exists(os.path.join(tar_path_root, "ILSVRC2012_img_val.tar")):
    print(f"{os.path.join(tar_path_root, 'ILSVRC2012_img_val.tar')} does not exist. Exit.")
if not os.path.exists(db_path_root):
    print(f"{db_path_root} does not exist. Exit.")

# Train set
# Open RocksDB
db_path = os.path.join(db_path_root, "train.db")
opts = rocksdb.Options()
opts.create_if_missing = True
db = rocksdb.DB(db_path, opts)

# Open ILSVRC2012_img_train.tar
tar_path = os.path.join(tar_path_root, "ILSVRC2012_img_train.tar")
with tarfile.open(tar_path, 'r') as outer_tar:
    for inner_tar in tqdm(outer_tar, total=1000, desc="Train set decompressing and saving:"):
        if inner_tar.isfile() and inner_tar.name.endswith('.tar'):
            # Batch i/o, faster
            batch = rocksdb.WriteBatch()
            
            inner_tar_name = inner_tar.name
            inner_tar_bytes = outer_tar.extractfile(inner_tar).read()
            inner_tar = tarfile.open(fileobj=BytesIO(inner_tar_bytes), mode='r')

            # class name (e.g., n01440764)
            class_name = os.path.splitext(inner_tar_name)[0]

            for image_member in inner_tar:
                if image_member.isfile() and image_member.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_data = inner_tar.extractfile(image_member).read()
                    # use "class_name/image_name" as the key in rocksdb
                    key = f"{class_name}/{image_member.name}".encode('utf-8') # key needs to be binary
                    batch.put(key, img_data)
                    print(f"Stored: {class_name}/{image_member.name}")
            
            # Batch i/o, faster
            db.write(batch)
            
            inner_tar.close()

# Test set
# Open RocksDB
db_path = os.path.join(db_path_root, "val.db")
opts = rocksdb.Options()
opts.create_if_missing = True
db = rocksdb.DB(db_path, opts)

label_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt"
print("Downloading synset labels...")
with urlopen(label_url) as f:
    synset_labels = [line.decode("utf-8").strip() for line in f]
print("Done")

# Open ILSVRC2012_img_val.tar
tar_path = os.path.join(tar_path_root, "ILSVRC2012_img_val.tar")
# Batch i/o, faster
batch = rocksdb.WriteBatch()
with tarfile.open(tar_path, 'r') as tar:
    members = [m for m in tar.getmembers() if m.isfile()]
    members.sort(key=lambda m: m.name)  # 确保顺序与 synset_labels 对齐

    assert len(members) == 50000, "Expected 50,000 images in tar"
    for i, member in enumerate(tqdm(members, desc="Val set decompressing and saving:")):
        img_data = tar.extractfile(member).read()
        class_name = synset_labels[i]
        image_name = os.path.basename(member.name)
        key = f"{class_name}/{image_name}".encode("utf-8")
        batch.put(key, img_data)
        print(f"Stored: {class_name}/{image_name}")  
# Batch i/o, faster
db.write(batch)

print(f"ImageNet saved to RocksDB at {db_path_root}!")