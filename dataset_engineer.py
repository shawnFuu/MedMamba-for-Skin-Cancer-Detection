import os
import shutil
import pandas as pd

def process_metadata(metadata_paths, object_datapath):
    combined_df = pd.DataFrame()

    for metadata_path in metadata_paths:
        # 读取metadata文件
        df = pd.read_csv(metadata_path, low_memory=False)

        # 筛选target为0或0.0的数据项
        filtered_df = df[(df['target'] == 0) | (df['target'] == 0.0)]
        isic_ids = filtered_df['isic_id'].values
        print(f"{len(isic_ids)}")

        # 获取metadata文件所在目录
        metadata_dir = os.path.dirname(metadata_path)

        # 确保目标图像目录存在
        target_image_dir = os.path.join(object_datapath, 'train-image', 'image')
        # os.makedirs(target_image_dir, exist_ok=True)

        for isic_id in isic_ids:
            # 构造图像文件路径
            img_name = os.path.join(metadata_dir, 'train-image', 'image', f'{isic_id}.jpg')
            downsampled_img_name = os.path.join(metadata_dir, 'train-image', 'image', f'{isic_id}_downsampled.jpg')

            # 检查图像文件是否存在
            if os.path.isfile(img_name):
                shutil.copy(img_name, target_image_dir)
                print(f"{img_name} copied!")
            elif os.path.isfile(downsampled_img_name):
                shutil.copy(downsampled_img_name, target_image_dir)
                print(f"{downsampled_img_name} copied!")
            else:
                print(f"{img_name} or {downsampled_img_name} not found!")

        # 合并metadata
        combined_df = pd.concat([combined_df, filtered_df], ignore_index=True)

    # 保存合并后的metadata
    combined_metadata_path = os.path.join(object_datapath, 'train-metadata.csv')
    if os.path.isfile(combined_metadata_path):
        existing_df = pd.read_csv(combined_metadata_path, low_memory=False)
        # 对齐列
        combined_df = combined_df.reindex(columns=existing_df.columns, fill_value=pd.NA)
        combined_df = pd.concat([existing_df, combined_df], ignore_index=True)
    else:
        # 如果文件不存在，直接使用combined_df
        combined_df.to_csv(combined_metadata_path, index=False)
        return

    combined_df.to_csv(combined_metadata_path, index=False)


from sklearn.model_selection import train_test_split


def create_test_set(object_datapath):
    # 读取元数据文件
    metadata_path = os.path.join(object_datapath, 'train-metadata.csv')
    metadata = pd.read_csv(metadata_path)

    # 从元数据中筛选出1/10作为测试集
    train_metadata, test_metadata = train_test_split(metadata, test_size=0.1, random_state=42)

    # 创建测试集图片目录
    test_image_dir = os.path.join(object_datapath, 'test-image0805/image')
    os.makedirs(test_image_dir, exist_ok=True)

    # 测试集元数据文件路径
    test_metadata_path = os.path.join(object_datapath, 'test-metadata0805.csv')

    # 遍历测试集元数据并复制图片和元数据
    for _, row in test_metadata.iterrows():
        isic_id = row['isic_id']
        found_image = False

        # 查询图片文件并复制到测试集目录
        for suffix in ['', '_downsampled']:
            image_name = f"{isic_id}{suffix}.jpg"
            image_path = os.path.join(object_datapath, 'train-image/image', image_name)
            if os.path.exists(image_path):
                shutil.copy(image_path, test_image_dir)
                print(f"{image_path} copied!")
                # 删除原始图片
                os.remove(image_path)
                found_image = True
                break

        if found_image:
            # 将当前数据项的元数据添加到测试集元数据文件
            row.to_frame().T.to_csv(test_metadata_path, mode='a', header=not os.path.exists(test_metadata_path),
                                    index=False)

    # 将剩余的训练集元数据保存到原始metadata文件
    train_metadata.to_csv(metadata_path, index=False)


def clean_metadata(object_datapath):
    # 读取元数据文件
    metadata_path = os.path.join(object_datapath, 'train-metadata.csv')
    metadata = pd.read_csv(metadata_path,low_memory=False)

    # 定义图片目录
    image_dir = os.path.join(object_datapath, 'data/fuxiaowen/isic-2024-challenge/train-image/image')

    # 记录需要保留的元数据项
    valid_metadata = []

    for _, row in metadata.iterrows():
        isic_id = row['isic_id']
        # 查询图片文件
        image_exists = False
        for suffix in ['', '_downsampled']:
            image_name = f"{isic_id}{suffix}.jpg"
            image_path = os.path.join(image_dir, image_name)
            if os.path.exists(image_path):
                image_exists = True
                break

        if image_exists:
            valid_metadata.append(row)

    # 创建新的DataFrame
    valid_metadata_df = pd.DataFrame(valid_metadata)

    # 将有效的元数据保存回原始元数据文件
    valid_metadata_df.to_csv(metadata_path, index=False)


# 示例调用
metadata_paths = [
    '/data/fuxiaowen/isic-2018-challenge/train-metadata.csv',
    '/data/fuxiaowen/isic-2019-challenge/train-metadata.csv',
    '/data/fuxiaowen/isic-2020-challenge/train-metadata.csv'
]
object_datapath = '/data/fuxiaowen/isic-2024-challenge'
# process_metadata(metadata_paths, object_datapath)
#create_test_set(object_datapath)
clean_metadata(object_datapath)