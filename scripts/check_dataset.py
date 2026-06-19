from pathlib import Path
from PIL import Image
import numpy as np


# 项目根目录：D:\uav_segmentation_project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 数据集路径
DATA_ROOT = PROJECT_ROOT / "data" / "raw"

# 允许读取的图片格式
IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_valid_image_file(path: Path) -> bool:
    """
    过滤掉 .DS_Store、._xxx.png 这类无效文件，只保留真正的图片文件。
    """
    if path.name.startswith("."):
        return False
    if path.name.startswith("._"):
        return False
    return path.suffix.lower() in IMG_SUFFIXES


def collect_files(folder: Path):
    """
    收集文件夹下所有合法图片文件，并按文件名排序。
    """
    if not folder.exists():
        print(f"[错误] 文件夹不存在：{folder}")
        return []

    files = [p for p in folder.iterdir() if p.is_file() and is_valid_image_file(p)]
    files = sorted(files, key=lambda x: x.name)
    return files


def check_split(split_name: str):
    """
    检查 train 或 test 数据集：
    1. 图片数量
    2. 标签数量
    3. 文件名是否一一对应
    4. 图片与标签尺寸是否一致
    5. 标签中出现了哪些类别编号
    """
    print("=" * 80)
    print(f"正在检查 {split_name} 数据集")
    print("=" * 80)

    image_dir = DATA_ROOT / split_name / "images"
    label_dir = DATA_ROOT / split_name / "labels"

    image_files = collect_files(image_dir)
    label_files = collect_files(label_dir)

    print(f"图片文件夹：{image_dir}")
    print(f"标签文件夹：{label_dir}")
    print(f"图片数量：{len(image_files)}")
    print(f"标签数量：{len(label_files)}")

    if len(image_files) == 0:
        print(f"[错误] {split_name}/images 中没有找到图片")
        return

    if len(label_files) == 0:
        print(f"[错误] {split_name}/labels 中没有找到标签")
        return

    image_names = {p.name for p in image_files}
    label_names = {p.name for p in label_files}

    missing_labels = sorted(image_names - label_names)
    missing_images = sorted(label_names - image_names)

    if missing_labels:
        print(f"[错误] 有 {len(missing_labels)} 张图片找不到对应标签，例如：")
        print(missing_labels[:10])
    else:
        print("[通过] 每张图片都能找到对应标签")

    if missing_images:
        print(f"[错误] 有 {len(missing_images)} 张标签找不到对应图片，例如：")
        print(missing_images[:10])
    else:
        print("[通过] 每张标签都能找到对应图片")

    # 只检查两边都有的文件
    common_names = sorted(image_names & label_names)

    all_label_values = set()
    size_mismatch_count = 0
    open_error_count = 0

    for name in common_names:
        image_path = image_dir / name
        label_path = label_dir / name

        try:
            image = Image.open(image_path)
            label = Image.open(label_path)

            image_size = image.size
            label_size = label.size

            if image_size != label_size:
                size_mismatch_count += 1
                if size_mismatch_count <= 10:
                    print(f"[尺寸不一致] {name}: image={image_size}, label={label_size}")

            label_array = np.array(label)
            unique_values = np.unique(label_array)
            all_label_values.update(unique_values.tolist())

        except Exception as e:
            open_error_count += 1
            print(f"[读取失败] {name}: {e}")

    if size_mismatch_count == 0:
        print("[通过] 所有图片和标签尺寸一致")
    else:
        print(f"[警告] 共有 {size_mismatch_count} 对图片和标签尺寸不一致")

    if open_error_count == 0:
        print("[通过] 所有图片和标签都可以正常读取")
    else:
        print(f"[警告] 共有 {open_error_count} 个文件读取失败")

    all_label_values = sorted(all_label_values)

    print(f"标签中出现的类别编号：{all_label_values}")
    print(f"类别数量：{len(all_label_values)}")

    # 顺便展示一张样例的信息
    sample_name = common_names[0]
    sample_image = Image.open(image_dir / sample_name)
    sample_label = Image.open(label_dir / sample_name)

    print("-" * 80)
    print("样例文件：")
    print(f"文件名：{sample_name}")
    print(f"原图模式：{sample_image.mode}, 尺寸：{sample_image.size}")
    print(f"标签模式：{sample_label.mode}, 尺寸：{sample_label.size}")
    print("-" * 80)


def main():
    print(f"项目根目录：{PROJECT_ROOT}")
    print(f"数据根目录：{DATA_ROOT}")

    check_split("train")
    check_split("test")

    print("=" * 80)
    print("数据集检查完成")
    print("=" * 80)


if __name__ == "__main__":
    main()