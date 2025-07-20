# -*- coding: utf-8 -*-
"""
数据增强模块

提供图像数据增强功能，支持旋转、缩放、亮度调整等变换，
同时保持YOLO格式标注的正确性。
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import numpy as np
import cv2

try:
    import albumentations as A
    from albumentations.core.bbox_utils import BboxParams
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    A = None
    BboxParams = None

from pydantic import BaseModel, Field, ValidationError


# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """数据增强配置"""
    # 几何变换
    rotation_range: Tuple[float, float] = (-15.0, 15.0)  # 旋转角度范围
    scale_range: Tuple[float, float] = (0.8, 1.2)        # 缩放范围
    translate_range: Tuple[float, float] = (-0.1, 0.1)   # 平移范围（相对于图像尺寸）
    
    # 颜色变换
    brightness_range: Tuple[float, float] = (-0.2, 0.2)  # 亮度调整范围
    contrast_range: Tuple[float, float] = (0.8, 1.2)     # 对比度范围
    saturation_range: Tuple[float, float] = (0.8, 1.2)   # 饱和度范围
    hue_range: Tuple[float, float] = (-0.1, 0.1)         # 色调范围
    
    # 噪声和模糊
    noise_probability: float = 0.3                        # 添加噪声的概率
    blur_probability: float = 0.2                         # 模糊的概率
    blur_kernel_size: Tuple[int, int] = (3, 7)          # 模糊核大小范围
    
    # 翻转
    horizontal_flip_probability: float = 0.5              # 水平翻转概率
    vertical_flip_probability: float = 0.0                # 垂直翻转概率（棋盘通常不垂直翻转）
    
    # 裁剪和填充
    crop_probability: float = 0.3                         # 随机裁剪概率
    crop_scale_range: Tuple[float, float] = (0.8, 1.0)   # 裁剪尺寸范围
    
    # 输出设置
    output_size: Optional[Tuple[int, int]] = None         # 输出图像尺寸
    preserve_aspect_ratio: bool = True                    # 保持宽高比


class YOLOBbox(BaseModel):
    """YOLO格式边界框"""
    class_id: int = Field(ge=0, description="类别ID")
    x_center: float = Field(ge=0.0, le=1.0, description="中心点x坐标（归一化）")
    y_center: float = Field(ge=0.0, le=1.0, description="中心点y坐标（归一化）")
    width: float = Field(gt=0.0, le=1.0, description="宽度（归一化）")
    height: float = Field(gt=0.0, le=1.0, description="高度（归一化）")
    
    def to_pascal_voc(self, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """转换为Pascal VOC格式 (x_min, y_min, x_max, y_max)"""
        x_min = (self.x_center - self.width / 2) * img_width
        y_min = (self.y_center - self.height / 2) * img_height
        x_max = (self.x_center + self.width / 2) * img_width
        y_max = (self.y_center + self.height / 2) * img_height
        return (x_min, y_min, x_max, y_max)
    
    @classmethod
    def from_pascal_voc(cls, x_min: float, y_min: float, x_max: float, y_max: float,
                       img_width: int, img_height: int, class_id: int) -> 'YOLOBbox':
        """从Pascal VOC格式创建"""
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        return cls(
            class_id=class_id,
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height
        )


class DataAugmentor:
    """
    数据增强器
    
    提供图像和标注的数据增强功能，支持多种变换类型。
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        初始化数据增强器
        
        Args:
            config: 增强配置，如果为None则使用默认配置
        """
        self.config = config or AugmentationConfig()
        
        # 检查Albumentations是否可用
        if not ALBUMENTATIONS_AVAILABLE:
            logger.warning("Albumentations未安装，将使用基础的OpenCV增强功能")
            self.use_albumentations = False
        else:
            self.use_albumentations = True
            self._setup_albumentations_pipeline()
        
        logger.info("数据增强器初始化完成")
    
    def _setup_albumentations_pipeline(self) -> None:
        """设置Albumentations增强管道"""
        if not self.use_albumentations:
            return
        
        transforms = []
        
        # 几何变换
        if self.config.rotation_range != (0, 0):
            transforms.append(A.Rotate(
                limit=self.config.rotation_range,
                p=0.7,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ))
        
        if self.config.scale_range != (1.0, 1.0):
            transforms.append(A.RandomScale(
                scale_limit=(self.config.scale_range[0] - 1.0, self.config.scale_range[1] - 1.0),
                p=0.5
            ))
        
        # 翻转
        if self.config.horizontal_flip_probability > 0:
            transforms.append(A.HorizontalFlip(p=self.config.horizontal_flip_probability))
        
        if self.config.vertical_flip_probability > 0:
            transforms.append(A.VerticalFlip(p=self.config.vertical_flip_probability))
        
        # 颜色变换
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=self.config.brightness_range,
            contrast_limit=(self.config.contrast_range[0] - 1.0, self.config.contrast_range[1] - 1.0),
            p=0.6
        ))
        
        transforms.append(A.HueSaturationValue(
            hue_shift_limit=(int(self.config.hue_range[0] * 180), int(self.config.hue_range[1] * 180)),
            sat_shift_limit=(int((self.config.saturation_range[0] - 1.0) * 100), 
                           int((self.config.saturation_range[1] - 1.0) * 100)),
            val_shift_limit=0,
            p=0.4
        ))
        
        # 噪声和模糊
        if self.config.noise_probability > 0:
            transforms.append(A.GaussNoise(
                var_limit=(10.0, 50.0),
                p=self.config.noise_probability
            ))
        
        if self.config.blur_probability > 0:
            transforms.append(A.Blur(
                blur_limit=self.config.blur_kernel_size,
                p=self.config.blur_probability
            ))
        
        # 随机裁剪
        if self.config.crop_probability > 0:
            transforms.append(A.RandomSizedBBoxSafeCrop(
                height=640,  # 默认尺寸，会在运行时调整
                width=640,
                erosion_rate=0.0,
                interpolation=cv2.INTER_LINEAR,
                p=self.config.crop_probability
            ))
        
        # 输出尺寸调整
        if self.config.output_size:
            transforms.append(A.Resize(
                height=self.config.output_size[1],
                width=self.config.output_size[0],
                interpolation=cv2.INTER_LINEAR
            ))
        
        # 创建增强管道
        self.albumentations_transform = A.Compose(
            transforms,
            bbox_params=BboxParams(
                format='pascal_voc',
                label_fields=['class_labels'],
                min_area=0,
                min_visibility=0.1
            )
        )
    
    def augment_image_and_bboxes(self, image: np.ndarray, bboxes: List[YOLOBbox]) -> Tuple[np.ndarray, List[YOLOBbox]]:
        """
        对图像和边界框进行增强
        
        Args:
            image: 输入图像 (H, W, C)
            bboxes: YOLO格式边界框列表
            
        Returns:
            增强后的图像和边界框
        """
        if len(bboxes) == 0:
            # 没有边界框时只增强图像
            return self._augment_image_only(image), []
        
        if self.use_albumentations:
            return self._augment_with_albumentations(image, bboxes)
        else:
            return self._augment_with_opencv(image, bboxes)
    
    def _augment_with_albumentations(self, image: np.ndarray, bboxes: List[YOLOBbox]) -> Tuple[np.ndarray, List[YOLOBbox]]:
        """使用Albumentations进行增强"""
        h, w = image.shape[:2]
        
        # 转换YOLO格式到Pascal VOC格式
        pascal_bboxes = []
        class_labels = []
        
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox.to_pascal_voc(w, h)
            pascal_bboxes.append([x_min, y_min, x_max, y_max])
            class_labels.append(bbox.class_id)
        
        try:
            # 应用增强
            transformed = self.albumentations_transform(
                image=image,
                bboxes=pascal_bboxes,
                class_labels=class_labels
            )
            
            augmented_image = transformed['image']
            augmented_bboxes = transformed['bboxes']
            augmented_labels = transformed['class_labels']
            
            # 转换回YOLO格式
            new_h, new_w = augmented_image.shape[:2]
            yolo_bboxes = []
            
            for (x_min, y_min, x_max, y_max), class_id in zip(augmented_bboxes, augmented_labels):
                try:
                    yolo_bbox = YOLOBbox.from_pascal_voc(
                        x_min, y_min, x_max, y_max, new_w, new_h, class_id
                    )
                    yolo_bboxes.append(yolo_bbox)
                except ValidationError as e:
                    logger.warning(f"跳过无效的边界框: {e}")
                    continue
            
            return augmented_image, yolo_bboxes
            
        except Exception as e:
            logger.warning(f"Albumentations增强失败，使用原始数据: {e}")
            return image, bboxes
    
    def _augment_with_opencv(self, image: np.ndarray, bboxes: List[YOLOBbox]) -> Tuple[np.ndarray, List[YOLOBbox]]:
        """使用OpenCV进行基础增强"""
        augmented_image = image.copy()
        augmented_bboxes = bboxes.copy()
        
        # 亮度调整
        if random.random() < 0.6:
            brightness = random.uniform(*self.config.brightness_range)
            augmented_image = cv2.convertScaleAbs(augmented_image, alpha=1.0, beta=brightness * 255)
        
        # 对比度调整
        if random.random() < 0.6:
            contrast = random.uniform(*self.config.contrast_range)
            augmented_image = cv2.convertScaleAbs(augmented_image, alpha=contrast, beta=0)
        
        # 水平翻转
        if random.random() < self.config.horizontal_flip_probability:
            augmented_image = cv2.flip(augmented_image, 1)
            # 翻转边界框
            for bbox in augmented_bboxes:
                bbox.x_center = 1.0 - bbox.x_center
        
        # 高斯模糊
        if random.random() < self.config.blur_probability:
            kernel_size = random.randrange(self.config.blur_kernel_size[0], 
                                         self.config.blur_kernel_size[1] + 1, 2)
            augmented_image = cv2.GaussianBlur(augmented_image, (kernel_size, kernel_size), 0)
        
        return augmented_image, augmented_bboxes
    
    def _augment_image_only(self, image: np.ndarray) -> np.ndarray:
        """仅对图像进行增强（无边界框）"""
        if self.use_albumentations:
            # 创建仅图像的增强管道
            image_only_transforms = A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=self.config.brightness_range,
                    contrast_limit=(self.config.contrast_range[0] - 1.0, self.config.contrast_range[1] - 1.0),
                    p=0.6
                ),
                A.HueSaturationValue(
                    hue_shift_limit=(int(self.config.hue_range[0] * 180), int(self.config.hue_range[1] * 180)),
                    sat_shift_limit=(int((self.config.saturation_range[0] - 1.0) * 100), 
                                   int((self.config.saturation_range[1] - 1.0) * 100)),
                    val_shift_limit=0,
                    p=0.4
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=self.config.noise_probability),
                A.Blur(blur_limit=self.config.blur_kernel_size, p=self.config.blur_probability),
                A.HorizontalFlip(p=self.config.horizontal_flip_probability)
            ])
            
            try:
                transformed = image_only_transforms(image=image)
                return transformed['image']
            except Exception as e:
                logger.warning(f"图像增强失败: {e}")
                return image
        else:
            return self._augment_with_opencv(image, [])[0]
    
    def augment_dataset(self, input_dir: Union[str, Path], output_dir: Union[str, Path],
                       augmentation_factor: int = 2, preserve_original: bool = True) -> Dict[str, int]:
        """
        对整个数据集进行增强
        
        Args:
            input_dir: 输入数据目录
            output_dir: 输出数据目录
            augmentation_factor: 增强倍数（每张原图生成多少张增强图）
            preserve_original: 是否保留原始数据
            
        Returns:
            增强统计信息
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # 创建输出目录
        output_images_dir = output_dir / "images"
        output_labels_dir = output_dir / "labels"
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找输入文件
        input_images_dir = input_dir / "images"
        input_labels_dir = input_dir / "labels"
        
        if not input_images_dir.exists():
            raise ValueError(f"输入图像目录不存在: {input_images_dir}")
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        # 在Windows上，文件系统不区分大小写，所以我们需要避免重复计数
        # 使用集合来存储唯一的文件路径
        unique_files = set()
        
        for ext in image_extensions:
            # 只使用小写扩展名进行搜索，避免在不区分大小写的文件系统上重复计数
            for file in input_images_dir.rglob(f"*{ext}"):
                unique_files.add(str(file))
        
        # 将唯一的文件路径转换回Path对象
        image_files = [Path(file) for file in unique_files]
        
        if not image_files:
            raise ValueError(f"在目录 {input_images_dir} 中未找到图像文件")
        
        stats = {
            'input_images': len(image_files),  # 输入图像总数
            'original_images': 0,  # 保留的原始图像数
            'augmented_images': 0,  # 增强生成的图像数
            'total_images': 0,  # 输出的总图像数
            'skipped_images': 0,
            'processing_errors': 0
        }
        
        logger.info(f"开始增强数据集，输入图像: {len(image_files)} 张，增强倍数: {augmentation_factor}")
        
        for image_file in image_files:
            try:
                # 读取图像
                image = cv2.imread(str(image_file))
                if image is None:
                    logger.warning(f"无法读取图像: {image_file}")
                    stats['skipped_images'] += 1
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 读取对应的标注文件
                label_file = input_labels_dir / f"{image_file.stem}.txt"
                bboxes = []
                
                if label_file.exists():
                    bboxes = self._load_yolo_annotations(label_file)
                
                # 保留原始数据
                if preserve_original:
                    self._save_augmented_data(
                        image, bboxes, output_images_dir, output_labels_dir,
                        f"{image_file.stem}_original", image_file.suffix
                    )
                    stats['original_images'] += 1
                    stats['total_images'] += 1
                
                # 生成增强数据
                for i in range(augmentation_factor):
                    try:
                        aug_image, aug_bboxes = self.augment_image_and_bboxes(image, bboxes)
                        
                        # 保存增强数据
                        self._save_augmented_data(
                            aug_image, aug_bboxes, output_images_dir, output_labels_dir,
                            f"{image_file.stem}_aug_{i:03d}", image_file.suffix
                        )
                        
                        stats['augmented_images'] += 1
                        stats['total_images'] += 1
                        
                    except Exception as e:
                        logger.warning(f"增强图像失败 {image_file} (第{i}次): {e}")
                        stats['processing_errors'] += 1
                        continue
                
            except Exception as e:
                logger.error(f"处理图像失败 {image_file}: {e}")
                stats['processing_errors'] += 1
                continue
        
        # 保存增强统计信息
        self._save_augmentation_stats(output_dir, stats)
        
        logger.info(f"数据集增强完成: 总计生成 {stats['total_images']} 张图像")
        
        return stats
    
    def _load_yolo_annotations(self, label_file: Path) -> List[YOLOBbox]:
        """加载YOLO格式标注文件"""
        bboxes = []
        
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    continue
                
                try:
                    bbox = YOLOBbox(
                        class_id=int(parts[0]),
                        x_center=float(parts[1]),
                        y_center=float(parts[2]),
                        width=float(parts[3]),
                        height=float(parts[4])
                    )
                    bboxes.append(bbox)
                except (ValueError, ValidationError) as e:
                    logger.warning(f"跳过无效标注行 {label_file}: {line} - {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"读取标注文件失败 {label_file}: {e}")
        
        return bboxes
    
    def _save_augmented_data(self, image: np.ndarray, bboxes: List[YOLOBbox],
                           images_dir: Path, labels_dir: Path,
                           filename_stem: str, image_extension: str) -> None:
        """保存增强后的图像和标注"""
        # 保存图像
        image_file = images_dir / f"{filename_stem}{image_extension}"
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(image_file), image_bgr)
        
        # 保存标注
        label_file = labels_dir / f"{filename_stem}.txt"
        with open(label_file, 'w', encoding='utf-8') as f:
            for bbox in bboxes:
                f.write(f"{bbox.class_id} {bbox.x_center:.6f} {bbox.y_center:.6f} "
                       f"{bbox.width:.6f} {bbox.height:.6f}\n")
    
    def _save_augmentation_stats(self, output_dir: Path, stats: Dict[str, int]) -> None:
        """保存增强统计信息"""
        stats_file = output_dir / "augmentation_stats.json"
        
        # 添加配置信息
        full_stats = {
            'statistics': stats,
            'configuration': {
                'rotation_range': self.config.rotation_range,
                'scale_range': self.config.scale_range,
                'brightness_range': self.config.brightness_range,
                'contrast_range': self.config.contrast_range,
                'horizontal_flip_probability': self.config.horizontal_flip_probability,
                'noise_probability': self.config.noise_probability,
                'blur_probability': self.config.blur_probability,
                'crop_probability': self.config.crop_probability,
                'use_albumentations': self.use_albumentations
            }
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(full_stats, f, indent=2, ensure_ascii=False)
        
        # 生成可读报告
        report_file = output_dir / "augmentation_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("数据增强报告\n")
            f.write("=" * 40 + "\n\n")
            
            f.write("统计信息:\n")
            f.write("-" * 20 + "\n")
            f.write(f"原始图像数: {stats['original_images']}\n")
            f.write(f"增强图像数: {stats['augmented_images']}\n")
            f.write(f"总图像数: {stats['total_images']}\n")
            f.write(f"跳过图像数: {stats['skipped_images']}\n")
            f.write(f"处理错误数: {stats['processing_errors']}\n")
            
            if stats['original_images'] > 0:
                augmentation_ratio = stats['augmented_images'] / stats['original_images']
                f.write(f"增强倍数: {augmentation_ratio:.2f}\n")
            
            f.write("\n配置信息:\n")
            f.write("-" * 20 + "\n")
            f.write(f"旋转范围: {self.config.rotation_range}\n")
            f.write(f"缩放范围: {self.config.scale_range}\n")
            f.write(f"亮度范围: {self.config.brightness_range}\n")
            f.write(f"对比度范围: {self.config.contrast_range}\n")
            f.write(f"水平翻转概率: {self.config.horizontal_flip_probability}\n")
            f.write(f"噪声概率: {self.config.noise_probability}\n")
            f.write(f"模糊概率: {self.config.blur_probability}\n")
            f.write(f"使用Albumentations: {self.use_albumentations}\n")
    
    def preview_augmentation(self, image: np.ndarray, bboxes: List[YOLOBbox],
                           num_samples: int = 4) -> List[Tuple[np.ndarray, List[YOLOBbox]]]:
        """
        预览增强效果
        
        Args:
            image: 输入图像
            bboxes: 边界框列表
            num_samples: 生成的预览样本数量
            
        Returns:
            增强样本列表
        """
        samples = []
        
        for i in range(num_samples):
            try:
                aug_image, aug_bboxes = self.augment_image_and_bboxes(image, bboxes)
                samples.append((aug_image, aug_bboxes))
            except Exception as e:
                logger.warning(f"生成预览样本失败 (第{i}个): {e}")
                continue
        
        return samples
    
    def get_augmentation_suggestions(self, dataset_stats: Dict[str, Any]) -> List[str]:
        """
        根据数据集统计信息提供增强建议
        
        Args:
            dataset_stats: 数据集统计信息
            
        Returns:
            增强建议列表
        """
        suggestions = []
        
        # 检查数据量
        total_images = dataset_stats.get('total_files', 0)
        if total_images < 1000:
            suggestions.append(f"数据量较少({total_images}张)，建议增强倍数设为3-5倍")
        elif total_images < 5000:
            suggestions.append(f"数据量中等({total_images}张)，建议增强倍数设为2-3倍")
        else:
            suggestions.append(f"数据量充足({total_images}张)，可适度增强1-2倍")
        
        # 检查类别分布
        class_distribution = dataset_stats.get('class_distribution', {})
        if class_distribution:
            min_count = min(class_distribution.values())
            max_count = max(class_distribution.values())
            
            if max_count / min_count > 5:
                suggestions.append("类别分布不均衡，建议对少数类别进行更多增强")
        
        # 检查图像质量相关建议
        suggestions.extend([
            "建议使用适度的旋转(-15°到15°)以增加棋盘角度变化",
            "亮度和对比度调整有助于适应不同光照条件",
            "水平翻转可以增加数据多样性，但注意棋子文字方向",
            "避免过度的几何变换，以免影响棋子识别准确性"
        ])
        
        return suggestions


def create_default_augmentation_config() -> AugmentationConfig:
    """创建适合棋盘识别的默认增强配置"""
    return AugmentationConfig(
        # 适度的几何变换，避免影响棋子识别
        rotation_range=(-10.0, 10.0),
        scale_range=(0.9, 1.1),
        translate_range=(-0.05, 0.05),
        
        # 颜色变换以适应不同光照
        brightness_range=(-0.15, 0.15),
        contrast_range=(0.9, 1.1),
        saturation_range=(0.9, 1.1),
        hue_range=(-0.05, 0.05),
        
        # 适度的噪声和模糊
        noise_probability=0.2,
        blur_probability=0.1,
        blur_kernel_size=(3, 5),
        
        # 水平翻转（注意棋子文字）
        horizontal_flip_probability=0.3,
        vertical_flip_probability=0.0,
        
        # 轻微裁剪
        crop_probability=0.2,
        crop_scale_range=(0.9, 1.0),
        
        # 保持原始尺寸
        output_size=None,
        preserve_aspect_ratio=True
    )