import os
import glob
import cv2
from ultralytics import YOLO
from pathlib import Path


class InferenceTool:
    """推理测试工具类"""

    def __init__(self, model_path):
        """
        初始化推理工具

        Args:
            model_path: 模型文件路径
        """
        self.model_path = model_path
        self.model = None

    def load_model(self):
        """加载模型"""
        if self.model is None:
            self.model = YOLO(self.model_path)
        return self.model

    def predict_single_image(self, image_path, conf=0.25, save=True, show_result=False):
        """
        单张图片推理

        Args:
            image_path: 图片路径
            conf: 置信度阈值
            save: 是否保存结果
            show_result: 是否显示结果

        Returns:
            results: 推理结果
        """
        model = self.load_model()
        results = model(image_path, save=save, conf=conf)

        # 显示结果
        if show_result and save:
            save_path = results[0].save_dir / results[0].path.name
            img = cv2.imread(str(save_path))
            if img is not None:
                cv2.imshow('Detection Result', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        return results

    def predict_batch_images(self, image_dir, conf=0.25, save=True):
        """
        批量图片推理

        Args:
            image_dir: 图片目录
            conf: 置信度阈值
            save: 是否保存结果

        Returns:
            results: 推理结果
        """
        model = self.load_model()
        results = model(image_dir, save=save, conf=conf)
        return results

    def predict_video(self, video_path, conf=0.25, save=True):
        """
        视频推理

        Args:
            video_path: 视频路径
            conf: 置信度阈值
            save: 是否保存结果

        Returns:
            results: 推理结果
        """
        model = self.load_model()
        results = model(video_path, save=save, conf=conf)
        return results

    def predict_webcam(self, camera_id=0, conf=0.25, show=True):
        """
        摄像头实时推理

        Args:
            camera_id: 摄像头ID
            conf: 置信度阈值
            show: 是否显示实时画面

        Returns:
            results: 推理结果
        """
        model = self.load_model()
        results = model(camera_id, show=show, conf=conf, stream=True)
        return results

    def validate(self, data_yaml=None, split='val', conf=0.5, iou=0.6):
        """
        模型验证

        Args:
            data_yaml: 数据集配置文件
            split: 数据集分割（'train', 'val', 'test'）
            conf: 置信度阈值
            iou: IoU阈值

        Returns:
            metrics: 验证指标
        """
        if not data_yaml or not os.path.exists(data_yaml):
            return None

        model = self.load_model()
        metrics = model.val(data=data_yaml, split=split, conf=conf, iou=iou)
        return metrics


def create_inference_tool(model_path):
    """
    创建推理工具实例

    Args:
        model_path: 模型路径

    Returns:
        InferenceTool 实例
    """
    return InferenceTool(model_path)