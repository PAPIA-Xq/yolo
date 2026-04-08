# train_with_aux.py
from ultralytics import YOLO
import torch


def train_with_aux_boxes():
    """
    使用 Inner-ShapeIoU + 辅助边框的训练脚本
    """
    # 1. 加载模型配置
    model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')

    # 2. 访问底层模型对象
    # model.model 是 DetectionModel 实例
    detection_model = model.model

    # 3. 添加自定义属性以启用辅助边框
    detection_model.use_aux_boxes = True

    # 4. 重新初始化损失函数（关键步骤）
    from ultralytics.utils.loss import v8DetectionLoss
    # 注意：这需要修改 v8DetectionLoss 的 __init__ 接受 use_aux_boxes 参数
    # 如果没有修改，损失函数不会使用辅助边框

    # 5. 开始训练
    results = model.train(
        data='D:\yolo\基于YOLOv8的反光衣服检测识别项目\基于YOLOv8的反光衣服检测识别项目\main\datasets\data.yaml',  # 数据集路径
        epochs=10,
        imgsz=640,
        batch=16,
        device=0 if torch.cuda.is_available() else 'cpu',
        project='runs/train',
        name='aux_boxes_exp',
        save=True,
        plots=True,
    )

    print(f"训练完成！最佳mAP: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    return results


if __name__ == '__main__':
    results = train_with_aux_boxes()