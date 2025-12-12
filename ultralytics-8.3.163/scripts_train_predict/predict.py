from ultralytics import YOLO

if __name__ == '__main__':

    # 定义保存路径
    save_dir = 'train_results'

    # Load a model
    model = YOLO(model='/home/hzq/ultralytics-8.3.163/yolo11n.pt')  
    model.predict(source='/home/hzq/ultralytics-8.3.163/ultralytics/assets/bus.jpg',
                  save=True,
                  show=True,
                  project=save_dir,  # 保存项目目录
                  name='predict_output/bus',  # 保存的子目录名称
                  exist_ok=True,  # 如果目录已存在，则覆盖
                  )
