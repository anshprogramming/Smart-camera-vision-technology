import torch


def download_yolov5():
    # Download YOLOv5s model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

    # Save the model
    torch.save(model.state_dict(), 'yolov5x.pt')


if __name__ == "__main__":
    download_yolov5()
    print("YOLOv5x model downloaded and saved to yolov5x.pt")