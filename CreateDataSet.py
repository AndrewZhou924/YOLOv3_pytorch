import argparse
import time
from sys import platform
from models import *
from utils.datasets import *
from utils.utils import *
from PIL import Image,ImageDraw  
import os
from tqdm import tqdm

def detect():
    # car detect model config
    cfg = 'cfg/yolov3-spp.cfg'
    data_cfg = 'data/coco.data'
    weights = 'weights/yolov3-spp.pt'

    cfg_plate = 'cfg/yolov3-plate-transfer.cfg'
    data_cfg_plate = 'data/transfer.data'
    weights_plate = 'weights/best.pt'

    # images = 'data/samples'
    images = 'pingxiangjiaojing/'
    output='output'
    img_size=416
    conf_thres=0.5
    nms_thres=0.5
    save_txt=False
    save_images=True
    webcam=False

    device = torch_utils.select_device()
    if not os.path.exists(output):
        os.makedirs(output)
        
    # if os.path.exists(output):
    #     shutil.rmtree(output)  # delete output folder
    # os.makedirs(output)  # make new output folder

    # Initialize model
    model = Darknet(cfg, img_size)
    plate_model = Darknet(cfg_plate, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
        plate_model.load_state_dict(torch.load(weights_plate, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    model.to(device).eval()
    plate_model.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None

    dataloader = LoadImages(images, img_size=img_size)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(data_cfg)['names'])
    classes_plate = load_classes(parse_data_cfg(data_cfg_plate)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    for i, (path, img, im0, vid_cap) in tqdm(enumerate(dataloader)):
        # t = time.time()
        # save_path = str(Path(output) / Path(path).name)
        # print("\n\n\n",Path(path).name.split('.')[0]+"\n\n\n\n")

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)

        # print("\n\ninfo:",img.shape,img.type)

        pred, _ = model(img)
        detections = non_max_suppression(pred, conf_thres, nms_thres)[0]

        # img是归一化后的图，im0是原图
        # print("====>",img.size())
        # print("\n====>", im0.shape) # eg:(2008, 3408, 3)

        if detections is not None and len(detections) > 0:
            # Rescale boxes from 416 to true image size
            scale_coords(img_size, detections[:, :4], im0.shape).round()

            # Print results to screen
            for c in detections[:, -1].unique():
                n = (detections[:, -1] == c).sum()
                # print('%g %ss' % (n, classes[int(c)]), end=', ')
                # print('%g %ss' % (n, classes[0]), end=', ')

            # Draw bounding boxes and labels of detections
            car_positions = []
            for *xyxy, conf, cls_conf, cls in detections:

                # print("\n\nxyxyxyxy:::",*xyxy,"\n\n")  
                x = xyxy
                temp_position = (int(x[0]), int(x[1]), int(x[2]), int(x[3]))
                car_positions.append(temp_position)

                # Add bbox to the image
                # label = '%s %.2f' % (classes[int(cls)], conf)

        # 保存车牌相对于原图的坐标
        all_plates = []
        for position in car_positions:
            # print(im0.shape) #(2008, 3408, 3)
            if position[3] > im0.shape[0] or position[2] > im0.shape[1]:
                print("error in create each_car_img0!\n")
                break

            each_car_img0 = im0[position[1]:position[3],position[0]:position[2]]
            origin_shape = each_car_img0.shape
            
            im = Image.fromarray(each_car_img0)
            im = im.resize((416,416))
            temp_array_416 = np.asarray(im)

            temp_img, _, _, _ = letterbox(temp_array_416, height=416)
            temp_img = temp_array_416[:, :, ::-1].transpose(2, 0, 1)
            temp_img = np.ascontiguousarray(temp_img, dtype=np.float32)
            temp_img /= 255.0
            # print("\n\n==>temp_img.shape:",temp_img.shape)

            # Get detections
            each_car_img = torch.from_numpy(temp_img).unsqueeze(0).to(device)

            plate_pred, _ = plate_model(each_car_img)

            plate_detections = non_max_suppression(plate_pred, conf_thres, nms_thres)[0]
            # print("\nplate_detections",plate_detections)
        
            plate_positions = []
            if plate_detections is not None and len(plate_detections) > 0:
                for *xyxy, conf, cls_conf, cls in plate_detections:
                    x = xyxy
                    temp_position = (int(x[0]), int(x[1]), int(x[2]), int(x[3]))
                    plate_positions.append(temp_position)   

            # change the coordination
            if len(plate_positions) != 0:
                x_radio = origin_shape[0]/416
                y_radio = origin_shape[1]/416

                for pos in plate_positions:
                    new_position = (pos[0]*y_radio+position[0], 
                    pos[1]*x_radio+position[1],
                    pos[2]*y_radio+position[0],
                    pos[3]*x_radio+position[1])
                    all_plates.append(new_position)


        '''
        save to txt
        format : label x y w h
        label(0 for car and 1 for plate)
        '''
        filePath = output + "/" + Path(path).name.split('.')[0] + ".txt"
        f = open(filePath,"w")  
        # print(im0.shape)
        length = im0.shape[0]
        height = im0.shape[1]

        # print(car_positions)
        total_info = ""
        for car in car_positions:
            # print(car)
            info = "0"+"\t"+str(car[1])+"\t"+str(car[0])+"\t"+str((car[3]-car[1])/length)+"\t"+str((car[2]-car[0])/height)+"\n"
            total_info += info
            # print(info)
            # f.write(info)

        for plate in all_plates:
            info = "1"+"\t"+str(plate[1])+"\t"+str(plate[0])+"\t"+str((plate[3]-plate[1])/length)+"\t"+str((plate[2]-plate[0])/height)+"\n"
            # print(info)
            total_info += info
        f.write(total_info)


if __name__ == '__main__':
    with torch.no_grad():
        detect()
