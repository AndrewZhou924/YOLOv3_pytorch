import argparse
import time
from sys import platform
from models import *
from utils.datasets import *
from utils.utils import *
from PIL import Image,ImageDraw  

def detect():
    # car detect model config
    cfg = 'cfg/yolov3-spp.cfg'
    data_cfg = 'data/coco.data'
    weights = 'weights/yolov3-spp.pt'

    cfg_plate = 'cfg/yolov3-plate-transfer.cfg'
    data_cfg_plate = 'data/transfer.data'
    weights_plate = 'weights/best.pt'

    images = 'data/samples'
    output='output'
    img_size=416
    conf_thres=0.5
    nms_thres=0.5
    save_txt=False
    save_images=True
    webcam=False

    device = torch_utils.select_device()
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

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

    for i, (path, img, im0, vid_cap) in enumerate(dataloader):
        t = time.time()
        save_path = str(Path(output) / Path(path).name)

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)

        print("\n\ninfo:",img.shape,img.type)

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

                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                # Add bbox to the image
                label = '%s %.2f' % (classes[int(cls)], conf)

        # print(car_positions)
        # print('Done. (%.3fs)' % (time.time() - t))
        car_positions.pop(-1)
        car_positions.pop(-1)

        # 保存车牌相对于原图的坐标
        all_plates = []
        for position in car_positions:
            # print(im0.shape) #(2008, 3408, 3)
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

            # print("\n\ninfo:",each_car_img.shape,each_car_img.type)
            # print("\n\neach_car_img.shape:",each_car_img.shape)

            plate_pred, _ = plate_model(each_car_img)

            plate_detections = non_max_suppression(plate_pred, conf_thres, nms_thres)[0]
            # print("\nplate_detections",plate_detections)
        
            plate_positions = []
            if plate_detections is not None and len(plate_detections) > 0:
                for *xyxy, conf, cls_conf, cls in plate_detections:
                    x = xyxy
                    temp_position = (int(x[0]), int(x[1]), int(x[2]), int(x[3]))
                    plate_positions.append(temp_position)
                
                # just for test
                # ===========================================================================
                # Rescale boxes from 416 to true image size
                scale_coords(img_size, plate_detections[:, :4], temp_array_416.shape).round()

                # Print results to screen
                for c in plate_detections[:, -1].unique():
                    n = (plate_detections[:, -1] == c).sum()
                    print('%g %ss' % (n, classes_plate[int(c)]), end=', ')

                # Draw bounding boxes and labels of detections
                for *xyxy, conf, cls_conf, cls in plate_detections:
                    # Add bbox to the image
                    print("\nin test:",xyxy)
                    label = '%s %.2f' % (classes_plate[int(cls)], conf)
                    plot_one_box(xyxy, temp_array_416, label=label, color=colors[int(cls)])
                cv2.imwrite(save_path, temp_array_416)
                 # ===========================================================================

            # change the coordination
            # TODO:fix bug
            if len(plate_positions) != 0:
                x_radio = origin_shape[0]/416
                y_radio = origin_shape[1]/416
                print("\n\n\n",x_radio,y_radio)
                for position in plate_positions:
                    new_position = (position[0]*x_radio+position[1], 
                    position[1]*y_radio+position[0],
                    position[2]*x_radio+position[1],
                    position[3]*y_radio+position[0])
                    all_plates.append(new_position)

                    new_position = (position[0]*x_radio+position[0], 
                    position[1]*y_radio+position[1],
                    position[2]*x_radio+position[0],
                    position[3]*y_radio+position[1])
                    all_plates.append(new_position)


        # print("\n\nall:",all_plates)

        # TODO:fix bug
        originPic = Image.fromarray(im0)
        draw = ImageDraw.Draw(originPic)
        # print("originPic.shape",originPic.size)

        for plate in all_plates:
            draw.rectangle(((plate[0],plate[1]),(plate[2],plate[3])), outline='red')
            # draw.rectangle(((plate[1],plate[0]),(plate[3],plate[2])), outline='red')

        originPic.save("AllpicTest.jpg")



if __name__ == '__main__':
    with torch.no_grad():
        detect()
