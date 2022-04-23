import time
import cv2
import pyodbc
import numpy as np
import tensorflow as tf
import os
import datetime
import base64

from inferenceutils import * 

path_file = os.getcwd().replace("\\", "/")+"/"

#Classification
# model = tf.keras.models.load_model(path_file+"model_casification_hm2/model2.h5")
# model.load_weights(path_file+'model_casification_hm2/weight2.h5')

model = tf.keras.models.load_model("D:/HM2/programs/model_casification_hm2/model2.h5")
model.load_weights("D:/HM2/programs/model_casification_hm2/weight2.h5")
classes=['corn','empty','wg']
img_width,img_height  = 224,224

class Auto_sampling_hm2:

    def __init__(self):
        pass

    def connect_db_hm2(self):
        for loop_connect_db in range(5):
            try:
                # connection server JobHM
                server = '192.168.1.11' 
                database = 'JobHM' 
                username = 'dbHM' 
                password = 'dbHM' 
                mydb = pyodbc.connect('DRIVER={ODBC Driver 11 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
                mycursor = mydb.cursor()
                loop_connect_db = None
                break
            except:
                pass
        if loop_connect_db == 4:
            mycursor = False
        return mycursor


    def ping_ip_hm(self):
        response = None
        ip_list = ['192.168.1.11']
        for ip in ip_list:
            response = os.popen(f"ping {ip}").read()
            if "Received = 4" in response:
                value_ip = True
                # print(f"UP {ip} Ping Successful")
            else:
                value_ip = False 
                # print(f"DOWN {ip} Ping Unsuccessful")
        return value_ip

    def time_strf(self):
        return time.strftime("%d-" "%m-" "%Y"" " "%H." "%M." "%S")

    def update_sampling_finish(self, values_sampling_finish):
        for i_ups in range(5):
            try:
                ups = self.connect_db_hm2()
                ups.execute("update SampingHM2 SET samping_finish={} WHERE id = 1".format(values_sampling_finish))
                ups.commit()
                i_ups = None
                status_loop = True
                break
            except:
                pass
            time.sleep(5)
        if i_ups == 4:
            status_loop = False
        return status_loop

    def update_pic_sampling(self, values_pic_sampling):
        for i_ups in range(5):
            try:
                ups = self.connect_db_hm2()
                ups.execute("update SampingHM2 SET picture_samping={} WHERE id = 1".format(values_pic_sampling))
                ups.commit()
                i_ups = None
                status_loop = True
                break
            except:
                pass
            time.sleep(5)
        if i_ups == 4:
            status_loop = False
        return status_loop

    def update_pic_screen(self, values_pic_screen):
        for i_pic_screen in range(5):
            try:
                pic_screen = self.connect_db_hm2()
                pic_screen.execute("update SampingHM2 SET picture_screen={} WHERE id = 1".format(values_pic_screen))
                pic_screen.commit()
                i_pic_screen = None
                status_picture_screen = True
                break
            except:
                pass
            time.sleep(5)
        if i_pic_screen == 4:
            status_picture_screen = False
        return status_picture_screen

    def update_status(self, values_status):
        for i_status in range(5):
            try:
                up_status = self.connect_db_hm2()
                up_status.execute("update SampingHM2 SET status={} WHERE id = 1".format(values_status))
                up_status.commit()
                i_status = None
                status_up_status = True
                break
            except:
                pass
            time.sleep(5)
        if i_status == 4:
            status_up_status = False
        return status_up_status

    def update_hm2(self, values_hm2):
        for i_hm2 in range(5):
            try:
                up_hm2 = self.connect_db_hm2()
                up_hm2.execute("update SampingHM2 SET hm_2={} WHERE id = 1".format(values_hm2))
                up_hm2.commit()
                i_hm2 = None
                status_hm2 = True
                break
            except:
                pass
            time.sleep(5)
        if i_hm2 == 4:
            status_hm2 = False
        return status_hm2

    def update_train(self, values_train):
        for i_hm2 in range(5):
            try:
                up_hm2 = self.connect_db_hm2()
                up_hm2.execute("update SampingHM2 SET train={} WHERE id = 1".format(values_train))
                up_hm2.commit()
                i_hm2 = None
                status_train = True
                break
            except:
                pass
            time.sleep(5)
        if i_hm2 == 4:
            status_train = False
        return status_train

    def up_run_process(self, values_run):
        for i in range(5):
            status_run = False
            try:
                up_run = self.connect_db_hm2()
                up_run.execute("update SampingHM2 SET run_process={} WHERE id = 1".format(values_run))
                up_run.commit()
                status_run = True
                break
            except:
                pass
        time.sleep(5)
        return status_run

    def select_run_process(self):
        for i_hm1 in range(5):
            status_run_hm1 = False
            try:
                run_process_hm1 = self.connect_db_hm2()
                run_process_hm1.execute("select * from SampingHM2 where id=2")
                row_hm1 = run_process_hm1.fetchone()
                if row_hm1[8] == 1:
                    status_run_hm1 = True
                    break
            except:
                pass
            time.sleep(2)

        for i_hm2 in range(5):
            status_run_hm2 = False
            try:
                run_process_hm2 = self.connect_db_hm2()
                run_process_hm2.execute("select * from SampingHM2 where id=1")
                row_hm2 = run_process_hm2.fetchone()
                if row_hm2[8] == 1:
                    status_run_hm2 = True
                    break
            except:
                pass
            time.sleep(2)
        
        return status_run_hm1, status_run_hm2

    def record_vdo_for_train(self, file_name):
        print(file_name)
        filename = '{}.mp4'.format(file_name)
        frames_per_second = 25.0
        res = '720p'

        # Set resolution for the video capture
        # Function adapted from
        def change_res(cap, width, height):
            cap.set(3, width)
            cap.set(4, height)

        # Standard Video Dimensions Sizes
        STD_DIMENSIONS =  {
            "480p": (640, 360),
            "720p": (1280, 720),
            "1080p": (1920, 1080),
            "4k": (3840, 2160),
        }

        # grab resolution dimensions and set video capture to it.
        def get_dims(cap, res='1080p'):
            width, height = STD_DIMENSIONS["480p"]
            if res in STD_DIMENSIONS:
                width,height = STD_DIMENSIONS[res]
            ## change the current caputre device
            ## to the resulting resolution
            change_res(cap, width, height)
            return width, height

        # Video Encoding, might require additional installs
        # Types of Codes: 
        VIDEO_TYPE = {
            'avi': cv2.VideoWriter_fourcc(*'XVID'),
            #'mp4': cv2.VideoWriter_fourcc(*'H264'),
            'mp4': cv2.VideoWriter_fourcc(*'XVID'),
        }

        def get_video_type(filename):
            filename, ext = os.path.splitext(filename)
            if ext in VIDEO_TYPE:
                return  VIDEO_TYPE[ext]
            return VIDEO_TYPE['avi']

        cap = cv2.VideoCapture('rtsp://admin:admin@192.168.1.92:554/11')
        out = cv2.VideoWriter(filename, get_video_type(filename), 25, get_dims(cap, res))

        time_record = 0

        while cap.isOpened() and time_record < 25:
            ret, frame = cap.read()
            out.write(frame)
            # cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time_record += 1
        print("...Record VDO Success")
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def crop_img(self, path_folder):
        stamp_tm = time.strftime("%d-" "%m-" "%Y"" " "%H." "%M." "%S")
        path_vdo = "D:/HM2/programs/"
        count = 0
        for list in os.listdir(path_vdo):
            if list == "classifi.mp4":
                cap = cv2.VideoCapture(path_vdo+list)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret == True:
                        frame_crop = frame[250:480,460:900,:]
                        cv2.imwrite("D:/HM2/programs/img_for_train/{}/{}_{}_{}.png" .format(path_folder, path_folder, stamp_tm, count), frame_crop)
                        count += 1
                    elif ret == False:
                        break

                    if cv2.waitKey(1) & 0xFF == ord ("q"):
                        break
                print("...Crop img Success")
                cap.release()
                cv2.destroyAllWindows()

    def update_picture_to_dataset(self):
        count_corn = 0
        count_wg = 0
        count_empty = 0

        list_empty = []
        for list in os.listdir("D:/HM2/programs/"):
            # print(list)
            if list == "img_for_train":
                pass
                for list_in in os.listdir("D:/HM2/programs/"+list):
                    # print(list_in)
                    count = 0
                    for l_pic in os.listdir("D:/HM2/programs/"+list+"/"+list_in):
                        # print(l_pic)
                        count += 1
                    print()
                    if list_in == "corn":
                        count_corn = count
                        list_empty.append(count_corn)
                    elif list_in == "wg":
                        count_wg = count
                        list_empty.append(count_wg)
                    elif list_in == "empty":
                        count_empty = count
                        list_empty.append(count_empty)

        min_score = min(list_empty)

        loop_i = min_score
        print("...Picture In Folder =", loop_i)

        for loop_folder in os.listdir(path_file):
            if loop_folder == "img_for_train":
                for loop_for_read in os.listdir("D:/HM2/programs/img_for_train"):
                    cooldown = 0
                    for loop_save_img in os.listdir("D:/HM2/programs/img_for_train/{}".format(loop_for_read)):
                        if cooldown == loop_i:
                            break
                        img = cv2.imread("D:/HM2/programs/img_for_train/{}/{}".format(loop_for_read, loop_save_img))
                        cv2.imwrite("D:/HM2/programs/dataset/{}/{}_{}_{}.png".format(loop_for_read, loop_for_read, cooldown, self.time_strf(), img))
                       
                        cooldown += 1

        for loop_remove in os.listdir("D:/HM2/programs/img_for_train"):
            for remove_img in os.listdir("D:/HM2/programs/img_for_train/{}".format(loop_remove)):
                os.remove("D:/HM2/programs/img_for_train/{}/{}".format(loop_remove,remove_img))
             

    # def remove_vdo(self):
    #     try:
    #         os.remove(path_file+"programs/classifi.mp4")
    #         print("...Remove VDO Success")
    #     except:
    #         print("...Not Remove VDO !!!")
    #         pass 

    def log_file(self, log):
        file_log = open("D:/HM2/programs/file_log_hm2.log", "a")
        file_log.write("{} {}\n".format(datetime.datetime.now(), log))
        file_log.close
    
    def save_img_to_com_hm_before(self):
        with open("Z:/img_before.png", "rb") as f:
            img_before = f.read()
        before_encode = base64.b64encode(img_before)
        before_encode = str(before_encode)
        before_replace = before_encode.replace("'", "")
        before_save_img = self.connect_db_hm2()
        before_save_img.execute("update show_picture_quality set line='HM2', create_at=getdate(), picture=? where id=3", before_replace)
        before_save_img.commit()
        
    def save_img_to_com_hm_after(self):
        with open("Z:/img_after.png", "rb") as f:
            img_after = f.read()
        after_encode = base64.b64encode(img_after)
        after_encode = str(after_encode)
        after_replace = after_encode.replace("'", "")
        after_save_img = self.connect_db_hm2()
        after_save_img.execute("update show_picture_quality set line='HM2', create_at=getdate(), picture=? where id=4", after_replace)
        after_save_img.commit()
        
########################################################################################################################################################
    
    def detection(self, values_class):

        # Model Object detection
        labelmap_path = 'D:/HM2/programs/object_detect_corn/labelmap.pbtxt'
        category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
        tf.keras.backend.clear_session()
        model = tf.saved_model.load('D:/HM2/programs/object_detect_corn/saved_model')

        print("...values class =",values_class)
        print("...STEP 2  ;  Object Detection Find Corn NG")
        self.log_file("STEP 2 ; Detect Object")
        loop_wait_detect = True
        screen_finish_success = False
        process_detect_success = False
        status_screen_finish = False
        count_loop_detect = 0
        while loop_wait_detect == True:
            ping_ip_main_process = self.ping_ip_hm()
            if ping_ip_main_process == True:
                main_connect_db = self.connect_db_hm2()
                try:
                    print("...Detection ; Check Screen Finish =", count_loop_detect)
                    main_connect_db.execute("SELECT * FROM SampingHM2 where id=1")
                    main_values_in_colum = main_connect_db.fetchall()
                    for loop_screen_finish in main_values_in_colum:
                        if loop_screen_finish[3] == 1:
                            print("...Detection ; Check Screen Finish Success")
                            screen_finish_success = True
                            loop_wait_detect = False
                            status_screen_finish = True
                            count_loop_detect = 0
                            break
                        else:
                            status_screen_finish = False
                except:
                    print("...Detection ; Not Select Values In Database !!!")
                    self.log_file("ERROR Detection ; Not Select Values In Database For Deetect Object")
                    pass
            count_loop_detect += 1
            if count_loop_detect == 10:
                print("...Detection ; Over Loop Count Screen Finish !!!")
                self.log_file("ERROR Detection ; Over Loop Count Screen Finish")
                break
            time.sleep(10)
        
        if screen_finish_success == True:
            # try:
            #     print("...Detection ; Record VDO For Dataset Empty")
            #     self.record_vdo_for_train("classifi")
            #     print("...Detection ; Record VDO Success")
            # except:
            #     print("...Detection ; Error Not Record VDO !!!")
            # self.crop_img("empty")
            # self.remove_vdo()

            for loop_detect in range(5):
                try:
                    time.sleep(10)
                    print("...Detection ; Wait Open Camera For Save Images")
                    cap = cv2.VideoCapture('rtsp://admin:admin@192.168.1.92:554/11')
                    while True:
                        _, frame = cap.read()
                        frame_crop = frame[230:580,340:1050,:]
                        frame_for_train = frame.copy()
                        frame_for_train = frame_for_train[250:480,460:900,:]
                        cv2.imwrite("D:/HM2/programs/picture_process/corn_detect.png", frame_crop)
                        cv2.imwrite("D:/HM2/programs/img_for_train/empty/empty_{}.png".format(self.time_strf()), frame_for_train)
                        img_detect = "D:/HM2/programs/picture_process/corn_detect.png"
                        print("...Detection ; Save Images Success")
                        image_np = load_image_into_numpy_array(img_detect)
                        output_dict = run_inference_for_single_image(model, image_np)
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            image_np,
                            output_dict['detection_boxes'],
                            output_dict['detection_classes'],
                            output_dict['detection_scores'],
                            category_index,
                            instance_masks=output_dict.get('detection_masks_reframed', None),
                            use_normalized_coordinates=True,
                            skip_scores=True,
                            min_score_thresh = 0.99,
                            line_thickness=2)
                        img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                        loop_detect = None
                        print("...Detection ; wait update pic_screen")
                        self.update_pic_screen(1)
                        print("...Detection ; Save Images And Detection Success")                   
                        cv2.imwrite("D:/HM2/programs/images/img_object/corn_object_%s.png" %self.time_strf(), img)
                        # process_detect_success = True
                        break
                    break
                except:
                    print("error detect not success")
                    self.log_file("ERROR Detection ; Detect Object Fails")
                    pass
    
            if loop_detect == 4:
                print("...Detection ; Not Open Camera For Save Images !!!")

            if process_detect_success == True:
                print("...Detection ; Count Score")
            values_detection = 0

            try:
                for i_score in output_dict['detection_scores']:
                    if i_score > 0.99:
                        values_detection += 1
                print("...Detection ; Corn NG =", values_detection)
            except:
                pass

            if values_detection == 1:
                cv2.imwrite("D:/HM2/programs/img_for_detect/img_for_detect_{}.png".format(self.time_strf()), frame_crop)
            elif values_detection >= 2:
                print("...Detection ; Values = NG")
                cv2.imwrite("D:/HM2/programs/img_for_detect/img_for_detect_{}.png".format(self.time_strf()), frame_crop)
                self.update_status(2)
            else:
                print("...Detection ; Values = OK")
                self.update_status(values_class)

            try:
                cv2.imwrite("D:/HM2/programs/img_for_show/img_after.png", img)   
                # self.save_img_to_com_hm_after()
            except:
                pass

            self.log_file("Find Corn In Picture = {}".format(values_detection))
        loop_wait_detect = False

        try:
            os.system("D:/HM2/programs/img_for_show/SaveImageHM12.exe")
        except:
            pass

        print("...Process Success")
        return

    def classifi(self):
        print("...STEP 1  ; Classification Raw Material")
        self.log_file("...STEP 1  ; Classification Raw Material")
        for loop_camera_classifi in range(5):
            try:
                print("...Classification ; Conect Camera For Classifi images ", loop_camera_classifi)
                cap = cv2.VideoCapture('rtsp://admin:admin@192.168.1.92:554/11')
                while True:
                    _, frame = cap.read()
                    frame_classifi = frame.copy()
                    frame_classifi_crop = frame.copy()
                    frame_classifi_crop = frame_classifi_crop[250:480,460:900,:] 
                    cv2.imwrite("D:/HM2/programs/picture_process/corn_classifi.png", frame)
                    status_pic1 = frame_classifi[250:480,460:900,:]   
                    im=cv2.resize(status_pic1,(img_width, img_height))
                    im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
                    y_pred = model.predict(im[np.newaxis])
                    y_pred_c=np.argmax(y_pred,axis=1)
                    y_pred_v=np.max(y_pred,axis=1)
                    softmax=tf.nn.softmax(y_pred[0]).numpy()
                    solitary=np.max(softmax)
                    print('...',classes[y_pred_c[0]] + str(softmax))
                    print("...Classification ; Classifi Success")
                    self.log_file("Classification ; Classifi Success")
                    break
                break
            except:
                self.log_file("ERROR ; Not Connect Camera For Classifi")
                
        stamp_tm = time.strftime("%d-" "%m-" "%Y"" " "%H." "%M." "%S")

        if classes[y_pred_c[0]] == "corn":
            self.log_file("Classifi corn")
            if solitary >= 0.40:
                name_corn = "corn"
            else:
                name_corn = "unknow"
            cv2.imwrite("D:/HM2/programs/images/corn/corn_%s.png" %self.time_strf(), frame)
            cv2.imwrite("D:/HM2/programs/img_for_train/corn/corn_{}.png".format(stamp_tm), frame_classifi_crop)
            cv2.imwrite("D:/HM2/programs/img_for_show/img_before.png", frame_classifi_crop)
            # self.save_img_to_com_hm_before()
            self.update_pic_sampling(1)
            # self.crop_img(name_corn)
            # self.remove_vdo()
            self.detection(1)

        elif classes[y_pred_c[0]] == "wg":
            self.log_file("Classifi wg")
            if solitary >= 0.40:
                name_wg = "wg"
            else:
                name_wg = "unknow"
            cv2.imwrite("D:/HM2/programs/images/wg/wg_%s.png" %self.time_strf(), frame)
            cv2.imwrite("D:/HM2/programs/img_for_train/wg/wg_{}.png".format(stamp_tm), frame_classifi_crop)
            cv2.imwrite("D:/HM2/programs/img_for_show/img_before.png", frame_classifi_crop)
            # self.save_img_to_com_hm_before()
            self.update_pic_sampling(1)
            # self.crop_img(name_wg)
            # self.remove_vdo()
            self.detection(3)

        elif classes[y_pred_c[0]] == "empty":
            self.log_file("Classifi empty")
            if solitary >= 0.40:
                name_empty = "empty"
            else:
                name_empty = "unknow"  
            cv2.imwrite("D:/HM2/programs/images/empty/empty_%s.png" %self.time_strf(), frame)
            cv2.imwrite("D:/HM2/programs/img_for_show/img_before.png", frame_classifi_crop)
            # self.save_img_to_com_hm_before()
            self.update_pic_sampling(1)
            # self.crop_img(name_empty)
            # self.remove_vdo()
            self.detection(1)
        return 

    def main_process(self):
        run_auto_train = False
        Hour_Now = None

        while True:
            check_run_process = self.select_run_process()
            print("...Auto Train =", run_auto_train)
            print("...DAY =", time.strftime("%a"))
            print("...Hour_Now =", Hour_Now)
            print("...Wait Sampling ", self.time_strf())
            print("...hm1 run process =".title(), check_run_process[0])
            print("...hm2 run process =".title(), check_run_process[1])
            ping_ip_main_process = self.ping_ip_hm()
            print("...Ping IP =", ping_ip_main_process)
            
            if ping_ip_main_process == True:
                main_connect_db = self.connect_db_hm2()
                try:
                    main_connect_db.execute("SELECT * FROM SampingHM2 where id=1")
                    main_values_in_colum = main_connect_db.fetchall()
                    for main_i in main_values_in_colum:
                        if main_i[1] == 1:
                            status_sampling_finish = True
                            break
                        else:
                            status_sampling_finish = False
                    print("...Status Sampling =", main_i[1])
                except:
                    self.log_file("not select values in database for sampling finish".title())
                    
                Hour_Now = time.strftime("%H")
                Hour_Now = int(Hour_Now)  
                if time.strftime("%a") == "Sun" and Hour_Now >= 0 and Hour_Now < 3 and run_auto_train == False:
                    print("...Train Model Classification")
                    train_model = self.update_train(1)
                    if train_model == True:
                        print("...Update Train Model Success")
                        time.sleep(2)
                        print("...Update Picture To Dataset")
                        self.update_picture_to_dataset()
                        print("...Update Picture Success")
                        time.sleep(2)
                        print("...Run Train Model")
                        os.system("D:/HM2/programs/auto_train_classifi_hm2.bat")
                        print("...Train Model Success")
                        train_model = self.update_train(0)
                        run_auto_train = True
                    elif train_model == False:
                        print("...Not Update Train Model")
                        run_auto_train = False  
                elif time.strftime("%a") != "Sun":
                    run_auto_train = False
                
                if main_i[1] == 2:
                    self.update_status(0)
                
                if main_i[1] == 1 and check_run_process[0] == False and check_run_process[1] == True:
                    # try:
                    #     self.record_vdo_for_train("classifi")
                    # except:
                    #     pass                       
                    try:
                        os.remove("D:/HM2/programs/picture_process/corn_classifi.png")
                        print("...Remove Picture Clssifi Success")
                    except:
                        print("...Not Remove Picture Classifi !!!")
                        pass
                    try:
                        os.remove("D:/HM2/programs/picture_process/corn_detect.png")
                        print("...Remove Picture Detection Success")
                    except:
                        print("...Not Remove Picture Detection !!!")
                        pass

                    self.classifi()
                    self.up_run_process(0)
                    self.log_file("------------------------------------------")
                    self.update_hm2(2)
                    self.update_sampling_finish(0)
                    self.update_pic_sampling(0)
                    self.update_pic_screen(0)
            
            print("********************************************")
            time.sleep(10)
                     
run_process = Auto_sampling_hm2()
run_process.main_process()


