import logging
import time
import random
import requests

from .constants import classes, landing_statuses
from .detected_object import DetectedObject
from .detected_translation import DetectedTranslation

from predict.main_vo import TrackerModel

from ultralytics import YOLO
import time

TASIT = 0
INSAN = 1
UAP = 2
UAI = 3

class ObjectDetectionModel:
    # Takımların modelleri icin tanimlanmis sinif
    

    def __init__(self, evaluation_server_url, translation):
        logging.info('Created Object Detection Model')
        self.evaulation_server = evaluation_server_url
        # Modelinizi bu kısımda init edebilirsiniz.
        self.model = self.getYoloModel()# Örnektir!
        self.TrackerModel = TrackerModel(translation)
        self.xliste=[]
        self.yliste=[]
        self.xCounter=0
        self.yCounter=0
        self.lastx=self.TrackerModel.TRANSLATION_X
        self.lasty=self.TrackerModel.TRANSLATION_Y
        self.realx=[]
        self.realy=[]

    @staticmethod
    def download_image(img_url, images_folder):
        t1 = time.perf_counter()
        img_bytes = requests.get(img_url).content
        image_name = img_url.split("/")[-1]  # frame_x.jpg
        with open(images_folder + image_name, 'wb') as img_file:
            img_file.write(img_bytes)

        t2 = time.perf_counter()

        logging.info(f'{img_url} - Download Finished in {t2 - t1} seconds to {images_folder + image_name}')
    
    def process(self, prediction,evaluation_server_url,health_status):
        # Yarışmacılar resim indirme, pre ve post process vb işlemlerini burada gerçekleştirebilir.
        # Download image (Example)
        self.download_image(evaluation_server_url + "media" + prediction.image_url, "./_images/")
        # Örnek: Burada OpenCV gibi bir tool ile preprocessing işlemi yapılabilir. (Tercihe Bağlı)
        # ...
        # Nesne tespiti ve pozisyon kestirim modelinin bulunduğu fonksiyonun (self.detect() ) çağırılması burada olmalıdır.
        frame_results = self.detect(prediction,health_status)
        # Tahminler objesi FramePrediction sınıfında return olarak dönülmelidir.
        return frame_results

    def detect(self, prediction,health_status):
        # Modelinizle bu fonksiyon içerisinde tahmin yapınız.
        image_path = "C:\\Users\\user\\Downloads\\HanSlam3\\_images\\" + prediction.image_url.split("/")[-1]
        print(image_path)
        results = self.evaluate(image_path) # Örnektir.
        
        # Burada örnek olması amacıyla 2 adet tahmin yapıldığı simüle edilmiştir.
        # Yarışma esnasında modelin tahmin olarak ürettiği sonuçlar kullanılmalıdır.
        # Örneğin :
        # for i in results: # gibi
        for result in results:
            
            d_obj = DetectedObject( result["cls"],
                                    result["landing_status"],
                                    result["top_left_x"],
                                    result["top_left_y"],
                                    result["bottom_right_x"],
                                    result["bottom_right_y"]
                                        )

            # Modelin tahmin ettiği her nesne prediction nesnesi içerisinde bulunan detected_objects listesine eklenmelidir.
            prediction.add_detected_object(d_obj)

        # Health Status biti hava aracinin uydu haberlesmesinin saglikli olup olmadigini gosterir.
        # Health Status 0 ise sistem calismali 1 ise gelen verinin aynisini gondermelidir.

        
        result = self.TrackerModel.path_tracker(image_path)
        print("Tracking Result: ", result)
        self.xliste.append(result["y"])
        self.yliste.append(result["x"])
        self.realx.append(round(float(prediction.gt_translation_x),2))
        self.realy.append(round(float(prediction.gt_translation_y),2))
        print(health_status)
        
        if health_status == '0':
            # Takimlar buraya kendi gelistirdikleri algoritmalarin sonuclarini entegre edebilirler.
            pred_translation_x = result["y"] # Ornek olmasi icin rastgele degerler atanmistir takimlar kendi sonuclarini kullanmalidirlar.
            pred_translation_y = result["x"]# Ornek olmasi icin rastgele degerler atanmistir takimlar kendi sonuclarini kullanmalidirlar.
        else :
            pred_translation_x = prediction.gt_translation_x
            pred_translation_y = prediction.gt_translation_y
        print(pred_translation_x)
        print(pred_translation_y)
        # Translation icin hesaplanilan degerleri sunucuya gondermek icin ilgili objeye dolduralim.
        try:
            with open('translation.txt', 'a') as f:
                f.write("Tracking CONSTS: "+ self.TrackerModel.TRANSLATION_X +" "+ self.TrackerModel.TRANSLATION_Y+ "\n")
                f.write("Tracking Result: " + result["x"]+result["y"]+"\n")
        except Exception as e:
            pass
        trans_obj = DetectedTranslation(pred_translation_x, pred_translation_y)
        prediction.add_translation_object(trans_obj)

        return prediction
    
    @staticmethod
    def getYoloModel():
        model = YOLO("C:\\Users\\user\\Downloads\\HanSlam3\\HanSlam-test\\HanSlam-test\\best (18).pt")
        return model
    
    
    def evaluate(self, imageURL):
        results = self.model.predict(source=imageURL,imgsz=640, conf=0.5)
        
        if(len(results) == 0):
            return None
        count = 0
        if (len(results) > 1):
            count += 1
            print(len(results), count)
        results = results[0]
        new_results = []
            
        boxes = results.boxes.xyxy
        scores = results.boxes.conf
        clsses = results.boxes.cls
        
        
        humans = [box for box, cls in zip(boxes, clsses) if cls == INSAN]
        landStat = landing_statuses["Inis Alani Degil"] # Inis Alani Degil
        for box, score, cls in zip(boxes, scores, clsses):
            new_result = {}
            cls = int(cls.item())
            top_left_x, top_left_y, bottom_right_x, bottom_right_y  = map(float,box)
            if(cls == UAP or cls == UAI):
                landStat = landing_statuses["Inilebilir"] # Inilebilir
                for human_box in humans:
                    if(self.check_intersect(box, human_box)):
                        landStat = landing_statuses["Inilemez"] # Inilemez
                        break
                    
            new_result["cls"] = cls   
            new_result["landing_status"] = landStat
            new_result["top_left_x"] = top_left_x
            new_result["top_left_y"] = top_left_y
            new_result["bottom_right_x"] = bottom_right_x
            new_result["bottom_right_y"] = bottom_right_y
        
            new_results.append(new_result)
        return new_results
    
    
    def check_intersect(self,box1,box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        intersectX = False
        intersectY = False
        
        if(x1 < x4 and x2 > x3):
            intersectX = True
        elif(x1>x4 and x2<x3):
            intersectX = True
            
        if (y1 < y4 and y2 > y3):
            intersectY = True
        elif(y1>y4 and y2<y3):
            intersectY = True
        
        return intersectX and intersectY
        