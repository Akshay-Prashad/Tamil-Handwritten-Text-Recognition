import cv2
import numpy as np
from scipy import ndimage
from torchvision import transforms

class Preprocessor(object):
    def __call__(self, img):
        img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        inverted = cv2.bitwise_not(img_gray)
        kernel = np.ones((5, 5), np.uint8)
        thick = cv2.dilate(inverted, kernel, iterations=1)
        
        h, w = thick.shape
        ratio = 48.0 / max(h, w)
        new_size = (int(round(w * ratio)), int(round(h * ratio)))
        res = cv2.resize(thick, new_size, interpolation=cv2.INTER_LANCZOS4)
        
        arr = np.asarray(res)
        com = ndimage.measurements.center_of_mass(arr)
        
        result = np.full((64, 64), 255, dtype=np.uint8)
        y_offset = int(round(32.0 - com[0]))
        x_offset = int(round(32.0 - com[1]))
        
        y1 = max(0, y_offset)
        y2 = min(64, y_offset + res.shape[0])
        x1 = max(0, x_offset)
        x2 = min(64, x_offset + res.shape[1])

        result[y1:y2, x1:x2] = res[max(0, -y_offset):min(res.shape[0], 64 - y_offset), 
                                   max(0, -x_offset):min(res.shape[1], 64 - x_offset)]
        
        return result

    def get_transform(self):
        return transforms.Compose([
            transforms.Lambda(self.__call__),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])