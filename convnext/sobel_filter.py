import numpy as np
import cv2


if __name__ == '__main__':
 
    image = cv2.imread('./0087_1_1.jpg')
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smoothened = cv2.GaussianBlur(img_gray, (3,3), 0)
    # sobel_xy = cv2.Sobel(smoothened, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    # cv2.imshow('Sobel X Y using Sobel() function', sobel_xy)
    # cv2.waitKey(0)

    grad_x = cv2.Sobel(smoothened, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(smoothened, cv2.CV_64F, 0, 1)
    grad = np.sqrt(grad_x**2 + grad_y**2) # normalize
    grad_norm = (grad * 255 / grad.max()).astype(np.uint8)
    cv2.imshow('Edges', grad_norm)
    cv2.waitKey(0)
    cv2.imwrite('./cow_edges.png',grad_norm)