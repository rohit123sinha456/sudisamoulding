import cv2
import numpy as np

def generatecrop(img):
    pts = np.array([[530,275],[490,300],[560,310],[590,280]])

    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()

    ## (2) make mask
    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ## (3) do bit-op
    image = cv2.bitwise_and(croped, croped, mask=mask)
    return image
    ## (4) add the white background
    # bg = np.ones_like(croped, np.uint8)*255
    # cv2.bitwise_not(bg,bg, mask=mask)
    # dst2 = bg+ dst  
def countfire(image): 
    blur = cv2.medianBlur(image, 5)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,210,255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    min_area = 0.1
    white_dots = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area:
            cv2.drawContours(image, [c], -1, (36, 255, 12), 2)
            white_dots.append(c)

    # print("White Dots count is:",len(white_dots))
    return(image,len(white_dots))

if __name__ == "__main__":
    cap = cv2.VideoCapture("video3.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    while True:
        ret,img0 = cap.read()
        scale_percent = 40 # percent of original size
        width = int(img0.shape[1] * scale_percent / 100)
        height = int(img0.shape[0] * scale_percent / 100)
        dim = (width, height)
        img0 = cv2.resize(img0, dim, interpolation = cv2.INTER_AREA)

        ret,imgb = cap.read()
        scale_percent = 40 # percent of original size
        width = int(imgb.shape[1] * scale_percent / 100)
        height = int(imgb.shape[0] * scale_percent / 100)
        dim = (width, height)
        imgb = cv2.resize(imgb, dim, interpolation = cv2.INTER_AREA)
        
        grayimg0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        framea = cv2.GaussianBlur(grayimg0, (21, 21), 0)
        cropframea = generatecrop(framea)
        
        grayimgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
        frameb = cv2.GaussianBlur(grayimgb, (21, 21), 0)
        cropframeb = generatecrop(frameb)
        
        frameDelta = cv2.absdiff(cropframea, cropframeb)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        threshold = thresh.sum()
        

        img1 = generatecrop(img0)
        img2,countdots = countfire(img1)
        cv2.imshow("Frame", img0) #[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
        cv2.imshow("Crop", img1) #[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
        if(threshold > 100):
            print("White Dots count is:",countdots)
        if cv2.waitKey(1) == 27:
            break  # esc to quit

    # img = cv2.imread('img4.png')
    # image = generatecrop(img)
    # image = countfire(image)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()