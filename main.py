import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import diff
from skimage.metrics import structural_similarity as compare_ssim


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

def scaleimage(img0):
    scale_percent = 40 # percent of original size
    width = int(img0.shape[1] * scale_percent / 100)
    height = int(img0.shape[0] * scale_percent / 100)
    dim = (width, height)
    img0 = cv2.resize(img0, dim, interpolation = cv2.INTER_AREA)
    return img0

def diffcurrentandpreviousframe(img0,imgb):
    grayimg0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    # framea = cv2.GaussianBlur(grayimg0, (21, 21), 0)
    cropframea = generatecrop(grayimg0)
        
    grayimgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
    # frameb = cv2.GaussianBlur(grayimgb, (21, 21), 0)
    cropframeb = generatecrop(grayimgb)
        
    frameDelta = cv2.absdiff(cropframea, cropframeb)

    # (score, frameDelta) = compare_ssim(cropframea, cropframeb, full=True)
    # frameDelta = (frameDelta * 255).astype("uint8")
    # print("SSIM: {}".format(score))
    kernel = np.ones((3,3), np.uint8)  
    thresh = cv2.threshold(frameDelta, 50, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = thresh/255
    return thresh

if __name__ == "__main__":
    cap = cv2.VideoCapture("video3.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    thresholdlist = []
    countwindow = 10
    count = 0
    sumval = 0
    captureflag = 1
    noo = 0
    x = 0
    while True:
        ret,img0 = cap.read()
        img0 = scaleimage(img0)

        ret,imgb = cap.read()
        imgb = scaleimage(imgb)
    
        thresh = diffcurrentandpreviousframe(img0,imgb)
        threshold = thresh.sum()


        img1 = generatecrop(img0)
        img2,countdots = countfire(img1)

        # Running average for countwindow consecutive frames
        if(count < countwindow):
            sumval = sumval + threshold
            count += 1
        else:
            sumval = 0
            count = 0

        # capturing frame once every seconds
        if(captureflag % int(fps) == 0):
            captureflag = 1
        else:
            captureflag+=1

        if(sumval > 100 and  noo < 5):
            noo += 1
        else:
            noo = 0
        
        if(noo == 5):
            for i in range(int(fps*4)):
                _,_ = cap.read()
            ret,frame0 = cap.read()
            frame0 = scaleimage(frame0)
            img1 = generatecrop(frame0)
            img2,countdots = countfire(img1)
            print("No of occurance",countdots)
        # thresholdlist.append(sum(prev))
        cv2.imshow("Frame", img2) #[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
        cv2.imshow("threshold", thresh)
        # plt.scatter(x,sumval)
        # plt.pause(0.05)
        
        if cv2.waitKey(1) == 27:
            break  # esc to quit
        x+=1
    cv2.destroyAllWindows()
    # plt.show()
    # plt.savefig('my_thresholdplot.png')
