# importing the module
import cv2

# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):

	# checking for left mouse clicks
	if event == cv2.EVENT_LBUTTONDOWN:

		# displaying the coordinates
		# on the Shell
		print(x, ' ', y)

		# displaying the coordinates
		# on the image window
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img, str(x) + ',' +
					str(y), (x,y), font,
					1, (255, 0, 0), 2)
		cv2.imshow('image', img)

	# checking for right mouse clicks	
	if event==cv2.EVENT_RBUTTONDOWN:

		# displaying the coordinates
		# on the Shell
		print(x, ' ', y)

		# displaying the coordinates
		# on the image window
		font = cv2.FONT_HERSHEY_SIMPLEX
		b = img[y, x, 0]
		g = img[y, x, 1]
		r = img[y, x, 2]
		cv2.putText(img, str(b) + ',' +
					str(g) + ',' + str(r),
					(x,y), font, 1,
					(255, 255, 0), 1)
		cv2.imshow('image', img)

# driver function
if __name__=="__main__":
    cap = cv2.VideoCapture("video3.mp4")
    while True:
        ret,img = cap.read()
        scale_percent = 40 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("image", img) #[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
        cv2.setMouseCallback('image', click_event)
        cv2.waitKey(0)
        if cv2.waitKey(1) == 27:
            break  # esc to quit

    cv2.destroyAllWindows()

	# # reading the image
	# img = cv2.imread('img2.png', 1)

	# # displaying the image
	# cv2.imshow('image', img)

	# # setting mouse handler for the image
	# # and calling the click_event() function
	# cv2.setMouseCallback('image', click_event)

	# # wait for a key to be pressed to exit
	# cv2.waitKey(0)

	# # close the window
	# cv2.destroyAllWindows()
