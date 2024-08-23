from imutils import face_utils
import dlib
import cv2

# Initialize the facial detector (HOG-based) and the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# Start the video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, image = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Loop over each face detected
    for (i, rect) in enumerate(rects):
        # Predict facial landmarks and convert them to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Draw each landmark point on the image with different colors
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Green for the landmarks

    # Display the image with the landmarks
    cv2.imshow("Output", image)

    # Break the loop if the 'Esc' key is pressed
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()