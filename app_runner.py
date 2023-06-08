import os
import cv2
import json
import dlib
import face_recognition
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

# Constants declaration
ROOT_DIR = './'
MODEL_LOCATION = "models"
GROUP_DIR = "model"
STATIC_DIR = "static"
TEST_IMAGE_FILENAME = "singleTestImage.jpg"
TEST_IMAGE_PATH = os.path.join(ROOT_DIR, STATIC_DIR, TEST_IMAGE_FILENAME)

# Load the trained model
with open(os.path.join(ROOT_DIR, MODEL_LOCATION, GROUP_DIR) + ".json", "r") as f:
    model_data = json.load(f)
    known_faces = model_data['known_faces']

@app.route('/')
def index():
    return render_template('index.html')


# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get the person's name from the form
        new_face_name = request.form['name']
        new_face_name = new_face_name.replace(" ", "_")
        # Create a folder for the new person's images
        new_face_folder = os.path.join(ROOT_DIR, GROUP_DIR, new_face_name)
        os.makedirs(new_face_folder, exist_ok=True)

        # Capture images from the camera
        camera = cv2.VideoCapture(0)  # Adjust the camera index if necessary

        for i in range(5):
            ret, frame = camera.read()
            if ret:
                image_path = os.path.join(new_face_folder, f"{new_face_name}_{i+1}.jpg")
                cv2.imwrite(image_path, frame)
                print(f"Image {i+1} captured!")
            else:
                print("Failed to capture image.")

        camera.release()

        # Rest of the code remains the same...


        # Process and store the captured images

        # Preprocess captured images using dlib and save them
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # Iterate over the images in the new_face_folder
        for filename in os.listdir(new_face_folder):
            # Check if the file is an image file
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                # Load the image
                img = cv2.imread(os.path.join(new_face_folder, filename))

                # Convert image to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Detect faces using dlib's detector
                faces = detector(gray, 1)

                # Iterate over each face
                for i, face in enumerate(faces):
                    # Get the landmarks/parts for the face
                    landmarks = predictor(gray, face)

                    # Extract face region as a numpy array
                    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                    face_region = img[y1:y2, x1:x2]

                    # Check if a face was detected
                    if face_region.size != 0:
                        # Convert face region to grayscale
                        face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

                        # Save the preprocessed image with the same file name
                        new_filename = f"{filename[:-4]}_{i}.png"
                        cv2.imwrite(os.path.join(new_face_folder, new_filename), face_gray)

                # Delete the old image
                os.remove(os.path.join(new_face_folder, filename))

        # Encode and add the new face to the known_faces dictionary
        new_face_encodings = []
        for filename in os.listdir(new_face_folder):
            image = face_recognition.load_image_file(os.path.join(new_face_folder, filename))
            encoding = face_recognition.face_encodings(image)
            if len(encoding) != 0:
                new_face_encodings.append(encoding[0].tolist())

        known_faces[new_face_name] = new_face_encodings

        # Save the updated known_faces dictionary to the model file
        with open(os.path.join(ROOT_DIR, MODEL_LOCATION, GROUP_DIR) + ".json", "w") as f:
            model_data['known_faces'] = known_faces
            json.dump(model_data, f)

        # Redirect back to the index page
        return redirect('/')
    else:
        return render_template('register.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    image_file = request.files['image']
    image_file.save(TEST_IMAGE_PATH)

    # Load the test image and extract its face encoding
    test_image = face_recognition.load_image_file(TEST_IMAGE_PATH)
    test_encoding = face_recognition.face_encodings(test_image)

    # Compare the test encoding with the known faces in the model
    if len(test_encoding) > 0:
        test_encoding = test_encoding[0]
        match_found = False  # Variable to track if a match is found
        for name, face_encodings in known_faces.items():
            matches = face_recognition.compare_faces(face_encodings, test_encoding, tolerance=0.4)
            if True in matches:
                return name
                match_found = True  # Set match_found to True if a match is found

        if not match_found:
            return "No match found in the model"
    else:
        return "No faces detected in the test image"

if __name__ == "__main__":
    app.run(host='localhost', port=5000)
