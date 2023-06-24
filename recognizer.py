# importing libraries
import math
import os
import os.path
import pickle
import shutil

import dlib
import face_recognition
import requests
from PIL import Image, ImageDraw
from sklearn import neighbors

# CLASS FACE RECOGNIZER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


class FaceRecognizer:
    def __init__(self, predictor_path, train_data):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.knn_clf = None
        self.train_data = train_data


    def train(self, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
        X = []
        y = []

        # Loop through each item in the training data
        for item in self.train_data["training_data"]:
            img_url = item["image_path"]
            label = item["label"]

            # Download the image and save it locally
            local_path = "temp_train_image.jpg"  # Choose a local path to save the image
            response = requests.get(img_url)
            with open(local_path, "wb") as f:
                f.write(response.content)

            # Load image file and find face locations

            image = face_recognition.load_image_file(local_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if not face_bounding_boxes:
                # If there are no faces (or too many faces) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_url,
                                                                          "Didn't find a face" if len(
                                                                              face_bounding_boxes) < 1 else
                                                                          "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(
                    face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(label)

        # Determine how many neighbors to use for weighting in the KNN classifier
        if n_neighbors is None:
            n_neighbors = int(round(math.sqrt(len(X))))
            if verbose:
                print("Chose n_neighbors automatically:", n_neighbors)

        print(X, y)
        # Create and train the KNN classifier
        self.knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
        self.knn_clf.fit(X, y)

        # Save the trained KNN classifier
        if model_save_path is not None:
            with open(model_save_path, 'wb') as f:
                pickle.dump(self.knn_clf, f)

    def predict_batch(self, X_img_paths, model_path=None, distance_threshold=0.6):
        if self.knn_clf is None and model_path is None:
            raise Exception("Must supply knn classifier either through knn_clf or model_path")

        # Load a trained KNN model (if one was passed in)
        if self.knn_clf is None:
            with open(model_path, 'rb') as f:
                self.knn_clf = pickle.load(f)

        predictions_batch = []
        for X_img_path in X_img_paths:
            local_path = "temp_image.jpg"  # Choose a local path to save the image
            response = requests.get(X_img_path)
            with open(local_path, "wb") as f:
                f.write(response.content)

            # Load image file and find face locations
            X_img = face_recognition.load_image_file(local_path)
            X_face_locations = face_recognition.face_locations(X_img)
            # If no faces are found in the image, add an empty result to the predictions batch.
            if len(X_face_locations) == 0:
                predictions_batch.append([])
                continue

            # Find encodings for faces in the test image
            faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)
            # Use the KNN model to find the best matches for the test faces
            closest_distances = self.knn_clf.kneighbors(faces_encodings, n_neighbors=1)
            proba = self.knn_clf.predict_proba(faces_encodings)
            print("probability of a person", proba)
            print("closest_distances", closest_distances)

            are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

            # Predict classes and remove classifications that aren't within the threshold
            predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                           zip(self.knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
            predictions_batch.append(predictions)

        return predictions_batch

    def show_prediction_labels_on_image(self, img_path, predictions):
        pil_image = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(pil_image)

        for name, (top, right, bottom, left) in predictions:
            # Draw a box around the face using the Pillow module
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

            # There's a bug in Pillow where it blows up with non-UTF-8 text
            # when using the default bitmap font
            name = name.encode("UTF-8")

            # Draw a label with a name below the face
            text_width, text_height = draw.textsize(name)
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

        # Remove the drawing library from memory as per the Pillow docs
        del draw

        # Display the resulting image
        pil_image.show()

    def save_to_database(self, name, image_path):
        # Create a subdirectory with the given name
        subdirectory_path = os.path.join(self.train_dir, name)
        os.makedirs(subdirectory_path, exist_ok=True)

        # Move the image to the subdirectory
        image_name = os.path.basename(image_path)
        destination_path = os.path.join(subdirectory_path, image_name)
        shutil.copy(image_path, destination_path)

        print("Image '{}' saved to the database in subdirectory '{}'.".format(image_name, name))


# CREATING AN INSTANCE

if __name__ == "__main__":
    predictor_path = "PATH OF PREDICTOR"
    train_data = {
        "training_data": [
            {
                "image_path": "PATH TO PERSON 1 IMAGE 1",
                "label": "Person 1"
            },
            {
                "image_path": "PATH TO PERSON 1 IMAGE 2",
                "label": "Person 1"
            },
            {
                "image_path": "PATH TO PERSON 2 IMAGE 1",
                "label": "Person 2"
            },
            ...
        ]
    }
    recognizer = FaceRecognizer(predictor_path, train_data)

    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    print("Training KNN classifier...")
    recognizer.train(model_save_path="PATH OF SAVED MODEL", n_neighbors=2)
    print("Training complete!")

    # STEP 2: Using the trained classifier, make predictions for unknown images
    test_image_paths = ["PATH OF IMAGE 1", "PATH OF IMAGE 2", ...]  # Add paths of images to perform recognition on
    predictions_batch = recognizer.predict_batch(test_image_paths)

    # Print results for each image
    for i, image_path in enumerate(test_image_paths):
        print("Predictions for image", i + 1)
        predictions = predictions_batch[i]
        for name, (top, right, bottom, left) in predictions:
            # Display the image with prediction labels
            recognizer.show_prediction_labels_on_image(image_path, predictions)
            if name == "unknown":
                user_name = input("Enter your name for the unknown person: ")
                recognizer.save_to_database(user_name, image_path)
                name = user_name
