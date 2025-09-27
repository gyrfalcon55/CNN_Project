# model_dev.py

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# This function should be called once when the app starts.
def load_dependencies(model_path="models/classifier_2_acc_85.h5"):
    """Loads the model and class names from disk."""
    print("Loading model and class names...")
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Load the class names
    
    class_names = ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra']
    print("Loading complete")
    return model, class_names

# This function will now process the image bytes directly
def predict_from_bytes(model, class_names, image_bytes):
    """
    Takes a loaded model, class names, and image bytes, 
    and returns the predicted class name.
    """
    # Load the image from bytes
    img = Image.open(io.BytesIO(image_bytes))
    # Ensure image is in RGB
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((224, 224))

    # Convert to array and preprocess
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make prediction
    preds = model.predict(x)
    pred_class_index = np.argmax(preds[0])
    
    return class_names[pred_class_index]

# This is the correct way to add a test block
if __name__ == "__main__":
    # This code will only run if you execute "python model_dev.py" directly
    # It will NOT run when imported by main.py
    print("Running a test prediction...")
    test_image_path = "path/to/your/test_image.jpg" # <--- Change this to a valid image path
    loaded_model, names = load_dependencies()
    
    # To test predict_from_bytes, you'd first need to read the image into bytes
    with open(test_image_path, "rb") as f:
        bytes_data = f.read()
    
    prediction = predict_from_bytes(loaded_model, names, bytes_data)
    print(f"The predicted class for {test_image_path} is: {prediction}")