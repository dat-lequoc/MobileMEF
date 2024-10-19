import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from model import get_model, convert_prediction, getYUVColorSpace, normalizeValue
from tqdm import tqdm

def load_and_preprocess_image(path, height, width):
    """
    Load an image from the specified path and preprocess it.
    
    Args:
        path (str): Path to the image.
        height (int): Desired image height.
        width (int): Desired image width.
    
    Returns:
        y (np.ndarray): Y channel of the image.
        uv (np.ndarray): UV channels of the image.
    """
    img_bgr = cv2.imread(path)
    img_bgr = cv2.resize(img_bgr, (width, height))
    
    y, u, v = getYUVColorSpace(img_bgr)
    
    y = normalizeValue(y)
    
    u_temp_half = cv2.resize(u, (width//4, height//4))
    v_temp_half = cv2.resize(v, (width//4, height//4))
    
    u = normalizeValue(u_temp_half)
    v = normalizeValue(v_temp_half)
    uv = np.stack([u, v], axis=-1).astype('float32')
    
    return y[np.newaxis,...,np.newaxis], uv[np.newaxis,...,np.newaxis]

def main():
    # Configuration
    model_weights_path = './h5/sice_ev_most.h5'  # Choose the appropriate weights
    input_folder = './inputs'                     # Folder containing 1.jpg to 5.jpg
    output_folder = './fused_results'             # Folder to save fused images
    os.makedirs(output_folder, exist_ok=True)
    
    # Define image dimensions (ensure these match your model's expected input size)
    HEIGHT = 2816
    WIDTH = 4096
    
    # Load the model
    model = get_model(shape=(HEIGHT, WIDTH), batch_size=1, resize_output=True)
    model.load_weights(model_weights_path)
    print("Model loaded successfully.")
    
    # Load all five input images
    exposure_levels = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg']
    input_paths = [os.path.join(input_folder, img) for img in exposure_levels]
    
    # Check if all input images exist
    for path in input_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input image not found: {path}")
    
    # Preprocess all input images
    y_channels = []
    uv_channels = []
    for path in input_paths:
        y, uv = load_and_preprocess_image(path, HEIGHT, WIDTH)
        y_channels.append(y)
        uv_channels.append(uv)
    
    # Stack Y and UV channels
    # Assuming the model can handle five exposures by concatenating their channels
    # This may require modifying the model architecture to accept more inputs
    # Here, we concatenate along the channel axis
    y_stack = np.concatenate(y_channels, axis=-1)  # Shape: (1, H, W, K)
    uv_stack = np.concatenate(uv_channels, axis=-1)  # Shape: (1, H//4, W//4, K*2)
    
    # Prepare inputs for the model
    # The existing model expects four inputs: y_low, uv_low, y_med, uv_med
    # To handle five exposures, you might need to modify the model to accept more inputs
    # For demonstration, we'll select the most under and over-exposed images
    # Alternatively, adapt the model to handle all five exposures
    y_low = y_channels[0]  # Assuming '1.jpg' is the most under-exposed
    uv_low = uv_channels[0]
    y_over = y_channels[-1]  # Assuming '5.jpg' is the most over-exposed
    uv_over = uv_channels[-1]
    
    # Load the weights into the model
    model.load_weights(model_weights_path)
    
    # Run inference
    y_pred, uv_pred = model.predict([y_low, uv_low, y_over, uv_over])
    
    # Convert prediction to RGB
    rgb_pred = convert_prediction(y_pred, uv_pred)
    
    # Resize the output to original image size if necessary
    rgb_pred = cv2.resize(rgb_pred, (WIDTH, HEIGHT))
    
    # Save the fused image
    output_path = os.path.join(output_folder, 'fused_output.jpg')
    cv2.imwrite(output_path, rgb_pred)
    print(f"Fused image saved at: {output_path}")

if __name__ == '__main__':
    main()
