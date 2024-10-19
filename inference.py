import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from model import get_model, convert_prediction, getYUVColorSpace, normalizeValue
from tqdm import tqdm
from itertools import combinations

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
    
    # Generate all unique pairs of indices without repetition
    pairs = list(combinations(range(len(exposure_levels)), 2))
    print(pairs)
    
    # Load the weights into the model (if not already loaded)
    # model.load_weights(model_weights_path)  # Already loaded above
    
    # Loop through each pair and process
    for idx1, idx2 in pairs:
        img1_name = os.path.splitext(exposure_levels[idx1])[0]
        img2_name = os.path.splitext(exposure_levels[idx2])[0]
        
        y_low = y_channels[idx1]
        uv_low = uv_channels[idx1]
        y_over = y_channels[idx2]
        uv_over = uv_channels[idx2]
        
        # Run inference
        y_pred, uv_pred = model.predict([y_low, uv_low, y_over, uv_over])
        
        # Convert prediction to RGB
        rgb_pred = convert_prediction(y_pred, uv_pred)
        
        # Resize the output to original image size if necessary
        rgb_pred = cv2.resize(rgb_pred, (WIDTH, HEIGHT))
        
        # Create a descriptive output filename
        output_filename = f'fused_{img1_name}_{img2_name}.jpg'
        output_path = os.path.join(output_folder, output_filename)
        
        # Save the fused image
        cv2.imwrite(output_path, rgb_pred)
        print(f"Fused image saved at: {output_path}")

if __name__ == '__main__':
    main()
