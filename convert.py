import argparse
import tensorflow as tf
import numpy as np
from PIL import Image
import os

def representative_dataset_generator(images_folder, input_size, max_images=200):
    # representative dataset generation
    images = sorted([f for f in os.listdir(images_folder) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"Found {len(images)} images for calibration")
    print(f"Using {min(max_images, len(images))} images for quantization calibration")
    
    for file_name in images[:max_images]:
        try:
            path = os.path.join(images_folder, file_name)
            img = Image.open(path).convert("RGB")
            img = img.resize(tuple(input_size))
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            yield [img_array]
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

def convert_to_fp32_tflite(keras_model_path, output_path):
    # keras --> fp32 tflite
    print(f"Converting {keras_model_path} to FP32 TFLite...")

    model = tf.keras.models.load_model(keras_model_path, compile=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"FP32 TFLite model saved at: {output_path}")

def convert_to_quantized_tflite(keras_model_path, output_path, quant_type, 
                               calibration_images_folder, input_size, max_calibration_images):
    #keras --> quantized int8 or uint8
    print(f"Converting {keras_model_path} to {quant_type.upper()} TFLite...")

    model = tf.keras.models.load_model(keras_model_path, compile=False)
    
    # create representative dataset
    def representative_dataset():
        return representative_dataset_generator(
            calibration_images_folder, input_size, max_calibration_images
        )
    
    # convert to TFLite with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    if quant_type.lower() == 'int8':
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    elif quant_type.lower() == 'uint8':
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    else:
        raise ValueError(f"Unsupported quantization type: {quant_type}")
    
    tflite_model = converter.convert()
    
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"{quant_type.upper()} TFLite model saved at: {output_path}")

def get_model_size_mb(model_path):
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def main(args):
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    original_size = get_model_size_mb(args.model_path)
    print(f"Original Keras model size: {original_size:.2f} MB")
    
    if args.quant_type.lower() == 'fp32':
        convert_to_fp32_tflite(args.model_path, args.output_path)
        
    elif args.quant_type.lower() in ['int8', 'uint8']:
        if not os.path.exists(args.calibration_images):
            raise FileNotFoundError(f"Calibration images folder not found: {args.calibration_images}")
        
        convert_to_quantized_tflite(
            args.model_path, 
            args.output_path, 
            args.quant_type,
            args.calibration_images,
            args.input_size,
            args.max_calibration_images
        )
    
    else:
        raise ValueError(f"Unsupported quantization type: {args.quant_type}")
    
    converted_size = get_model_size_mb(args.output_path)
    compression_ratio = original_size / converted_size
    
    print(f"Converted model size: {converted_size:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print("Conversion completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified conversion script for Keras to TFLite models")
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the Keras model file (.keras or .h5)")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output path for the TFLite model (.tflite)")
    
    # Quantization arguments
    parser.add_argument("--quant_type", type=str, choices=['fp32', 'int8', 'uint8'], 
                       default='fp32', help="Quantization type. Default: fp32")
    parser.add_argument("--calibration_images", type=str, 
                       default="./vggface/val",
                       help="Path to calibration images folder (for quantization). Default: ./vggface/val")
    parser.add_argument("--input_size", type=int, nargs=2, default=[128, 128],
                       help="Model input size (height width). Default: 128 128")
    parser.add_argument("--max_calibration_images", type=int, default=200,
                       help="Maximum number of calibration images. Default: 200")
    
    args = parser.parse_args()
    main(args)

