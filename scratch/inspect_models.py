
import os
import tensorflow as tf

def inspect_model_full(model_path):
    print(f"\n--- Inspecting: {model_path} ---")
    if not os.path.exists(model_path):
        print("File not found.")
        return
    try:
        model = tf.keras.models.load_model(model_path)
        # Architecture summary
        model.summary()
        
        # Optimizer and training params
        if hasattr(model, 'optimizer') and model.optimizer is not None:
            print("\nOptimizer Config:")
            opt_config = model.optimizer.get_config()
            for k, v in opt_config.items():
                print(f"  {k}: {v}")
            
            # Try to get learning rate
            lr = getattr(model.optimizer, 'learning_rate', None)
            if lr is not None:
                if hasattr(lr, 'numpy'):
                    print(f"  Current Learning Rate: {lr.numpy()}")
                else:
                    print(f"  Current Learning Rate: {lr}")
        
        # Loss
        print(f"\nLoss: {model.loss}")
        
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    inspect_model_full("artifacts/models/v2/final_model.keras")
    inspect_model_full("artifacts/models/v3/final_model.keras")
