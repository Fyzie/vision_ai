import os
import rfdetr

def train_rfdetr():
    # Ensure this directory contains 'train', 'valid' subfolders with '_annotations.coco.json'
    DATASET_DIR = "C:/Users/Hafizi/Documents/computer vision/07-powercable/powercables.v1" 
    OUTPUT_DIR = "C:/Users/Hafizi/Documents/computer vision/07-powercable/runs/nano1"
    
    NUM_CLASSES = 2 
    model = rfdetr.RFDETRSegNano(num_classes=NUM_CLASSES)
    
    model.train(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        device="cuda",
        val_interval=1,
        save_interval=10,
        early_stopping = True,
        early_stopping_patience = 10,
    )
    
    print(f"Training completed! Output checkpoints saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train_rfdetr()
