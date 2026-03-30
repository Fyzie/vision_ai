import rfdetr
from rfdetr.detr import RFDETR
from helpers import setup_training, collage_predictions, custom_config

def main(**kwargs):
    session = setup_training(kwargs)

    # training parameters should modified @ helpers.custom_config.py (ctrl+click the module at the top)
    # for live monitoring, run on terminal env: tensorboard --logdir="path/to/session_dir"
    session.model.train(
            dataset_dir=session.dataset_dir,
            output_dir=session.session_dir,
            **session.train_cfg.__dict__
        )
    
    collage_predictions(
        session.dataset_dir, 
        session.session_dir, 
        session.model_type
    )

if __name__ == "__main__":
    CONFIG = {
        "model_type" : "RFDETRSegLarge",
        "dataset_dir": "C:/Users/MachineLearning/Documents/Datasets/Lens/lens_station2.v22.combined_polar_unpolar2_sliced_overlap",
        "output_dir" : "C:/Users/MachineLearning/Documents/Weight/rfdetr/trials",
    }
    main(**CONFIG)
