# BaseForest2

This project is a satellite image classification system designed to detect deforestation. It uses a Convolutional Neural Network (CNN) to classify whether a given satellite image tile shows a deforested area or not. The project also includes a web interface built with FastAPI for users to upload an image and get a prediction.

-----

## Features

  * **Deforestation Classification**: The core of the project is a CNN model that can classify satellite images into 'deforested' and 'non-deforested' classes.
  * **Web Interface**: A user-friendly web interface allows users to upload an image and get a prediction from the trained model. The interface is built with FastAPI and uses Jinja2 for templating.
  * **Training Pipeline**: The project includes a complete training pipeline. You can train the model from scratch using the provided dataset.
  * **Inference Script**: A script is available to run predictions on single images.

-----

## Dataset

The dataset used for this project consists of satellite images categorized into 'deforested' and 'non-deforested' areas. The data is split into training, validation, and test sets. The images are in JPEG format.

  * **Training data**: Used to train the model.
  * **Validation data**: Used to evaluate the model during training.
  * **Test data**: Used to test the final performance of the model.

-----

## Installation

To set up the project, follow these steps:

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/1ndrayu/baseForest2.git
    ```

2.  **Create a virtual environment and install dependencies**:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    pip install -r requirements.txt
    ```

-----

## Usage

There are two main ways to use this project: through the web app or by using the training script.

### Web App

To start the web app, run the following command:

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

This will start a local server, and you can access the web interface by navigating to `http://127.0.0.1:8000` in your web browser. From there, you can upload a satellite image to get a prediction.

### Model Training

To train the model, you can use the `train.py` script. The script takes the following arguments:

  * `--data`: Path to the data folder. Default is `data`.
  * `--epochs`: Number of epochs to train for. Default is 3.
  * `--batch`: Batch size. Default is 32.
  * `--out`: Path to save the trained model. Default is `model.pth`.

Example of how to run the training script:

```bash
python train.py --epochs 10 --batch 64
```

-----

## Technologies Used

  * **Python**: The main programming language used in the project.
  * **PyTorch**: The deep learning framework used to build and train the CNN model.
  * **FastAPI**: A modern, fast (high-performance) web framework for building APIs with Python.
  * **Uvicorn**: A lightning-fast ASGI server, used to run the FastAPI application.
  * **Pillow**: A powerful image processing library for Python.
  * **scikit-learn**: A machine learning library used for label encoding.
  * **Jinja2**: A templating engine for Python, used for rendering the HTML pages.
  * **Pandas**: A data analysis and manipulation library.

-----

## File Structure

```
.
├── .gitattributes
├── .gitignore
├── README.md
├── app.py
├── data
│   └── deforestation dataset
│       ├── test data
│       ├── train data
│       └── val data
├── dataset.py
├── infer.py
├── model.py
├── requirements.txt
├── static
│   └── style.css
├── templates
│   ├── index.html
│   └── result.html
├── tests
│   └── test_imports.py
└── train.py
```
