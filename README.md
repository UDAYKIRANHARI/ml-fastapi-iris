# 🌸 Iris Classifier API

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 👉 [**TRY THE LIVE INTERACTIVE UI HERE**](https://uday0715-ml-fastapi-iris.hf.space/ui) 👈

## 🚀 Live Deployment

**Try it now**: [https://uday0715-ml-fastapi-iris.hf.space](https://uday0715-ml-fastapi-iris.hf.space)

- **🌐 Interactive UI**: [https://uday0715-ml-fastapi-iris.hf.space/ui](https://uday0715-ml-fastapi-iris.hf.space/ui)
- - **📚 API Documentation**: [https://uday0715-ml-fastapi-iris.hf.space/docs](https://uday0715-ml-fastapi-iris.hf.space/docs)
  - - **🔬 API Testing**: [https://uday0715-ml-fastapi-iris.hf.space/redoc](https://uday0715-ml-fastapi-iris.hf.space/redoc)

A production-ready Machine Learning REST API built with FastAPI that classifies Iris flower species using a trained Logistic Regression model. Features a clean web UI for interactive predictions and comprehensive API documentation.

## 🎯 Features

- **RESTful API**: Built with FastAPI for high performance and automatic OpenAPI documentation
- **Machine Learning**: Logistic Regression model trained on the classic Iris dataset
- **Interactive UI**: User-friendly HTML interface for real-time predictions
- **Model Persistence**: Trained model saved with joblib for efficient loading
- **CORS Enabled**: Supports cross-origin requests for frontend integration
- **Probability Outputs**: Returns confidence scores for all three species

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/UDAYKIRANHARI/ml-fastapi-iris.git
cd ml-fastapi-iris
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the model**
```bash
python train_model.py
```
This will create a `models/` directory and save the trained model as `iris_model.joblib`.

4. **Start the API server**
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## 📖 Usage

### Interactive Web UI

Visit `http://localhost:8000/ui` to access the web interface where you can:
- Input flower measurements using sliders or text fields
- Get instant predictions with probability scores
- Visualize the classification results

### API Endpoints

#### 1. Health Check
```http
GET /
```
Returns API status and available Iris species.

**Response:**
```json
{
  "message": "Iris Classifier API is running.",
  "model_loaded": true,
  "target_names": ["setosa", "versicolor", "virginica"]
}
```

#### 2. Make Predictions
```http
POST /predict
```

**Request Body:**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response:**
```json
{
  "predicted_class": "setosa",
  "predicted_class_index": 0,
  "class_probabilities": {
    "setosa": 0.97,
    "versicolor": 0.02,
    "virginica": 0.01
  }
}
```

### API Documentation

FastAPI automatically generates interactive API documentation:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🗂️ Project Structure

```
ml-fastapi-iris/
├── main.py              # FastAPI application and endpoints
├── train_model.py       # Model training script
├── iris_ui.html         # Interactive web interface
├── requirements.txt     # Python dependencies
├── models/              # Trained model storage (generated)
│   └── iris_model.joblib
└── README.md            # Project documentation
```

## 🧪 Model Details

- **Algorithm**: Logistic Regression (max_iter=200)
- **Dataset**: Scikit-learn's built-in Iris dataset
- **Features**: 4 measurements (sepal length/width, petal length/width)
- **Classes**: 3 Iris species (Setosa, Versicolor, Virginica)
- **Train/Test Split**: 80/20 split with random_state=42
- **Performance**: ~97% accuracy on test set

## 🛠️ Technologies Used

- **[FastAPI](https://fastapi.tiangolo.com/)**: Modern, fast web framework for building APIs
- **[scikit-learn](https://scikit-learn.org/)**: Machine learning library for model training
- **[Uvicorn](https://www.uvicorn.org/)**: Lightning-fast ASGI server
- **[Pydantic](https://pydantic-docs.helpmanual.io/)**: Data validation using Python type hints
- **[Joblib](https://joblib.readthedocs.io/)**: Efficient model serialization

## 📚 Learning Resources

This project demonstrates:
- Building ML-powered REST APIs with FastAPI
- Model training and persistence with scikit-learn
- API design best practices
- Interactive documentation with OpenAPI/Swagger
- CORS configuration for web applications

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Uday Kiran Hari**
- GitHub: [@UDAYKIRANHARI](https://github.com/UDAYKIRANHARI)

## 🌟 Acknowledgments

- Iris dataset from the UCI Machine Learning Repository
- FastAPI documentation and community
- Scikit-learn team for the excellent ML library

---

⭐ If you find this project helpful, please consider giving it a star!
