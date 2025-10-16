# QuantumPredict: A Hybrid AI & Quantum Financial Analysis Dashboard 🚀

This project is a full-stack application that leverages classical machine learning, real-time sentiment analysis, and experimental quantum machine learning to provide comprehensive financial insights.

![Project Demo GIF](link-to-your-demo-gif-or-screenshot)
*(You can convert your video to a GIF and upload it here for a great visual!)*

## ✨ Key Features

- **Hybrid Modeling:** Fuses a classical **LSTM** model with two distinct **Quantum** models (Classifier & Regressor).
- **Real-Time Sentiment Analysis:** Integrates a live **NewsAPI** feed with a **Hugging Face Transformers model** to analyze market sentiment from news headlines.
- **Quantum Experimentation:** Compares a simple quantum circuit with an advanced, entangled `TwoLocal` circuit to test performance differences.
- **Interactive Dashboard:** A user-friendly web interface built with **FastAPI** and modern HTML/JS to visualize all data streams.
- **Persistent Logging:** All analysis results are saved to a **SQLite** database for reproducibility.

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **Classical ML:** TensorFlow, Keras, Scikit-learn
- **Quantum ML:** Qiskit, Qiskit Machine Learning
- **NLP:** Hugging Face Transformers
- **Data:** yfinance, NewsAPI, Pandas
- **Frontend:** HTML, Tailwind CSS, Chart.js
- **Database:** SQLite

## ⚙️ How to Run

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
4.  Create a `.env` file in the root directory and add your NewsAPI key:
    ```
    NEWS_API_KEY="your_actual_api_key_here"
    ```
5.  Run the backend server:
    ```bash
    uvicorn backend.api:app --reload
    ```
6.  Open your browser and navigate to `http://127.0.0.1:8000`.
