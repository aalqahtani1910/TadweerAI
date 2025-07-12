# TadweerAI – ذكاء التدوير المجتمعي

## Setup

1. **Clone or download this repo:**
   ```bash
   git clone https://github.com/yourusername/tadweerai.git
   cd tadweerai
   ```

2. **Create a virtual environment (recommended) and install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure your environment variables:**
   - Copy the example file:
     ```bash
     cp .env.example .env
     ```
   - Open `.env` and set your OpenAI API key:
     ```env
     OPENAI_API_KEY=sk-...
     ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser and navigate to:**
   ```
   http://localhost:8501
   ```

Enjoy building and upcycling with AI! 🎉
