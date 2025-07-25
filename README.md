# Sentiment Analysis Web & Terminal App

This project provides two simple interfaces to perform sentiment analysis on user-provided text:  
- A **Web-based interface** with a user-friendly text box  
- A **Terminal-based interface** for quick command-line usage

---

## 🖥️ Web Interface

Launch an interactive webpage where you can paste your text and get the sentiment result.

### 🔄 How to Run

1. Ensure your Python environment is activated and dependencies are installed.
2. Run the following command in the terminal:

   ```bash
   python app.py
   ```

3. After running, a URL (e.g., `http://127.0.0.1:5000`) will be displayed in the terminal.  
4. Open the link in your web browser.  
5. Enter your text in the box and click **Analyze** to view the sentiment (e.g., Positive, Negative, Neutral).

---

## 💻 Terminal Interface

Use this interface to run sentiment analysis directly in the terminal.

### 🔄 How to Run

1. Make sure you're in a valid Python environment.
2. Run the following command:

   ```bash
   python sentiment_analysis.py
   ```

3. You'll be prompted to enter a line of text.  
4. The sentiment result will be printed directly in the terminal.

---

## 🔧 Requirements

Make sure you have **Python 3.7 or higher** installed.

Install all required packages with:

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
.
├── app.py                  # Web interface using Flask
├── sentiment_analysis.py   # CLI-based sentiment tool
├── requirements.txt        # List of required Python packages
└── README.md               # Project documentation
```

---

## 📬 Feedback

Have suggestions or issues?  
Feel free to [open an issue](https://github.com/your-username/your-repo-name/issues) or submit a pull request!
