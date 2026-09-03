# LearnPath AI

Your Personalized AI Learning Roadmap

## Streamlit Community Cloud deployment

Repository structure:

- `app.py`
- `requirements.txt`
- `.gitignore`
- `README.md`

### Streamlit Secrets

In Streamlit Community Cloud, open:

**App → Settings → Secrets**

Add:

```toml
GEMINI_API_KEY = "YOUR_ACTUAL_GEMINI_API_KEY"
```

The secret name is intentionally kept as `GEMINI_API_KEY`, matching the original Colab project.

### Important

- Do **not** add your Gemini API key to GitHub.
- Do **not** add an ngrok token to Streamlit Cloud.
- ngrok is only needed for the temporary Google Colab tunnel; Streamlit Community Cloud provides its own public URL.
- The model remains `gemini-3.6-flash`, matching the working project configuration.
- PDF generation/download has been removed. Markdown and TXT downloads remain.
