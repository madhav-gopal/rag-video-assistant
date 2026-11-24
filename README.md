# AI Video Assistant

A Streamlit app that turns a video or audio source into a transcript, summary, action items, decisions, and open questions. You can also ask questions about the processed transcript using retrieval-augmented generation (RAG).

## Features

- Process a YouTube URL or a local audio/video file.
- Transcribe speech locally with Whisper, with an optional Sarvam transcription path.
- Generate summaries, action items, decisions, and open questions with Mistral.
- Ask follow-up questions grounded in the transcript.
- Store transcript embeddings locally with ChromaDB.

## Requirements

- Python 3.10 or newer
- FFmpeg installed and available on `PATH`
- A Mistral API key for summaries, extraction, and RAG chat
- A Sarvam API key only if using the Sarvam transcription path

## Setup

```bash
git clone https://github.com/madhav-gopal/rag-video-assistant.git
cd rag-video-assistant
python -m venv .venv
```

Activate the virtual environment, then install dependencies:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install -r Requirements.txt
```

Copy `.env.example` to `.env` and add the API keys for the services you plan to use. Keep `.env` private; it is excluded from Git.

## Run the app

```bash
streamlit run app.py
```

The command line workflow is also available:

```bash
python main.py
```

## Generated files

Downloaded media, the local ChromaDB index, logs, Python caches, and environment files are stored locally and excluded from version control.
