# Stage 3 App: Quick Start (5 Minutes)

## 🚀 Fastest Way to Get Running

### Prerequisites (You should have these)

✅ Fine-tuned model folder: `~/Downloads/fine_tuned_model/`  
✅ Google API Key (from https://makersuite.google.com/app/apikey)  
✅ Python 3.8+ (via Anaconda)  

---

## ⚡ 5-Minute Setup

### Step 1: Create Folder (30 seconds)

```bash
mkdir -p ~/Documents/Stage3_App
cd ~/Documents/Stage3_App
```

### Step 2: Copy Files (1 minute)

```bash
# Copy the app file
cp ~/Downloads/stage3_app.py ~/Documents/Stage3_App/

# Copy fine-tuned model
cp -r ~/Downloads/fine_tuned_model ~/Documents/Stage3_App/

# Verify both exist
ls -la ~/Documents/Stage3_App/
```

Should show:
```
stage3_app.py
fine_tuned_model/ (with adapter_model.bin inside)
```

### Step 3: Install Packages (2 minutes)

```bash
# One-line install
pip install streamlit transformers torch langchain langchain-community langchain-google-genai chromadb pdfplumber
```

### Step 4: Run the App (1.5 minutes)

```bash
cd ~/Documents/Stage3_App
streamlit run stage3_app.py
```

Browser opens automatically! 🎉

---

## 🔑 First Time Setup (In the App)

### 1. Enter API Key
- Sidebar: Paste Google API Key
- Get from: https://makersuite.google.com/app/apikey

### 2. Verify Model Path
- Default: `./fine_tuned_model`
- Change if needed in sidebar

### 3. Upload PDFs (Optional)
- Add interview framework documents
- Click "📥 Load Documents"
- Wait for: "✅ RAG Ready!"

---

## 💬 Test It!

Ask questions like:
- "What are tokens and why do they cost?"
- "Explain latency-quality-cost tradeoff"
- "How would you design an AI feature for a bank?"
- "Should we use RAG or fine-tuning?"

---

## ✅ You're Done!

**You've built Stage 3:** Fine-tuned RAG Expert System!

### What You Have:
- ✅ Stage 1: Text Summarizer
- ✅ Stage 2: RAG Chatbot  
- ✅ Stage 3: Fine-tuned RAG Expert

### What's Next:
1. Use for interview prep
2. Deploy to production
3. Fine-tune on your own data
4. Build for your domain

---

## 🆘 Troubleshooting

**"Module not found error"**
```bash
pip install missing_module_name
```

**"Model not found"**
```bash
# Check model exists
ls ~/Documents/Stage3_App/fine_tuned_model/

# If not there, copy again
cp -r ~/Downloads/fine_tuned_model ~/Documents/Stage3_App/
```

**"API Key error"**
- Get new key from makersuite.google.com
- Paste entire key (no spaces)
- Wait 5 seconds

**"App runs slow"**
- Reduce PDFs (load fewer documents)
- Reduce chunk_size in sidebar
- Reduce max_tokens

---

## 🎓 You Did It! 🎉

Go from **zero** to **AI PM expert system** in 30 minutes!

Interview prep just got supercharged! 🚀
