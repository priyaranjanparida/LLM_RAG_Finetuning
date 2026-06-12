# Stage 3 App: Installation and Usage Guide

## 🎯 What This App Does

**Combines:**
- ✅ Fine-tuned Llama model (learns AI PM patterns)
- ✅ RAG pipeline (retrieves interview frameworks)
- ✅ Streamlit UI (beautiful, interactive interface)

**Result:** Expert-level AI PM interview answers

---

## 📋 Prerequisites

Before running, you need:

1. **Fine-tuned model** (downloaded from Colab)
   - Location: `~/Documents/RAG_FineTuning/fine_tuned_model/`
   - Contains: `adapter_model.bin`, `tokenizer.model`, etc.

2. **Google API Key**
   - Get from: https://makersuite.google.com/app/apikey
   - Keep it safe (treat like password)

3. **Interview framework PDFs** (optional but recommended)
   - Any PDF with interview concepts
   - Example: Your Stage 1 interview guide PDF
   - Will be used for RAG retrieval

4. **Python installed** (via Anaconda on Mac)
   - Check: `python --version` should show 3.8+

---

## 🚀 Installation (First Time Only)

### Step 1: Create Project Folder

```bash
mkdir -p ~/Documents/Stage3_App
cd ~/Documents/Stage3_App
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using Anaconda (recommended)
conda create -n stage3 python=3.11
conda activate stage3

# Or using venv
python -m venv venv
source venv/bin/activate  # Mac/Linux
```

### Step 3: Copy the App File

```bash
# Copy stage3_app.py to your project folder
cp ~/Downloads/stage3_app.py ~/Documents/Stage3_App/
```

### Step 4: Install Dependencies

```bash
pip install streamlit
pip install transformers torch
pip install langchain langchain-community langchain-google-genai
pip install chromadb
pip install pdf2image pdfplumber pypdf
```

**Alternative: Install from requirements.txt**

Create `requirements.txt`:
```
streamlit==1.56.0
transformers==4.36.0
torch==2.1.0
langchain==0.1.0
langchain-community==0.0.30
langchain-google-genai==1.0.0
chromadb==0.4.0
pdfplumber==0.10.0
google-generativeai==0.3.0
```

Then install:
```bash
pip install -r requirements.txt
```

### Step 5: Copy Fine-tuned Model

```bash
# Copy the fine-tuned model to your app folder
cp -r ~/Documents/RAG_FineTuning/fine_tuned_model \
      ~/Documents/Stage3_App/

# Verify it exists
ls -la ~/Documents/Stage3_App/fine_tuned_model/
```

Should show:
```
adapter_config.json
adapter_model.bin
tokenizer.model
special_tokens_map.json
training_args.bin
```

---

## ▶️ Running the App

### Method 1: Simple (Recommended)

```bash
# Navigate to app folder
cd ~/Documents/Stage3_App

# Activate environment
conda activate stage3

# Run the app
streamlit run stage3_app.py
```

**Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501

  Stopping...
```

Your app automatically opens in browser! 🎉

### Method 2: Command Line (If browser doesn't open)

```bash
# Run app
streamlit run stage3_app.py

# Then manually open:
# http://localhost:8501
```

### Method 3: Full Path

```bash
streamlit run ~/Documents/Stage3_App/stage3_app.py
```

---

## 🔑 First Time Setup (In the App)

### Step 1: Enter Google API Key

1. Go to: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key (long string starting with `AIza...`)
4. In app sidebar: Paste into "Google API Key" field
5. ✅ Should be ready!

### Step 2: Verify Model Path

**Default:** `./fine_tuned_model`

**If in different location:**
- Update path in sidebar: "Path to fine-tuned model"
- Example: `/Users/prpindia/Documents/Stage3_App/fine_tuned_model`

### Step 3: Upload Interview Frameworks (Optional)

1. In sidebar: "Upload Interview Framework Documents"
2. Select PDF files with interview concepts
3. Suggested PDFs:
   - Your Stage 1 interview guide
   - AI/ML concepts documents
   - Case study PDFs

4. Click "📥 Load Documents for RAG"
5. Wait for: "✅ RAG Ready!"

---

## 💬 Using the App

### Basic Flow

1. **Setup** (first time):
   - ✅ Enter API key
   - ✅ Upload PDFs
   - ✅ Click "Load Documents"

2. **Ask Questions**:
   - Type interview question in chat
   - Choose: Use RAG? Use Fine-tuned?
   - Press Enter

3. **Get Answer**:
   - See expert response
   - Click "📚 Sources" to see which docs were used
   - See model info at bottom

### Example Questions to Ask

```
"Explain the latency-quality-cost tradeoff"
"How would you design an AI feature for a bank?"
"What's the difference between RAG and fine-tuning?"
"Tell me about tokens and why they cost money"
"Design a system for 1 million AI model fine-tunings"
"How do you handle API failures in production?"
"What metrics would you use to measure AI feature success?"
```

### Understanding the Response

**Response includes:**
1. Expert answer (from fine-tuned model + RAG)
2. "Sources Used" section (shows retrieved documents)
3. Model info (which approaches were used)

**Sources:**
- If RAG enabled: Shows relevant document chunks
- Helps you understand what frameworks were applied
- Click to see full context

---

## 🔧 Configuration Options

### In Sidebar

**Fine-tuned Model:**
- Path: Where your model is stored
- Leave default if in same folder

**RAG Configuration:**
- **Chunk size:** How many tokens per document piece
  - Lower (200): More pieces, more search options
  - Higher (1000): Fewer pieces, better context
  - **Default: 500** (good balance)

- **Chunk overlap:** Tokens repeated between chunks
  - Prevents losing info at boundaries
  - **Default: 100** (usually good)

- **Top K documents:** How many documents to retrieve
  - More (5-10): More context, slower
  - Fewer (1-3): Faster, less context
  - **Default: 3** (balanced)

**Generation Parameters:**
- **Temperature:** Creativity level
  - 0.0: Always same answer (deterministic)
  - 0.5: Balanced
  - 1.0: Very creative
  - **Default: 0.7** (good for interviews)

- **Max tokens:** Response length
  - 100-500: Short answers
  - 500-1000: Medium answers
  - 1000+: Comprehensive answers
  - **Default: 500** (good length)

---

## 📚 Tips and Tricks

### For Best Results

1. **Ask specific questions:** Vague questions = vague answers
   - ❌ "Tell me about AI"
   - ✅ "How would you reduce AI model costs by 50%?"

2. **Include context:** More info = better answers
   - ❌ "How to improve RAG?"
   - ✅ "We have 1M documents in RAG. Search is slow. How to improve?"

3. **Enable both:** RAG + Fine-tuning together works best
   - Fine-tuned: Knows PM thinking
   - RAG: Has fresh context
   - Together: Expert-level answers

4. **Upload relevant PDFs:** Better documents = better RAG
   - Upload your interview prep materials
   - Add case studies
   - Include frameworks

### Troubleshooting

**Problem: "Model not found"**
```
Fix: Check fine_tuned_model folder exists in app directory
ls ~/Documents/Stage3_App/fine_tuned_model/
If not there:
cp -r ~/Downloads/fine_tuned_model ~/Documents/Stage3_App/
```

**Problem: "API key error"**
```
Fix: 
1. Go to https://makersuite.google.com/app/apikey
2. Create new key
3. Copy entire key
4. Paste in sidebar
5. Make sure no spaces before/after
```

**Problem: "RAG not working"**
```
Fix:
1. Click "Reset RAG" in sidebar
2. Upload PDFs again
3. Click "Load Documents"
4. Wait for "✅ RAG Ready!"
```

**Problem: "App runs slow"**
```
Fix:
1. Reduce chunk_size (fewer pieces to search)
2. Reduce top_k (retrieve fewer documents)
3. Reduce max_tokens (shorter responses)
4. Disable RAG if not needed
```

**Problem: "Out of memory"**
```
Fix:
1. Close other apps on Mac
2. Reduce batch size
3. Don't upload huge PDFs
4. Use fewer documents
```

---

## 🎓 What You Can Do With This App

### Interview Preparation
```
Ask: "Typical AI PM interview question"
Get: Expert-level answer
Study: Frameworks and approaches
```

### Learning
```
Upload: AI/ML concept PDFs
Ask: Deep questions
Learn: How concepts apply to real systems
```

### Prototyping
```
Ask: System design questions
Get: Architecture recommendations
Iterate: Refine with follow-up questions
```

### Teaching
```
Upload: Your lecture notes
Ask: Explain like I'm a PM
Share: App with colleagues
```

---

## 📊 Understanding the Pipeline

**Data Flow:**

```
User Question
    ↓
    ├─→ [RAG Retrieval] → Retrieve relevant docs
    │                    ↓
    │                   Docs + Question
    │
    ├─→ [Fine-tuned Model] → Domain understanding
    │                         ↓
    │                        PM Patterns
    │
    └─→ [LLM Generator] → Combine RAG + Fine-tune
                           ↓
                        Expert Answer
```

**What each part does:**

1. **RAG Retrieval:**
   - Searches your PDFs
   - Finds relevant sections
   - Provides context

2. **Fine-tuned Model:**
   - Knows AI PM interview patterns
   - Understands trade-offs
   - Uses business thinking

3. **LLM Generator:**
   - Combines context + expertise
   - Generates coherent response
   - Explains reasoning

---

## 🚀 Advanced Usage

### Custom Fine-tuned Model

Instead of using our Llama model:

```python
# In stage3_app.py, modify load_fine_tuned_model():
model_path = "/path/to/your/model"
```

### Different LLM for Generation

Instead of Gemini:

```python
# Use Claude:
from langchain.chat_models import ChatAnthropic
llm = ChatAnthropic(model="claude-2")

# Or OpenAI:
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")
```

### Different Vector Database

Instead of Chroma:

```python
# Use FAISS:
from langchain.vectorstores import FAISS
vector_store = FAISS.from_documents(docs, embeddings)

# Or Pinecone:
from langchain.vectorstores import Pinecone
vector_store = Pinecone.from_documents(docs, embeddings, index_name="interview-expert")
```

---

## 📈 Performance Tips

**For faster responses:**
- Reduce chunk_size (faster search)
- Reduce top_k (fewer docs to search)
- Use Gemini Flash (default) not Pro
- Disable RAG if not needed

**For better answers:**
- Increase chunk_size (more context)
- Increase top_k (more documents)
- Use Gemini Pro (better quality)
- Enable both RAG and Fine-tune
- Upload more relevant PDFs

---

## 🎉 You've Built Stage 3!

Congratulations! You've created:

✅ **Stage 1:** Text Summarizer
✅ **Stage 2:** RAG Chatbot
✅ **Stage 3:** Fine-tuned RAG Expert

**What you learned:**
- Fine-tuning LLMs
- RAG pipelines
- Building with Streamlit
- Production thinking

**Next steps:**
1. Use for interview prep
2. Deploy to production
3. Fine-tune on your own data
4. Build custom domain experts

---

## 🆘 Need Help?

If something doesn't work:

1. **Check Python version:** `python --version` (should be 3.8+)
2. **Check packages:** `pip list | grep streamlit`
3. **Check folder:** `ls ~/Documents/Stage3_App/`
4. **Check API key:** Make sure it's valid
5. **Read errors:** Often tells exactly what's wrong

**Common errors:**
- `ModuleNotFoundError`: Missing package → `pip install package_name`
- `FileNotFoundError`: Wrong path → Check folder location
- `API Error`: Invalid key → Get new one from makersuite.google.com
- `OutOfMemory`: Too many docs → Upload fewer PDFs

---

**You're all set!** 🚀

Run: `streamlit run stage3_app.py` and start interviewing!
