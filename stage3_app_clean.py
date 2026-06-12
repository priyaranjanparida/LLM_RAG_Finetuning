import os
import streamlit as st
from datetime import datetime
import pdfplumber

# For fine-tuned model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# PAGE SETUP
# ============================================================================

st.set_page_config(
    page_title="Stage 3: Fine-Tuned AI PM Expert",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Stage 3: Fine-Tuned AI PM Expert")
st.markdown("**Interview prep with fine-tuned Llama 2**")

# ============================================================================
# SIDEBAR: SETTINGS
# ============================================================================

with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model path
    model_path = st.text_input(
        "Fine-Tuned Model Path",
        value="./fine_tuned_model",
        help="Path to folder with adapter_model.safetensors"
    )
    
    # Generation settings
    st.subheader("🎯 Generation Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens", 50, 300, 200)
    
    st.divider()
    
    # Model info
    st.subheader("ℹ️ Model Info")
    st.markdown("""
    **Fine-Tuned Model:** Llama 2 7B + LoRA
    
    **PDF Processing:** pdfplumber
    
    **Device:** CPU/GPU Auto
    
    **Status:** Ready ✅
    """)

# ============================================================================
# PDF UPLOAD & EXTRACTION
# ============================================================================

st.subheader("📄 Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDFs for reference",
    type="pdf",
    accept_multiple_files=True
)

pdf_content = ""

if uploaded_files:
    with st.spinner("Processing PDFs..."):
        try:
            for pdf_file in uploaded_files:
                # Extract text from PDF using pdfplumber
                with pdfplumber.open(pdf_file) as pdf:
                    file_text = ""
                    for page_num, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        if text:
                            file_text += f"\n--- Page {page_num + 1} ---\n{text}\n"
                    
                    pdf_content += f"\n=== {pdf_file.name} ===\n{file_text}\n"
            
            st.success(f"✅ Successfully processed {len(uploaded_files)} PDF(s)")
            
            # Show preview
            with st.expander("📋 Preview extracted text"):
                st.text(pdf_content[:1000] + "...")
        
        except Exception as e:
            st.error(f"❌ Error processing PDFs: {str(e)}")
            pdf_content = ""

# ============================================================================
# LOAD FINE-TUNED MODEL
# ============================================================================

@st.cache_resource
def load_fine_tuned_model(model_path):
    """Load fine-tuned Llama model with LoRA adapters"""
    try:
        if not os.path.exists(model_path):
            st.error(f"❌ Model not found at {model_path}")
            st.info(f"Expected at: {os.path.abspath(model_path)}")
            return None, None
        
        with st.spinner("📦 Loading fine-tuned model..."):
            st.info("⏳ This may take 1-2 minutes on first run...")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-2-7b-hf",
                trust_remote_code=True
            )
            tokenizer.pad_token = tokenizer.eos_token
            
            # Load base model - simple approach
            model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-2-7b-hf",
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            
            # Load LoRA adapters
            model = PeftModel.from_pretrained(model, model_path)
            model = model.merge_and_unload()
            
            st.success("✅ Fine-tuned model loaded!")
            return model, tokenizer
            
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

# ============================================================================
# CHAT INTERFACE
# ============================================================================

st.subheader("💬 Interview Q&A")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_question = st.chat_input("Ask an interview question...")

if user_question:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })
    
    with st.chat_message("user"):
        st.markdown(user_question)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Generating expert answer..."):
            try:
                # Load fine-tuned model
                fine_tuned_model, tokenizer = load_fine_tuned_model(model_path)
                
                if fine_tuned_model is None:
                    st.error("Could not load fine-tuned model")
                    st.stop()
                
                # Build context from uploaded PDFs
                context = ""
                if pdf_content:
                    context = f"\n\nReference Documents:\n{pdf_content[:2000]}"
                
                # Create prompt
                prompt = f"""You are an expert AI Product Manager with deep knowledge of AI/ML systems, product strategy, and technical implementation.

Your expertise includes:
- AI/ML product strategy and roadmaps
- Prompt engineering and LLM optimization
- Model selection and trade-offs (latency, quality, cost)
- User experience and product design
- Data strategy and metrics

{context}

User Question: {user_question}

Provide a comprehensive, expert-level answer that:
1. Shows deep understanding of AI/ML concepts
2. Includes real-world examples and trade-offs
3. Connects to business impact and ROI
4. Demonstrates senior PM thinking
5. Addresses hidden assumptions in the question
6. Provides actionable insights

Answer:"""
                
                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt")
                
                # Generate
                with torch.no_grad():
                    outputs = fine_tuned_model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=0.95,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Clean up the answer
                if "Answer:" in answer:
                    answer = answer.split("Answer:")[-1].strip()
                elif "User Question:" in answer:
                    answer = answer.split("User Question:")[-1].strip()
                    if "Answer:" in answer:
                        answer = answer.split("Answer:")[-1].strip()
                
                st.markdown(answer)
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
                
            except Exception as e:
                st.error(f"❌ Error generating answer: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
### 🎓 How to Use

1. **Upload PDFs** (optional): Add interview frameworks or study materials
2. **Ask Questions**: Type any AI PM interview question
3. **Get Answers**: Powered by fine-tuned Llama 2 model

### 🤖 Model Stack

- **Base Model**: Llama 2 7B (Meta)
- **Fine-Tuning**: LoRA adapters (trained on AI PM knowledge)
- **PDF Processing**: pdfplumber for text extraction
- **Device**: CPU (Mac-friendly)

### 💡 Example Questions

- "Explain the latency vs quality tradeoff in LLM selection"
- "How would you design an AI feature for a banking app?"
- "What metrics would you use to measure AI product success?"
- "Walk me through your approach to prompt optimization"

### ⚡ Performance

- **First question**: 2-5 minutes (model loads)
- **Other questions**: 30-60 seconds each
- **Note**: Running on CPU, so slower than GPU

### 🔐 Setup

Make sure you've run:
```bash
huggingface-cli login
```

This allows the app to download the Llama 2 model.

---

Made with ❤️ for AI PM Interview Prep | Stage 3: Fine-Tuned Model
""")
