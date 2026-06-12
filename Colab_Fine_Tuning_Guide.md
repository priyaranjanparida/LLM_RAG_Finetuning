# Google Colab Fine-tuning Guide
## Run the Stage 3 AI PM Interview Expert Model

**Time:** ~1 hour total  
**Cost:** FREE (Google Colab GPU)  
**Outcome:** Fine-tuned Llama model ready for interview prep

---

## 📋 Pre-flight Checklist

Before you start, you need:

✅ **Google Account** (for Colab access)  
✅ **Training Data File** (`stage3_training_data.jsonl`)  
✅ **Hugging Face Token** (to access Llama model)  

---

## 🔑 Getting a Hugging Face Token (5 minutes)

The Llama model requires authentication. Here's how:

### Step 1: Create Hugging Face Account
1. Go to https://huggingface.co/join
2. Sign up with Google/GitHub (easiest)
3. Verify email

### Step 2: Get Your API Token
1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it: "Colab Fine-tuning"
4. Select: "Read" access
5. Create token
6. **Copy the token** (you'll need it in Colab)

### Step 3: Accept Llama License
1. Go to https://huggingface.co/meta-llama/Llama-2-7b-hf
2. Click "Accept" to agree to the license
3. Done!

---

## 🚀 Running the Colab Notebook

### Step 1: Open Colab
1. Go to https://colab.research.google.com
2. Click "File" → "Open notebook"
3. Click "Upload" tab
4. Upload `Stage3_FineTuning_Colab.ipynb`

### Step 2: Enable GPU
1. Click "Runtime" in the top menu
2. Click "Change runtime type"
3. Select: **GPU** (T4 or V100)
4. Click "Save"
5. You'll see "GPU" indicator at top right

### Step 3: Run Cells in Order

**Cell 1-2: Check GPU and Install Packages**
```python
# Check if GPU is available
# Should print: GPU Available: True
# GPU Name: Tesla T4 (or similar)

# Install packages (2 min)
```
⏱️ **Time: 2 minutes**

**Cell 3: Upload Training Data**

You have two options:

**Option A: Manual Upload (Recommended)**
1. In the left sidebar, click the **folder icon** 📁
2. Click the **upload icon** (↑ arrow)
3. Select `stage3_training_data.jsonl` from your computer
4. Wait for upload to complete
5. The file will be at `/content/stage3_training_data.jsonl`

**Option B: Use Sample Data**
- Run the cell to create sample data automatically
- Good for testing, but uses smaller dataset

⏱️ **Time: 1-2 minutes (or instant for sample)**

**Cell 4: Load and Verify Data**
```python
# Loads your training data
# Shows: "✅ Loaded 13 training examples"
```
⏱️ **Time: 30 seconds**

**Cell 5: Load Llama Model**
```python
# Downloads the Llama model (~5GB)
# First time: 2-3 minutes
# Shows: "✅ Model loaded successfully!"
# You may see: "Llama token required"
# If so: Press Ctrl+Enter to authenticate with HF token
```
⏱️ **Time: 2-3 minutes**

**Cell 6: Setup LoRA**
```python
# Configures efficient fine-tuning
# Shows: Trainable parameters (should be ~50-100M)
```
⏱️ **Time: 30 seconds**

**Cell 7: Prepare Dataset**
```python
# Formats data for training
# Shows training/validation split
```
⏱️ **Time: 30 seconds**

**Cell 8: Configure Training**
```python
# Sets hyperparameters
# Shows: Expected training time 20-30 minutes
```
⏱️ **Time: 10 seconds**

**Cell 9: TRAIN THE MODEL** 🚀
```python
# ⏱️ This is the big one: 20-30 MINUTES
# You'll see:
# - Loss decreasing (good!)
# - Progress bar moving
# - ETA countdown
# 
# DO NOT close this tab or interrupt!
# Let it run to completion
```
⏱️ **Time: 20-30 minutes**

When complete, you'll see:
```
✅ Training completed!
Checkpoint saved to ./results
```

**Cell 10: Save the Model**
```python
# Saves fine-tuned weights
# Shows: "✅ Model saved to /content/fine_tuned_model"
```
⏱️ **Time: 1 minute**

**Cell 11: Test the Model (Optional)**
```python
# Tests on a sample question
# Shows the fine-tuned model's response
# Good way to verify it worked!
```
⏱️ **Time: 1 minute**

---

## 📥 Download the Model

After training completes:

### Step 1: Open File Browser
- Click the **folder icon** 📁 on the left sidebar

### Step 2: Find and Download
1. Look for folder named `fine_tuned_model`
2. Right-click on it
3. Select **"Download"**
4. File will download as ZIP (~50-100MB)

### Step 3: Extract to Your Mac
1. Go to Downloads folder
2. Right-click the ZIP file
3. Select "Extract"
4. You'll have folder: `fine_tuned_model/`

---

## 🛠️ Troubleshooting

### Problem: "GPU not available"
**Solution:**
- Restart the runtime: Runtime → Restart session
- Make sure GPU is enabled: Runtime → Change runtime type → GPU
- Try again

### Problem: "Model download failed"
**Solution:**
- Check internet connection (stable required)
- Authenticating with Hugging Face token:
  - Install: `!huggingface-cli login`
  - Paste your HF token when prompted
  - Try loading model again

### Problem: "Out of memory"
**Solution:**
- Reduce batch size in Cell 8: `per_device_train_batch_size=2` (instead of 4)
- Reduce epochs: `num_train_epochs=2` (instead of 3)
- Restart runtime and try again

### Problem: "Training too slow"
**Solution:**
- Check that GPU is running (not CPU)
- You should see ~50 steps per minute
- If slower, restart runtime and check GPU is T4 or better

---

## ✅ What Success Looks Like

### After Training:
```
Loss: 0.45 → 0.12 (decreasing ✅)
Validation accuracy: 78% → 85% (increasing ✅)
Training completed in 28 minutes
✅ Model saved to /content/fine_tuned_model
```

### Model Ready:
- Folder `fine_tuned_model/` exists locally
- Contains: `adapter_config.json`, `adapter_model.bin`, `tokenizer.model`
- Ready to download to your Mac

---

## 🎯 Next Steps

After downloading the model:

1. **Move to project folder:**
   ```bash
   mv ~/Downloads/fine_tuned_model ~/Documents/RAG_Chatbot/
   ```

2. **Build Stage 3 app** in VS Code combining:
   - Fine-tuned model (expert knowledge)
   - RAG pipeline (fresh frameworks)
   - Streamlit UI (user interface)

3. **Test locally:**
   ```bash
   cd ~/Documents/RAG_Chatbot/
   streamlit run stage3_app.py
   ```

---

## 📊 Training Metrics to Watch

### Loss (should decrease):
- Start: ~2.5
- After 1 epoch: ~1.2
- After 3 epochs: ~0.3-0.5
- ✅ If decreasing, training is working

### Training time:
- Per epoch: ~8-10 minutes
- Total (3 epochs): ~24-30 minutes
- ✅ Normal for T4 GPU

### GPU utilization:
- Should be 90%+ for efficient training
- If lower, increase batch size

---

## 💡 Pro Tips

1. **Keep the Colab tab open** during training
   - Don't close, minimize, or navigate away
   - Closing tab might interrupt training

2. **Use "Save a copy in Drive"** 
   - Colab menu → "Save a copy in Drive"
   - Backs up your notebook
   - Can restart if connection drops

3. **Adjust learning rate if needed**
   - Too high (>5e-4): Loss jumps around
   - Too low (<1e-5): Training stalls
   - 2e-4 is good starting point

4. **Monitor from your phone**
   - Can open Colab in browser and check progress
   - But still don't close the original tab

---

## 🎓 What You've Learned

After this fine-tuning session, you understand:

✅ How GPU training works (T4 vs CPU)  
✅ LoRA efficiency (50MB model vs 13GB)  
✅ Hyperparameter tuning (learning rate, batch size, epochs)  
✅ Real-world ML workflow (data → train → download)  
✅ How to scale: from 10 examples to 1000s  

---

## 🚀 You're Ready!

You've now completed:
- ✅ Stage 1: Text Summarizer
- ✅ Stage 2: RAG System
- ✅ Stage 3: Fine-tuned Model

Next: **Build the Stage 3 app and ace those interviews!**

---

## Questions?

If something goes wrong:
1. Check GPU is enabled (Runtime → Change runtime type)
2. Restart session (Runtime → Restart all runtimes)
3. Copy error message
4. Try again from the failed cell

Good luck! 🚀
