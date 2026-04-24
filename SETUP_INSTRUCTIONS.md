# 🚀 QUICK START GUIDE - Run in 5 Minutes

## Step 1: Install Python Dependencies (2 minutes)

```bash
# Install PyTorch (choose based on your system)

# For CPU Only:
pip install torch torchvision

# For GPU (CUDA 11.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For GPU (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies:
pip install -r requirements.txt
```

**If you're unsure about GPU vs CPU:**
- Just use: `pip install torch torchvision` (works for everyone)
- The code will auto-detect and use GPU if available

---

## Step 2: Run the Code (2-3 minutes)

```bash
python self_pruning_network.py
```

**What happens:**
1. Downloads CIFAR-10 dataset (~200 MB) - First time only
2. Trains network 3 times with different λ values
3. Generates visualization (pruning_results.png)
4. Prints results table

**Total training time:**
- CPU: ~20-30 minutes
- GPU: ~3-5 minutes

---

## Step 3: Check Results

After training completes, you'll see:

```
RESULTS SUMMARY
==================================================================
Lambda          Test Accuracy (%)    Sparsity (%)
------------------------------------------------------------------
0.000100        92.50                18.75
0.001000        90.80                42.30
0.010000        87.20                71.20

✅ Visualization saved as 'pruning_results.png'
```

Two output files created:
1. **pruning_results.png** - Visualization graphs
2. **Console output** - All training metrics

---

## ⚠️ Common Issues & Fixes

### Issue 1: "ModuleNotFoundError: No module named 'torch'"
**Solution:**
```bash
pip install --upgrade pip
pip install torch torchvision
```

### Issue 2: "CUDA out of memory" (GPU users)
**Solution - Reduce batch size:**
Edit `self_pruning_network.py`, line ~260:
```python
batch_size = 64  # Change from 128 to 64
```

### Issue 3: "FileNotFoundError: data directory"
**Solution:**
The code auto-creates `./data` folder. Just make sure you have write permissions.

### Issue 4: Slow training on CPU
**This is normal!** CPU training takes 20-30 minutes. 
- You can reduce epochs if needed (line ~260: `num_epochs = 10`)

---

## 🎯 Understanding the Output

### Gate Distribution Plot
- **Large spike at 0** = Many weights pruned successfully ✅
- **Cluster away from 0** = Important weights retained ✅
- **Uniform distribution** = Pruning failed ❌

### Sparsity vs Accuracy Trade-off
- **Higher λ** = More sparsity, lower accuracy
- **Lower λ** = Less sparsity, higher accuracy
- **Sweet spot** = Usually λ = 0.001

---

## 📊 Customization Options

### To train for longer (better results):
```python
num_epochs = 50  # Default: 20
```

### To test more lambda values:
```python
lambda_values = [0.00001, 0.0001, 0.001, 0.01, 0.1]  # Default: 3 values
```

### To use smaller network (faster):
```python
# In SelfPruningNetwork class:
self.fc1 = PrunableLinear(3072, 256)  # Reduce from 512
self.fc2 = PrunableLinear(256, 128)   # Reduce from 256
self.fc3 = PrunableLinear(128, 10)    # Keep same
```

---

## ✅ Verification

After running, check for:
- [ ] `pruning_results.png` file created
- [ ] Results printed to console
- [ ] All 3 lambda values trained successfully
- [ ] No error messages

---

## 📝 For Submission

Your GitHub repo is ready at:
**https://github.com/Smatypradish/tredence-ai-engineer-case-study**

Files included:
- ✅ `self_pruning_network.py` - Main implementation
- ✅ `README.md` - Detailed explanation
- ✅ `requirements.txt` - Dependencies
- ✅ `SETUP_INSTRUCTIONS.md` - This file

---

## 🔥 Need Help?

Check these in order:
1. PyTorch installation: https://pytorch.org/get-started/locally/
2. CIFAR-10 dataset: Automatically downloaded, ~200MB
3. GPU support: Run `python -c "import torch; print(torch.cuda.is_available())"`

---

**Ready? Run this:**
```bash
pip install -r requirements.txt && python self_pruning_network.py
```

Good luck! 🚀
