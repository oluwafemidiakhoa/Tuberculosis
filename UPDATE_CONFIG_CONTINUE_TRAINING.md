# Continue TB Training with More Patience

## Update Config in Colab

Run this cell in your Colab notebook to update the config:

```python
import yaml

# Load current config
with open('configs/config_tb_ast.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Update settings for longer training
config['patience'] = 15  # Increase from 7 to 15
config['epochs'] = 50    # Increase from 30 to 50 (in case it needs more)

# Save updated config
with open('configs/config_tb_ast.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("✅ Config updated!")
print(f"   Patience: {config['patience']} epochs")
print(f"   Max epochs: {config['epochs']}")
```

## Resume Training

Then run this to continue from your best checkpoint:

```python
# Resume training from where it stopped
!python train_ast.py --config configs/config_tb_ast.yaml
```

The training will:
- ✅ Load your best checkpoint (93.97% accuracy)
- ✅ Continue from epoch 12
- ✅ Wait 15 epochs for improvement before stopping
- ✅ Keep tracking energy savings

Expected:
- Training will continue until epoch ~26-30
- Might reach 94-95% accuracy
- Energy savings should stay around 80-82%

---

**Just copy these code blocks into new Colab cells and run them!**
