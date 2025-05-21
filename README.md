# Defend against Label Inference Attacks in Vertical Federated Learning via Label Compression
Vertical Federated Learning (VFL) has gained popularity for collaborative decision-making but suffers from serious privacy risks, especially label inference attacks. Existing defenses struggle against recent model completion-based attacks without sacrificing significant performance.

To address this, we propose **LCD**, a novel defense method based on two key ideas: **label compression** and **embedding compaction**. 
- **Label compression** generates fake labels from deep features to decouple true labels from the bottom model.  
- **Embedding compaction**, based on center loss, increases the difficulty of inferring labels by tightening intra-class feature distributions.  

## 🚀 How to run
### 1. Prepare your dataset
```bash
Place your dataset files into the `Datasets/` folder. For example: `Datasets/mnist`.
```
### 2. Train the LCD-protected VFL model
```bash
#active attack
python lc.py --use-mal-optim True -d mnist --path-dataset ./Datasets/mnist --epochs 100 --lr 5e-2 --half 14 --batch-size 128 --gpu_id 1 --weight_cent 10
#passive attack
python lc.py --use-mal-optim False -d mnist --path-dataset ./Datasets/mnist --epochs 100 --lr 5e-2 --half 14 --batch-size 128 --gpu_id 1 --weight_cent 100
```
### 3. Launch the label inference attack (model completion)
```bash
python model_completion.py --dataset-name mnist --dataset-path ./Datasets/mnist --n-labeled 400 --party-num 2 --half 14 --k 5 --resume-dir ./baselines/saved_models/ --resume-name model.pth --print-to-txt 1 --epochs 50 --gpu_id 1
```


## 📃 Acknowledgments

We would like to thank the authors of [https://github.com/FuChong-cyber/label-inference-attacks](https://github.com/FuChong-cyber/label-inference-attacks) for their valuable work on label inference attacks in VFL. Their research provided critical insights and experimental foundations that inspired and guided our defense design in this paper.
