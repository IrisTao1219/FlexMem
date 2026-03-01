# Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism

## 👣Introduction

This repository implements FlexMem, a novel and training-free visual memory mechanism for Multimodal Large language Models (MLLMs). FlexMem can help MLLMs continually watch video content and recall the most relevant memory fragments to answer the question.

![overview](images/overview.png)

### Key advantages:

- **Video Understanding of Infinite Lengths:** We study the long video understanding of MLLMs from the perspective of visual memory mechanism, and propose a novel approached termed FlexMem to scale up the input of video frames.
- **Outstanding Model Performance with Low Resource Requirement:** On a set of benchmarks, our FelxMem can greatly improve the capabilities of base MLLMs and outperform a set of SOTA methods using only one 3090 GPU.

## 🛠️ Usage


### Installation
```bash
conda create -n flexmem python=3.10 
pip install -r requirements.txt
conda activate flexmem
```
### Long Video Benchmark Evaluation
For **LongVideoBench** evaluation, you can use the following script to evaluate.

First, download the **LongVideoBench** dataset and **LLaVA-Video-7B-Qwen2** model weights to your local machine, assume their root are **data_root** and **model_root**, and replace **CKPT** and **DATA_ROOT** in FlexMem/scripts/video/lvbench/lvbench_eval_stream.sh with your local **model_root** and **data_root**.  

Then, you can use the following script:  
```bash
bash FlexMem/scripts/video/lvbench/lvbench_eval_stream.sh
```
For **FlexMem-Fast** evaluation on **MLVU**.

First download the **MLVU** dataset and the **LLaVA-Video-7B-Qwen2** model weights to your local machine, assume their roots are **data_root** and **model_root**, and replace **CKPT** and **DATA_ROOT** in FlexMem-fast/scripts/video/lvbench/lvbench_eval_stream.sh with your local **model_root** and **data_root**. 

Then, you can use the following script:  
```bash
bash FlexMem-fast/scripts/video/mlvu/mlvu_eval_stream.sh
```

## 🙏 Acknowledgements

- **LLaVA-NeXT**: the codebase we used for evaluation.
- **Video-XL**: the codebase we built upon. 



