# EPB-YOLO-PD
A detection algorithm for identifying exotic pet beetles

 The illegal smuggling of exotic pet beetles presents a growing threat to global ecosystems. Customs authorities serve as the first line of defense against biological invasions, yet current identification methods rely heavily on expert knowledge and time-consuming laboratory analysis, limiting timely responses at ports of entry. To address this, we propose EPB-YOLO-PD, a lightweight, mobile-deployable detection model designed for real-time recognition of exotic pet beetles.   

 EPB-YOLO-PD integrates a Feature Aggregation and Mixing Network (FAMNet), a Multi-Scale Efficient Lightweight Optimization Network (MELON), a Partial Multi-Head Self-Attention Residual Block (C4PMS), a Coordinate Attention Head (CAHead), and Slide Loss. Structural pruning and knowledge distillation are employed to reduce model size and speed up inference. Tested on a custom dataset comprising 13 intercepted species of exotic pet beetles, the model achieved detection accuracies of 93.3%-99.3%. Compared to the YOLOv11n baseline, EPB-YOLO-PD achieved a 2.0% increase in mAP0.5 (97.3%), a 74.04% reduction in model size (1.35MB), and a 65.08% decrease in computational complexity (2.2 GFLOPs). The PetBeetle Finder app, based on this model, runs at over 25 FPS on a Huawei Mate 40 smartphone.

 EPB-YOLO-PD provides a practical and efficient solution for real-time surveillance of exotic pet beetles at customs checkpoints. By enabling early detection and classification, it supports proactive pest management and offers a replicable approach for intercepting other invasive species. 

Experimental Setup:
The experiments were performed using PyTorch 2.2.2 and CUDA 12.1 on a Windows 10 64-bit system. The hardware configuration included an Intel Core i5-13600KF octa-core processor (3.50 GHz), 64 GB of RAM, and an NVIDIA RTX 3090 GPU. 
