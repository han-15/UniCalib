# UniCalib: Targetless LiDAR–Camera Calibration via Probabilistic Flow on Unified Depth Representations

Official repository for the paper:

> **UniCalib: Targetless LiDAR–Camera Calibration via Probabilistic Flow on Unified Depth Representations**  
> Shu Han, Xubo Zhu, Ji Wu, Ximeng Cai, Wen Yang, Huai Yu, and Gui-Song Xia  
> **WACV 2026**

[[Paper (arXiv)]](https://arxiv.org/abs/2504.01416) 

---

### 🧠 Overview
**UniCalib** introduces a probabilistic framework for *targetless LiDAR–camera calibration* by unifying both modalities into dense depth representations.  
We model the calibration problem as **probabilistic flow estimation** in the unified depth space, incorporating flow uncertainty and a perceptually weighted sparse flow loss to achieve robust cross-sensor alignment.

---

### 🌟 Key Highlights
- **Probabilistic depth flow** reframing 2D–3D correspondence for targetless calibration.  
- **Unified depth representation** bridging LiDAR and camera via a shared encoder.  
- **Reliability-aware modeling** with the novel **PWSF loss** for robust optimization.  
- **Strong generalization** with accurate results across diverse datasets.  
---

### 🚧 Code Availability
The source code will be released soon after the official publication of the paper.  
Please stay tuned and ⭐ star this repository for updates!

---

### 📄 Citation
If you find this work useful, please consider citing:

```bibtex
@article{han2025unicalib,
  title={UniCalib: Targetless LiDAR-Camera Calibration via Depth Flow},
  author={Han, Shu and Zhu, Xubo and Wu, Ji and Cai, Ximeng and Yang, Wen and Yu, Huai and Xia, Gui-Song},
  journal={arXiv preprint arXiv:2504.01416},
  year={2025}
}
