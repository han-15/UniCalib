# UniCalib: Targetless LiDAR–Camera Calibration via Probabilistic Flow on Unified Depth Representations

Official repository for the paper:

> **UniCalib: Targetless LiDAR–Camera Calibration via Probabilistic Flow on Unified Depth Representations**  
> Shu Han, Xubo Zhu, Ji Wu, Ximeng Cai, Wen Yang, Huai Yu, and Gui-Song Xia  
> Accepted to **WACV 2026**

[[Paper (arXiv)]](https://arxiv.org/abs/2504.01416) 

---

### 🧠 Overview
**UniCalib** introduces a probabilistic framework for *targetless LiDAR–camera calibration* by unifying both modalities into dense depth representations.  
We model the calibration problem as **probabilistic flow estimation** in the unified depth space, incorporating flow uncertainty and a perceptually weighted sparse flow loss to achieve robust cross-sensor alignment.

---

### 🌟 Key Highlights
- **Targetless and data-driven** calibration without explicit calibration patterns.  
- **Unified representation** bridging LiDAR and image domains through dense depth maps.  
- **Probabilistic modeling** using Laplace distributions to capture flow uncertainty.  
- **Perceptually weighted sparse flow (PWSF) loss** for stable and accurate optimization.

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
