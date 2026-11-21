#import "@preview/charged-ieee:0.1.4": ieee
#import "sections/introduction.typ": introduction
#import "sections/methodology.typ": methodology
#import "sections/related-work.typ": related-work
#import "sections/results-and-discussions.typ": results-and-discussions

#show bibliography: set std.bibliography(title: text(10pt)[References])

#show: ieee.with(
  title: [Yolo26 Underwater: YOLOv26-based Object Detection for Underwater Scenarios],
  abstract: [
    Abstrac.s
  ],
  authors: yaml("authors.yml"),
  bibliography: bibliography("refs.bib", full: true),
)

#introduction
#methodology
#related-work
#results-and-discussions

/*
References for when needed

@article{depthanything3,
  title = {Depth Anything 3: Recovering the Visual Space from Any Views},
  author = {Lin, Haotong and Chen, Sili and Liew, Jun Hao and Chen, Donny Y. and Li, Zhenyu and Shi, Guang and Feng, Jiashi and Kang, Bingyi},
  journal = {arXiv preprint arXiv:2511.10647},
  year = {2025},
  url = {https://arxiv.org/abs/2511.10647},
}

@article{lptmono,
  title = {LPTMono: Monocular Depth Estimation for Underwater Images Using Local Perception Transformer and Global-Local Context Fusion},
  author = {Liu, Dingshuo and Liu, Yiran and Li, Beibei and Duan, Qingling},
  journal = {The Journal of Supercomputing},
  volume = {81},
  pages = {1265},
  year = {2025},
  doi = {10.1007/s11227-025-07740-7},
}

@article{maudepth,
  title = {MAU-Depth: A Multi-Attention-Based Underwater Lightweight Self-Supervised Monocular Depth Estimation Method},
  author = {Yao, Peng and Wang, Yalu and Yang, Dongdong and Liu, Qiming and Yu, Jiatao},
  journal = {Intelligent Marine Technology and Systems},
  volume = {3},
  pages = {30},
  year = {2025},
  doi = {10.1007/s44295-025-00079-y},
}

@inproceedings{uwdepth,
  title = {Metrically Scaled Monocular Depth Estimation through Sparse Priors for Underwater Robots},
  author = {Ebner, Luca and Billings, Gideon and Williams, Stefan},
  booktitle = {2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages = {3751--3757},
  year = {2024},
  doi = {10.1109/ICRA57147.2024.10611007},
  eprint = {2310.16750},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2310.16750},
}




@inproceedings{seathru,
  title = {Sea-thru: A Method For Removing Water From Underwater Images},
  author = {Akkaynak, Derya and Treibitz, Tali},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {1682--1691},
  year = {2019},
  doi = {10.1109/CVPR.2019.00178},
}

@article{spade,
  title = {SPADE: Sparsity Adaptive Depth Estimator for Zero-Shot, Real-Time, Monocular Depth Estimation in Underwater Environments},
  author = {Zhang, Hongjie and Billings, Gideon and Williams, Stefan B.},
  journal = {arXiv preprint arXiv:2510.25463},
  year = {2025},
  url = {https://arxiv.org/abs/2510.25463},
}

@article{hazelines,
  title = {Underwater Single Image Color Restoration Using Haze-Lines and a New Quantitative Dataset},
  author = {Berman, Dana and Levy, Deborah and Avidan, Shai and Treibitz, Tali},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume = {43},
  number = {8},
  pages = {2822--2837},
  year = {2021},
  doi = {10.1109/TPAMI.2020.2977624},
  eprint = {1811.01343},
}


@article{uwadapter,
  title = {UW-Adapter: Adapting Monocular Depth Estimation Model in Underwater Scenes},
  author = {Ye, Xinchen and Chang, Yue and Xu, Rui and Li, Haojie},
  journal = {IEEE Transactions on Multimedia},
  volume = {27},
  year = {2025},
  doi = {10.1109/TMM.2025.3543089},
}

@article{watermono,
  title = {WaterMono: Teacher-Guided Anomaly Masking and Enhancement Boosting for Robust Underwater Self-Supervised Monocular Depth Estimation},
  author = {Ding, Yilin and Li, Kunqian and Mei, Han and Liu, Shuaixin and Hou, Guojia},
  journal = {arXiv preprint arXiv:2406.13344},
  year = {2024},
  url = {https://arxiv.org/abs/2406.13344},
}


*/
