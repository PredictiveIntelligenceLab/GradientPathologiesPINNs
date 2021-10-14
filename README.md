# Understanding and mitigating gradient pathologies in physics-informed neural networks

The widespread use of neural networks across different scientific domains often involves constraining them to satisfy certain symmetries, conservation laws, or other domain knowledge. Such constraints are often imposed as soft penalties during model training and effectively act as domain-specific regularizers of the empirical risk loss. Physics-informed neural networks is an example of this philosophy in which the outputs of deep neural networks are constrained to approximately satisfy a given set of partial differential equations. In this work we review recent advances in scientific machine learning with a specific focus on the effectiveness of physics-informed neural networks in predicting outcomes of physical systems and discovering hidden physics from noisy data. We will also identify and analyze a fundamental mode of failure of such approaches that is related to numerical stiffness leading to unbalanced back-propagated gradients during model training. To address this limitation we present a learning rate annealing algorithm that utilizes gradient statistics during model training to balance the interplay between different terms in composite loss functions. We also propose a novel neural network architecture that is more resilient to such gradient pathologies. Taken together, our developments provide new insights into the training of constrained neural networks and consistently improve the predictive accuracy of physics-informed neural networks by a factor of 50-100x across a range of problems in computational physics.

- Sifan Wang, Yujun Teng, Paris Perdikaris.


## Citation

    @article{wang2021understanding,
      title={Understanding and mitigating gradient flow pathologies in physics-informed neural networks},
      author={Wang, Sifan and Teng, Yujun and Perdikaris, Paris},
      journal={SIAM Journal on Scientific Computing},
      volume={43},
      number={5},
      pages={A3055--A3081},
      year={2021},
      publisher={SIAM}
      }
