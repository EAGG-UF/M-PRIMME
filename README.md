# M-PRIMME
## Introduction
Predicting grain growth dynamics is essential for understanding and controlling microstructural evolution in materials. The novel Physics-Regularized Interpretable Machine Learning Microstructure Evolution (PRIMME) model has been shown to accurately predict 2D isotropic grain growth, but requires training on five consecutive grain structures, making it impractical training from experimental data.

To address this limitation, we introduce *multi-step PRIMME* (**M-PRIMME**) that requires only two non-adjacent grain structures, reducing the amount of required training data by two-thirds. 

**M-PRIMME** predicts grain growth by learning the probability of site-wise grain transitions rather than directly predicting future grain structures. It incorporates interfacial energy regularization to enforce physical consistency and employs a recurrent neural network to enable multi-step predictions from non-adjacent grain structure observations. 

Trained using Monte Carlo Potts simulations, **M-PRIMME** effectively captures 2D normal grain growth and outperforms PRIMME in several examples. Additionally, **M-PRIMME** allows for tunable grain growth rate predictions. 

The model’s effectiveness is demonstrated through geometric and topological analyses, including validation against the von Neumann-Mullins relationship, confirming its robustness for grain evolution prediction with significantly reduced data requirements.

## Results
<img src="./images/Grain%20Growth%20(grain(512_512_512))%20Methods%20(Step).png" alt="Evaluation of the evolution of polycrystalline grains within a 512 × 512 pixel domain">
Evaluation of the evolution of polycrystalline grains within a 512 × 512 pixel domain

