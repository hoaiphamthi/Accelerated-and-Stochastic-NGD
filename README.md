This code is used for experiments in paper Nguyen Phung Hai Chung, Nguyen Hoang Chung, and Pham Thi Hoai "Accelerated and stochastic versions of gradient descent method with NGD stepsize and applications". This implementation is based on the experiment code of the papaer Y. Malitsky and K. Mishchenko "Adaptive Gradient Descent without Descent" (two-column [[https://proceedings.icml.cc/static/paper_files/icml/2020/2854-Paper.pdf][ICML]] or one-column [[https://arxiv.org/pdf/1910.09529.pdf][arxiv]])

--------
# Usage
There are 5 experiments which compare accelerated and stochastic NGD methods with other related methods. Particularly,

- [Logistic regression](Accelerated_NGD/logistic_regression.ipynb)
- [Matrix factorization](Accelerated_NGD/matrix_factorization.ipynb)
- [Cubic regularization](Accelerated_NGD/cubic_regularization.ipynb)
- [Linesearch for logisitic regresion](Accelerated_NGD/linesearch_logistic_regression_w8a.ipynb)
- [Neural networks](SNGD/optimizer.py)
--------
# Reference
This paper is under review.

python run.py --dataset mnist --optimizer SNGDh --lr 1e-5 --eta0 0.2 --eta1 0.15 --beta 0.0 --alpha 0.9
