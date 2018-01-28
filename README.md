# keras-mnist-analysis

MNIST data analysis using TensorFlow and Keras. Uses a Convotional Neural
Network with a relatively simple architecture to recognize hand written
digits from the [MNIST][mnist] dataset. Also features some dropout layers to
avoid overfitting and improve accuracy.

Analysis steps:
- Loading MNIST dataset
- Data type conversions
- Scaling
- One-hot encoding of the result classes
- Defining Keras sequential model and compiling it
- Model fitting
- Metrics and results

## Dependencies

You need to have:
- [TensorFlow][tf]. I recommend using GPU if possible for better performance.
- [Keras][keras] using TensorFlow backend. Offersa high-level API and a more
  *pythonic* way to develop deep learning network architectures. Cuts down the
  development time unless you need a lot of custom estimators.
- [Numpy][numpy]
- [scikit-learn][scikit]. Used for easier generation of the confusion matrix.

It's always recommended to use [Virtualenv][ve] to create an isolated python
environment for easier handling of dependencies and potential problems.

## Running

After all the dependencies are met, you can run the analysis simply with:

```python
python analysis.py
```

The analysis should complete in a few minutes depending on your hardware
configuration. Finally, the results are printed. They include loss, accuracy
and the confusion matrix. It should look similar to the following:

```
Loss:  0.0138718557994
Accuracy:  0.9959
[[ 978    0    0    0    0    0    0    1    1    0]
 [   0 1132    0    1    0    0    1    1    0    0]
 [   0    1 1028    0    0    0    1    2    0    0]
 [   0    0    0 1007    0    3    0    0    0    0]
 [   0    0    0    0  980    0    0    0    0    2]
 [   0    0    0    5    0  886    1    0    0    0]
 [   2    2    0    0    1    1  952    0    0    0]
 [   0    2    0    0    0    0    0 1025    0    1]
 [   0    0    1    1    0    0    0    0  970    2]
 [   1    0    0    1    2    3    0    1    0 1001]]
```

[mnist]: http://yann.lecun.com/exdb/mnist/
[tf]: https://www.tensorflow.org/
[keras]: https://keras.io/
[numpy]: http://www.numpy.org/
[scikit]: http://scikit-learn.org/stable/
[ve]: https://virtualenv.pypa.io/en/stable/
