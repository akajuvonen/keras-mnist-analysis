# keras-mnist-analysis

MNIST data analysis using TensorFlow and Keras. Uses a Convotional Neural
Network with a relatively simple architecture to recognize hand written
digits from the [MNIST][mnist] dataset. Also features some dropout layers to
avoid overfitting and improve accuracy.

## Dependencies

You need to have:

- [TensorFlow][tf]. I recommend using GPU if possible for better performance.
- [Keras][keras] using TensorFlow backend. Offersa high-level API and a more
  *pythonic* way to develop deep learning network architectures. Cuts down the
  development time unless you need a lot of custom estimators.
- [Numpy][numpy]
- [scikit-learn][scikit]. Used for easier generation of the confusion matrix.

## Running

TODO.

[mnist]: http://yann.lecun.com/exdb/mnist/
[tf]: https://www.tensorflow.org/
[keras]: https://keras.io/
[numpy]: http://www.numpy.org/
[scikit]: http://scikit-learn.org/stable/
