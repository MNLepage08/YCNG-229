# YCNG-229: Neural Networks and Deep Learning

<p align="center">
<img width="1000" alt="Capture d’écran, le 2023-06-04 à 13 49 19" src="https://github.com/MNLepage08/YCNG-232/assets/113123425/649c81dd-0051-4014-b5e0-f4f0b163b3c3">

## :rocket: Assignments

1. [Build a small Xception network using model subclassing: ](https://github.com/MNLepage08/YCNG-229/blob/main/Small_Xception.ipynb)Collect pets image dataset with prefetching samples memory to maximize GPU utilization. Preprocess with scale and data augmentation. Build small Xception with model subclassing. Evaluate with binary cross-entropy loss fonction. Accuracy: 90%.<p>
  
2. [Create an autoencoder with subclassing and customizing fit function for image denoising: ](https://github.com/MNLepage08/YCNG-229/blob/main/Denoising_AutoEncoder.ipynb)Collect data from Fashion MNIST. Pre-process the image with a scale of 0 to 1. Add random Gaussian noise. Build an autoencoder model with binary cross-entropy metric. Valuation with train and test. Loss value of 0.26 after 10 epochs.<p>
  
3. [Build CNN model with fit function and custom training loop. Compare with transfer learning and fine-tuning: ](https://github.com/MNLepage08/YCNG-229/blob/main/Assignments_3_MNL.ipynb)Collect flower image data from TensorFlow with prefetching technique. Pre-process with scale and data augmentation. Build CNN model. Customize fit function with sparse categorical by overriding train / test step and custom training loop. Accuracy of 71% (30 epochs)). Applied transfer learning with Xception model (20 epochs) and fine-tuning (10 epochs). Accuracy of 91%.<p>
  
4. [Project - Use transformer to automatically classify, extract, and structure the information contained in the document: ](https://github.com/MNLepage08/Project-Report/blob/main/Project%20-%20MNL.pdf)Collection of scanned receipt. Use PaddleOCR model to provide data annotation. Preprocess the dataset for reorganize it before the fine-tuning step with the Hugging Face LayoutLM model. F1-Score of 94%. 

  
## :mortar_board: Courses

| # | Sessions |
| --- | --- |
| 1 | Sequential model vs. Functional API |
| 2 | Custom Layers & Models |  
| 3 | Custom Callbacks, Distributed Training |
| 4 | Transfert Learning and Fine-Tuning, Visual Similarity Search at Scale |
| 5 | Siamese Neural Networks |
| 6 | Semantic Similarity Classification with BERT |
| 7 | Multi-Label Text Classification with Hugging Face |
| 8 | Unsupervised Deep Embedding for Clustering Analysis |
| 9 | Hyperparameter Tuning with KerasTurner |
| 10 | Semantix Similarity Analysis as Scale |
| 11 | Zero-Shot Text Classification with Hugging Face |
| 12 | Projects Presentation |

  
## :pencil2: Notes

<details close>
<summary>1. Sequential model vs. Functional API</summary>
  
* [Understanding Sequential vs. Functional API in Keras](https://www.analyticsvidhya.com/blog/2021/07/understanding-sequential-vs-functional-api-in-keras/)
* [Sequential: ](https://keras.io/guides/sequential_model/)Create layer-by-layer. Very simple and easy to use. Sharing of layers or branching of layers is not allowed. You can’t have multiple inputs or outputs. Essentially used when each layer has exactly one input tensor and one output tensor.
* A sequential model is not appropriate when: Your model has multiple inputs or multiple outputs. Any of our layers has multiple inputs or multiple outputs. You need to do layer sharing. You want non-linear topology (e.g. a residual connection, a multi-branch model).
* 2 applications that we can use sequential model: Feature extraction with a sequential model and Transfer learning with a sequential model.
* [Feature extraction: ](https://pyimagesearch.com/2019/05/27/keras-feature-extraction-on-large-datasets-with-deep-learning/)Use Keras for feature extraction on image datasets too big to fit in memory. Can use numerical data also. [Other article](https://www.tutorialspoint.com/how-can-keras-be-used-for-feature-extraction-using-a-sequential-model-using-python)
* [Transfert Learning: ](https://pyimagesearch.com/2019/05/20/transfer-learning-with-keras-and-deep-learning/) Consists of freezing the bottom layers in a model and only training the top layers. Blueprint 1: stacking a pre-trained model and some freshly initialized classification layers. Blueprint 2: freezing all layers except the last one
* [Functional API: ](https://keras.io/guides/functional_api/)More flexible than the sequential. Can handle models with non-linear topology, shared layers, and even multiple inputs or outputs, Directed Acyclic Graph (DAG) of layers. Model can be nested: a model can contain sub-models (since a model is just like a layer). Models with multiple inputs and outputs. (we need concatenation).

</details>
  
<details close>
<summary>2. Custom Layers & Models</summary>
  
* [Dense Layer: ]([https://keras.io/guides/making_new_layers_and_models_via_subclassing/](https://machinelearningknowledge.ai/keras-dense-layer-explained-for-beginners/))Lambda layers are simple layers in TensorFlow that can be used to create some custom activation functions. But lambda layers have many limitations, especially when it comes to training these layers. So, the idea is to create custom layers that are trainable, using the inheritable Keras layers in TensorFlow — with a special focus on Dense layers.
* <img width="340" align="right" alt="Capture d’écran, le 2023-06-04 à 14 27 16" src="https://github.com/MNLepage08/MNLepage08/assets/113123425/86450498-ba6a-4611-9942-0d3bf668a594">[A layer is a class](https://towardsdatascience.com/creating-and-training-custom-layers-in-tensorflow-2-6382292f48c2) that receives some parameters, passes them through state and computations, and passes out an output, as required by the neural network (Y = (w*X+c)). Every model architecture contains multiple layers, be it a Sequential or a Functional API. <p>**Use case as example:** You want to develop a machine translation (LSTM seq to seq), we need to use a mechanism which called attention (ex: we might pay attention to understand some particulars caps in the phrase to understand the context). No attention layer in keras and needs custom layer.
* [Custom Layers: ](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)The most recommended way to create your own customized layer is extending the tf.keras.Layer and implement: **Init:** specifies all input-independent initialization (ex:  number of units in my dense layer). **Call:** specifies the computation done buy the layer (Y = (w*X+c)). **Build:** creates the weights (states) of the layer (this is just a style convention since you can create weights in init as well).
* The Layer class: the combination of state (weights) and some computation. Layers can have non-trainable weights. Best practice: deferring weight creation until the shape of the inputs is known. Layers are recursively composable. The add_loss() method. You can optionally enable serialization on your layers. Privileged training argument in the call() method.
* Layer class: to define inner computation blocks. Model class: to define the object that we will train. (model class is compose of layer class).
* Example: In a ResNet50 model, we would have several ResNet blocks subclassing Layer, and a single Model encompassing the entire ResNet50 network. The model class has the same API as Layer, with the following differences: It exposes built-in training, evaluation, and prediction loops (model.fit(), model.evaluate(), model.predict()) It exposes the list of its inner layers, via the model.layers property. It exposes saving and serialization APIs.
* [Introduction aux encodeurs automatiques](https://www.tensorflow.org/tutorials/generative/autoencoder?hl=fr#define_a_convolutional_autoencoder)
* [Convolutional autoencoder for image denoising](https://keras.io/examples/vision/autoencoder/)
* [Transformer Library (NLP, Computer Vision, Tool Pipeline)](https://huggingface.co/)

  
</details>
