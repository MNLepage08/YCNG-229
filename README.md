# YCNG-229: Neural Networks and Deep Learning

<p align="center">
<img width="900" alt="Capture d’écran, le 2023-06-05 à 13 25 57" src="https://github.com/MNLepage08/YCBS-255/assets/113123425/fb358716-3d9f-479a-9e98-599cb3c09692">

## :rocket: Assignments

1. [Build a small Xception network using model subclassing: ](https://github.com/MNLepage08/YCNG-229/blob/main/Small_Xception.ipynb)Collect pets image dataset with prefetching samples memory to maximize GPU utilization. Preprocess with scale and data augmentation. Build small Xception with model subclassing. Evaluate with binary cross-entropy loss fonction. Accuracy: 90%.<p>
  
2. [Create an autoencoder with subclassing and customizing fit function for image denoising: ](https://github.com/MNLepage08/YCNG-229/blob/main/Denoising_AutoEncoder.ipynb)Collect data from Fashion MNIST. Pre-process the image with a scale of 0 to 1. Add random Gaussian noise. Build an autoencoder model with binary cross-entropy metric. Valuation with train and test. Loss value of 0.26 after 10 epochs.<p>
  
3. [Build CNN model with fit function and custom training loop. Compare with transfer learning and fine-tuning: ](https://github.com/MNLepage08/YCNG-229/blob/main/Assignments_3_MNL.ipynb)Collect flower image data from TensorFlow with prefetching technique. Pre-process with scale and data augmentation. Build CNN model. Customize fit function with sparse categorical by overriding train / test step and custom training loop. Accuracy of 71% (30 epochs)). Applied transfer learning with Xception model (20 epochs) and fine-tuning (10 epochs). Accuracy of 91%.<p>
  
4. [Project - Use transformer to automatically classify, extract, and structure the information contained in the document: ](https://github.com/MNLepage08/Project-Report/blob/main/Project%20-%20MNL.pdf)Collection of scanned receipt. Use PaddleOCR model to provide data annotation. Preprocess the dataset for reorganize it before the fine-tuning step with the Hugging Face LayoutLM model. F1-Score of 94%. 

  
## :mortar_board: Courses

| # | Sessions |
| --- | --- |
| 1 | Sequential model vs. Functional API |
| 2 | Custom Layers & Models, Autoencoder |  
| 3 | GANs and Data input pipelines |
| 4 | Preprocessing |
| 5 | Custom Fit Function, DCGAN |
| 6 | Custom Training Loop, Tuning the custom training loop |
| 7 | Transfer Learning and Fine Tuning |
| 8 | Multi-Label Text Classification with Hugging Face |
| 9 | Semantic Similarity Classification with BERT |
| 10 | Siamese Neural Network |
| 11 | Zero-Shot Text Classification with Hugging Face |
| 12 | Projects Presentation |

  
## :pencil2: Notes

<details close>
<summary>1. Sequential model vs. Functional API<p></summary>
  
* [Understanding Sequential vs. Functional API in Keras](https://www.analyticsvidhya.com/blog/2021/07/understanding-sequential-vs-functional-api-in-keras/)<p>
  
* [Sequential: ](https://keras.io/guides/sequential_model/)Create layer-by-layer. Very simple and easy to use. Sharing of layers or branching of layers is not allowed. You can’t have multiple inputs or outputs. Essentially used when each layer has exactly one input tensor and one output tensor.<p>
  
* A sequential model is not appropriate when: Your model has multiple inputs or multiple outputs. Any of our layers has multiple inputs or multiple outputs. You need to do layer sharing. You want non-linear topology (e.g. a residual connection, a multi-branch model).<p>
  
* 2 applications that we can use sequential model: Feature extraction with a sequential model and Transfer learning with a sequential model.<p>
  
* [Feature extraction: ](https://pyimagesearch.com/2019/05/27/keras-feature-extraction-on-large-datasets-with-deep-learning/)Use Keras for feature extraction on image datasets too big to fit in memory. Can use numerical data also. [Other article](https://www.tutorialspoint.com/how-can-keras-be-used-for-feature-extraction-using-a-sequential-model-using-python)<p>
  
* [Transfert Learning: ](https://pyimagesearch.com/2019/05/20/transfer-learning-with-keras-and-deep-learning/) Consists of freezing the bottom layers in a model and only training the top layers. Blueprint 1: stacking a pre-trained model and some freshly initialized classification layers. Blueprint 2: freezing all layers except the last one.<p>
  
* [Functional API: ](https://keras.io/guides/functional_api/)More flexible than the sequential. Can handle models with non-linear topology, shared layers, and even multiple inputs or outputs, Directed Acyclic Graph (DAG) of layers. Model can be nested: a model can contain sub-models (since a model is just like a layer). Models with multiple inputs and outputs. (we need concatenation).

</details>
  
<details close>
<summary>2. Custom Layers & Models, Autoencoder<p></summary>

* [Dense Layer: ]([https://keras.io/guides/making_new_layers_and_models_via_subclassing/](https://machinelearningknowledge.ai/keras-dense-layer-explained-for-beginners/))Lambda layers are simple layers in TensorFlow that can be used to create some custom activation functions. But lambda layers have many limitations, especially when it comes to training these layers. So, the idea is to create custom layers that are trainable, using the inheritable Keras layers in TensorFlow — with a special focus on Dense layers.<p>
  
* <img width="340" align="right" alt="Capture d’écran, le 2023-06-04 à 14 27 16" src="https://github.com/MNLepage08/MNLepage08/assets/113123425/86450498-ba6a-4611-9942-0d3bf668a594">[A layer is a class](https://towardsdatascience.com/creating-and-training-custom-layers-in-tensorflow-2-6382292f48c2) that receives some parameters, passes them through state and computations, and passes out an output, as required by the neural network (Y = (w*X+c)). Every model architecture contains multiple layers, be it a Sequential or a Functional API. <p>**Use case as example:** You want to develop a machine translation (LSTM seq to seq), we need to use a mechanism which called attention (ex: we might pay attention to understand some particulars caps in the phrase to understand the context). No attention layer in keras and needs custom layer.<p>
  
* [Custom Layers: ](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)The most recommended way to create your own customized layer is extending the tf.keras.Layer and implement: **Init:** specifies all input-independent initialization (ex:  number of units in my dense layer). **Call:** specifies the computation done buy the layer (Y = (w*X+c)). **Build:** creates the weights (states) of the layer (this is just a style convention since you can create weights in init as well).<p>
  
* The Layer class: the combination of state (weights) and some computation. Layers can have non-trainable weights. Best practice: deferring weight creation until the shape of the inputs is known. Layers are recursively composable. The add_loss() method. You can optionally enable serialization on your layers. Privileged training argument in the call() method.<p>
  
* Layer class: to define inner computation blocks. Model class: to define the object that we will train. (model class is compose of layer class).<p>
  
* Example: In a ResNet50 model, we would have several ResNet blocks subclassing Layer, and a single Model encompassing the entire ResNet50 network. The model class has the same API as Layer, with the following differences: It exposes built-in training, evaluation, and prediction loops (model.fit(), model.evaluate(), model.predict()) It exposes the list of its inner layers, via the model.layers property. It exposes saving and serialization APIs.<p>
  
* [Introduction aux encodeurs automatiques](https://www.tensorflow.org/tutorials/generative/autoencoder?hl=fr#define_a_convolutional_autoencoder)<p>
  
* [Convolutional autoencoder for image denoising](https://keras.io/examples/vision/autoencoder/)<p>
  
* [Transformer Library (NLP, Computer Vision, Tool Pipeline)](https://huggingface.co/)<p>
 
</details>
  
<details close>
<summary>3. GANs and Data input pipelines<p></summary>

* [GANs with Keras and TensorFlow: ](https://pyimagesearch.com/2020/11/16/gans-with-keras-and-tensorflow/)In GANs, two models are trained simultaneously (adversarial process): **Generator:** learns to create images that look real (the artist). **Discriminator:** learns to tell real images apart from fakes (the art critic)<p>
  
* <img width="306" align="right" alt="Capture d’écran, le 2023-06-04 à 16 28 42" src="https://github.com/MNLepage08/YCNG-229/assets/113123425/cf148f94-e254-4e2e-ac8e-81c0548720ab">GANs are usually trained using the following steps.<p>At first the generator is doing a poor job though it progressively becomes better at creating images that look real, while the discriminator becomes better at telling them apart. It reaches to the point where the discriminator is no longer able to spot the difference between the images.<p>
  
* [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)<p>
  
* [How to Develop a GAN for Generating MNIST Handwritten Digits:](https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/)<p>
  
* [Generative Adversarial Network (GAN) using Keras](https://medium.datadriveninvestor.com/generative-adversarial-network-gan-using-keras-ce1c05cfdfd3)<p>
  
* [Data input pipelines: ](https://www.tensorflow.org/guide/data?hl=fr#using_tfdata_with_tfkeras)The tf.data API enables us to build complex input pipelines from simple, reusable pieces. **Example 1. Image model:** aggregate data from files in a distributed file system, apply random perturbations to each image, merge randomly selected images into a batch for training. **Example 2. Text model:** extracting symbols from raw text data, converting them to embedding identifiers with a lookup table, batching together sequences of different lengths.<p>
  
* The tf.data API makes it possible to handle large amounts of data, read from different data formats, and perform complex transformations. The tf.data API introduces a tf.data.Dataset abstraction representing a sequence of elements, in which each element consists of one or more components. **Example:** In an image pipeline, an element might be a single training example, with a pair of tensor components representing the image and its label.<p>
  
* There are two distinct ways to create a dataset: A data source constructs a Dataset from data stored in memory or in one or more files. A data transformation constructs a dataset from one or more tf.data.Dataset objects.<p>
  
* Once you have a Dataset object, you can transform it into a new Dataset by chaining method calls on the tf.data.Dataset object. **Example:**  Apply per-element transformations such as Dataset.map. Apply multi-element transformations such as Dataset.batch.<p>
  
* **Reading input data:** NumPy arrays, Python generators, TFRecord Data, Text Data, CSV Data, Set of files.<p>
  
* **Batching dataset elements:** equivalent to update the weights or the gradients. Pass the entire dataset.<p>
  
* **Processing multiple epochs:** The simple way to iterate over a dataset un multiple epochs is to use the Dataset.repeat() transformation. Dataset.batch applied after Dataset.repeat will yield batches that straddle epoch boundaries.<p>
  
* **Randomly shuffling input data:** The Dataset.shuffle() transformation maintains a fixed-size buffer and choses the next element uniformly at random from that buffer.<p>
  
* **Preprocessing data:** When training a neural network on real-world image data, it is often necessary to convert images of different sizes to a common size, so they may be batched into a fixed size. Dataset.cache: keep the data in memory after they’re loaded off disk during the first epoch. This will ensure the dataset does not become a bottleneck while training your model. If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache. Dataset.prefetch: overlaps data preprocessing and model execution while training. Tf.data.AUTOTUNE, the best value of buffer size for you. Train_flag = false, don’t need shuffle the validation set.<p>
  
* [Classement des images](https://www.tensorflow.org/tutorials/images/classification?hl=fr)<p>
  
* [Simple MNIST convnet](https://keras.io/examples/vision/mnist_convnet/)<p>
  
</details>

<details close>
<summary>4. Preprocessing<p></summary>
  
* [Asynchronous vs. Synchronous: ](http://www.cse.unsw.edu.au/~billw/mldict.html)When a neural network is viewed as a collection of connected computation devices, the question arises whether the nodes/devices share a common clock, so that they all perform their computations (‘fire’) as the same time (i.e. synchronously - where the gradients from the workers are aggreged and then applied all at once) or whether they fire at different times, e.g. they may fire equally often on average, but in a random sequence (i.e. asynchronously - where workers apply the gradients without waiting for others).<p>
  
* [Working with preprocessing layers](https://keras.io/guides/preprocessing_layers/)<p>
  
* [Text classification from scratch: ](https://keras.io/examples/nlp/text_classification_from_scratch/)Data Extract & Transform. Two options to vectorize the data: **Option 1:** Make it part of the model. With this option, preprocessing will happen on device, synchronously with the rest of the model execution, meaning that it will benefit from GPU acceleration. If you are training on GPU, this is the best option for all image preprocessing and data augmentation layers. **Option 2:** Apply it to the tf.Dataset, so as to obtain a dataset that yields batches of preprocessed data. With this option, the preprocessing will happen on a CPU, asynchronously, and will be buffered before going into the model. This is the best option for TextVectorization, and all structured data preprocessing layers. It can also be a good option if you are training on the CPU and you use image preprocessing layers.<p>
  
* The TextVectorization layer can only be executed on a CPU, as it mostly a dictionary lookup operation. Therefore, if you are training your model on GPU, you should put the TextVectorization layer in the td.data.pipeline to get the best performance.<p>
  
* Benefits of doing preprocessing inside the model at inference time: Even if we go with option 2, we may later want to export an inference-only end-to-end model that will include the preprocessing layers. The key benefit to doing this is that it makes the model portable. When all data preprocessing is part of the model, everyone can load and use the model without having to be aware of how each feature is expected to be encoded and normalized. The model will be able to process raw data as it is.<p>
  
* How to make it an end-to-end model? Given that we initially put the preprocessing layer in the tf.data pipeline, we can export an inference model that packages the preprocessing. This model is capable of processing raw strings. The solution is to instantiate a new model that chains the preprocessing layers and the training model.<p>
  
* [Image classification from scratch: ](https://keras.io/examples/vision/image_classification_from_scratch/)Create Dataset & Data Augmentation. Two options to preprocess the data: **Option 1:** Make it part of the model. **Option 2:** apply it to the dataset.
  
</details>
  
  
<details close>
<summary>5. Custom Fit Function, DCGAN<p></summary>

* [Customize fit(): ](https://keras.io/guides/customizing_what_happens_in_fit/)We should override the training_step function of the model class. This is the function that is called by fit() for every batch of data. We. Will then be able to call fit() as usual and it will be running our own learning algorithm.<p>

* [tf.GratientTape: ](https://www.tensorflow.org/api_docs/python/tf/GradientTape)Metrics remarks - Each metrics in Keras has three main methods: **Update_state:** it uses the targets y_true and the model predictions y_pred to update the state variables. **Result:** it uses the state variables to compute the final results. **Reset_state:** it reinitializes the state of the metric (each epoc).<p>
  
* Lower level: We can skip passing loss function or metrics in compile().<p>
  
* [Support sample_weight and class_weight:](https://keras.io/guides/training_with_built_in_methods/#sample-weights) Unpack sample_weight from the data argument. Pass it to compiled_loss and compiled_metrics. Class weights and Sample weights have different objectives in Keras but both are used for decreasing the training loss of an artificial neural network.<p>
  
* **Scenarios where we usually use class weights:** When the data contains an imbalanced number of classes. When some classes need more attention in some scenarios even with a balanced data set. When we consider the F1 score as a more important metric than Accuracy.<p>
  
* **Scenarios where we usually use sample weights:** When some samples need more attention according to time and characteristics. When we believe that giving priority to the latest or oldest samples may increase the accuracy of the model. When the model is required to adapt quickly to data generated at the latest time period. When we believe that the real information in training data is segregated only to a fewer number of samples.<p>
  
* [TF dataset does not support class rate and need implemented the get_sample_weight function](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data?hl=fr#oversampling)<p>

* [DCGAN to generate face images](https://keras.io/examples/generative/dcgan_overriding_train_step/)

</details>
  
  
<details close>
<summary>6. Custom Training Loop, Tuning the custom training loop<p></summary>
  
* [Getting started with KerasTurner: ](https://keras.io/guides/keras_tuner/getting_started/#keep-keras-code-separate)Define sesarch space ([Optuna: ](https://optuna.org)An open source hyperparameter optimization framework to automate hyperparameter search.), Search start (RandomSearch, BayesOptimization, Hyperband, SklearnTuner), Query results, Retrain model.<p>
  
* **Tune model training:** **Build:** creates a keras model using the hyperparameters and returns it. **Fit:** accepts the model returned by HyperModel.build(), hp and all the arguments passed to search().<p>
  
* **Tune data preprocessing:** To tune data preprocessing, we just add an additional step in HyperModel.fit(), where we can access the dataset from the arguments.<p>

* **Specify the tuning objective:** Built-in metric as the objective. Custom metric as the objective.<p>
  
* [Tuning the custom training loop: ](https://keras.io/guides/keras_tuner/custom_tuner/)We will subclass the HyperModel class and write a custom training loop by overriding HyperModel.fit().<p>
  
* [Automatic Hyperparameter Optimization with keras tuner](https://blog.paperspace.com/hyperparameter-optimization-with-keras-tuner/)<p>
  
</details>
  
  
<details close>
<summary>7. Transfer Learning and Fine Tuning<p></summary>

* [Transfer Learning & Fine Tuning: ](https://keras.io/guides/transfer_learning/)Transfer learning consists of taking features learned on one problem, and leveraging them on a new, similar problem. It is usually done for tasks where our dataset has too little data to train a full-scale model from scratch.<p>
  1. Take layers from a previously trained model.
  2. Freeze them, so as to avoid destroying any of the information they contain during future training rounds.
  3. Add some new, trainable layers on top of the frozen layers. They will learn to turn the old features into predictions on a new dataset.
  4. Train the new layers on our dataset.

* One last, optional step, is fine-tuning, which consists of unfreezing the entire or part of the model we obtained above, and re-training it on the new data with a very low learning rate. This can potentially achieve meaningful improvements, by incrementally adapting the pretrained features to the new data. It can also potentially lead to quick overfitting.

* Note that an alternative, more lightweight workflow could also be:<p>
  1. Instantiate a base model and load pre-trained weights into it.
  2. Run the new dataset through it and record the output of one (or several) layers from the base model. This is called feature extraction.
  3. Use that output as input data for a new, smaller model.
 
* **Fine-tuning:** Once our model has converged on the new data, we can try to unfreeze all or part of the base model and retrain the whole model end-to-end with a very low learning rate.<p>

  It is critical to only do this step after the model with frozen layers has been trained to convergence. If we mix randomly-initialized trainable layers with trainable layers that hold pre-trained features, the randomly-initialized layers will cause very large gradient updates during training, which will destroy our pre-trained features.<p>
  
  It is also critical to use a very low learning rate at this stage, because we are training a much larger model than in the first round of training, on a dataset that is typically very small. As a result, we are at risk of overfitting very quickly if we apply large weight updates.<p>
  
  When we unfreeze a model that contains BatchNormalization layers to do fine-tuning, we should keep the BatchNormalization layers in inference mode by passing training=False when calling the base model. 
  Otherwise, the updates applied to the non-trainable weights will destroy what the model has learned.


  
  
</details>


<details close>
<summary>11. Zero-Shot Text Classification with Hugging Face <p></summary>

* Zero-shot Learning is a setup in which a model can learn to recognize things that it hasn’t explicitly seen before in training. This is exactly how zero shot classification works. We have a pre trained model (eg. a language model) which serves as the knowledge base since it has been trained on a huge amount of text from many websites. For any type of task, we give relevant class descriptors and let the model infer what the task actually is.
  
* There are different zero-shot learning approaches, but a commonality is that auxiliary information such as textual descriptions are used or encoded during the training process instead of explicit labels. Needless to say, the more labelled data we provide, the better the results would be. And sometimes, zero-shot learning doesn’t work very well. If we have a few samples of labelled data but not enough for fine tuning, few shots is the way to go. Zero shot and few shot learning methods are reducing the reliance on annotated data. The GPT-2 and GPT-3,GPT-4 models have shown remarkable results to prove this.
  
* [Understanding Zero-Shot Learning](https://towardsdatascience.com/understanding-zero-shot-learning-making-ml-more-human-4653ac35ccab)
  
* [Zero and Few Shot Learning](https://towardsdatascience.com/zero-and-few-shot-learning-c08e145dc4ed)

* [Understanding Contrastive Learning](https://towardsdatascience.com/understanding-contrastive-learning-d5b19fd96607)
  
* [Pre-trained model](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli)
  
* [Zero-Shot Learning in Modern NLP](https://joeddav.github.io/blog/2020/05/29/ZSL.html)
  
</details>
