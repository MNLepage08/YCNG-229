# YCNG-229: Neural Networks and Deep Learning

## :rocket: Assignments

1. [Build a small Xception network using model subclassing: ](https://github.com/MNLepage08/YCNG-229/blob/main/Small_Xception.ipynb)Collect pets image dataset with prefetching samples memory to maximize GPU utilization. Preprocess with scale and data augmentation. Build small Xception with model subclassing. Evaluate with binary cross-entropy loss fonction. Accuracy: 90%.<p>
  
2. [Create an autoencoder with subclassing and customizing fit function for image denoising: ](https://github.com/MNLepage08/YCNG-229/blob/main/Denoising_AutoEncoder.ipynb)Collect data from Fashion MNIST. Pre-process the image with a scale of 0 to 1. Add random Gaussian noise. Build an autoencoder model with binary cross-entropy metric. Valuation with train and test. Loss value of 0.26 after 10 epochs.<p>
  
3. [Build CNN model with fit function and custom training loop. Compare with transfer learning and fine-tuning: ](https://github.com/MNLepage08/YCNG-229/blob/main/Assignments_3_MNL.ipynb)Collect flower image data from TensorFlow with prefetching technique. Pre-process with scale and data augmentation. Build CNN model. Customize fit function with sparse categorical by overriding train / test step and custom training loop. Accuracy of 71% (30 epochs)). Applied transfer learning with Xception model (20 epochs) and fine-tuning (10 epochs). Accuracy of 91%.<p>
  
4. [Project - Use transformer to automatically classify, extract, and structure the information contained in the document: ](https://github.com/MNLepage08/Project-Report/blob/main/Project%20-%20MNL.pdf)Collection of scanned receipt. Use PaddleOCR model to provide data annotation. Preprocess the dataset for reorganize it before the fine-tuning step with the Hugging Face LayoutLM model. F1-Score of 94%. 

  
## :mortar_board: Courses

| # | Sessions |
| --- | --- |
| 1 | Sequential model vs. Functional API |
| 2 | Custom Layers & Models & Training |  
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
