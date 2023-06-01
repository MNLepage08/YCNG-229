# YCNG-229
## Neural Networks and Deep Learning

#### 1. Sequential model vs. Functional API
  - [Understanding Sequential vs. Functional API in Keras](https://www.analyticsvidhya.com/blog/2021/07/understanding-sequential-vs-functional-api-in-keras/)
  - [Sequential: ](https://keras.io/guides/sequential_model/)Create layer-by-layer. Very simple and easy to use. Sharing of layers or branching of layers is not allowed. You can’t have multiple inputs or outputs. Essentially used when each layer has exactly one input tensor and one output tensor.
  - A sequential model is not appropriate when: Your model has multiple inputs or multiple outputs. Any of our layers has multiple inputs or multiple outputs. You need to do layer sharing. You want non-linear topology (e.g. a residual connection, a multi-branch model).
  - 2 applications that we can use sequential model: Feature extraction with a sequential model and Transfer learning with a sequential model.
  - [Feature extraction: ](https://pyimagesearch.com/2019/05/27/keras-feature-extraction-on-large-datasets-with-deep-learning/)Use Keras for feature extraction on image datasets too big to fit in memory. Can use numerical data also. [Other article](https://www.tutorialspoint.com/how-can-keras-be-used-for-feature-extraction-using-a-sequential-model-using-python)
  - [Transfert Learning: ](https://pyimagesearch.com/2019/05/20/transfer-learning-with-keras-and-deep-learning/) Consists of freezing the bottom layers in a model and only training the top layers. Blueprint 1: stacking a pre-trained model and some freshly initialized classification layers. Blueprint 2: freezing all layers except the last one
  - [Functional API: ](https://keras.io/guides/functional_api/)More flexible than the sequential. Can handle models with non-linear topology, shared layers, and even multiple inputs or outputs, Directed Acyclic Graph (DAG) of layers. Model can be nested: a model can contain sub-models (since a model is just like a layer). Models with multiple inputs and outputs. (we need concatenation).

#### 2. Custom Layers & Models & Training
#### 3. Custom Callbacks, Distributed Training
#### 4. Transfert Learning and Fine-Tuning, Visual Similarity Search at Scale
#### 5. Siamese Neural Networks
#### 6. Semantic Similarity Classification with BERT
#### 7. Multi-Label Text Classification with Hugging Face
#### 8. Unsupervised Deep Embedding for Clustering Analysis
#### 9. Hyperparameter Tuning with KerasTurner
#### 10. Semantix Similarity Analysis as Scale
#### 11. Zero-Shot Text Classification with Hugging Face
#### 12. Projects Presentation
