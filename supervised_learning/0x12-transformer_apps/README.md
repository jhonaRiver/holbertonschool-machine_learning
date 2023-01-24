# 0x12 Transformer Applications

> A **transformer** is a deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data. It is used primarily in the fields of natural language processing (NLP) and computer vision (CV).
> Like recurrent neural networks (RNNs), transformers are designed to process sequential input data, such as natural language, with applications towards tasks such as translation and text summarization. However, unlike RNNs, transformers process the entire input all at once. The attention mechanism provides context for any position in the input sequence. For example, if the input data is a natural language sentence, the transformer does not have to process one word at a time. This allows for more parallelization than RNNs and therefore reduces training times.

At the end of this project I was able to answer these conceptual questions:

* How to use Transformers for Machine Translation
* How to write a custom train/test loop in Keras
* How to use Tensorflow Datasets

## Tasks

0. Create the class `Dataset` that loads and preps a dataset for machine translation:

    * Class constructor `def __init__(self):`
        * creates the instance attributes:
            * `data_train`, which contains the `ted_hrlr_translate/pt_to_en` `tf.data.Dataset` `train` split, loaded `as_supervided`
            * `data_valid`, which contains the `ted_hrlr_translate/pt_to_en` `tf.data.Dataset` `validate` split, loaded `as_supervided`
            * `tokenizer_pt` is the Portuguese tokenizer created from the training set
            * `tokenizer_en` is the English tokenizer created from the training set
    * Create the instance method `def tokenize_dataset(self, data):` that creates sub-word tokenizers for our dataset:
        * `data` is a `tf.data.Dataset` whose examples are formatted as a tuple `(pt, en)`
            * `pt` is the `tf.Tensor` containing the Portuguese sentence
            * `en` is the `tf.Tensor` containing the corresponding English sentence
        * The maximum vocab size should be set to `2**15`
        * Returns: `tokenizer_pt, tokenizer_en`
            * `tokenizer_pt` is the Portuguese tokenizer
            * `tokenizer_en` is the English tokenizer

1. Update the class `Dataset`:

    * Create the instance method `def encode(self, pt, en):` that encodes a translation into tokens:
        * `pt` is the `tf.Tensor` containing the Portuguese sentence
        * `en` is the `tf.Tensor` containing the corresponding English sentence
        * The tokenized sentences should include the start and end of sentence tokens
        * The start token should be indexed as `vocab_size`
        * The end token should be indexed as `vocab_size + 1`
        * Returns: `pt_tokens, en_tokens`
            * `pt_tokens` is a `np.ndarray` containing the Portuguese tokens
            * `en_tokens` is a `np.ndarray` containing the English tokens

2. Update the class `Dataset`:

    * Create the instance method `def tf_encode(self, pt, en):` that acts as a `tensorflow` wrapper for the `encode` instance method
        * Make sure to set the shape of the `pt` and `en` return tensors
    * Update the class constructor `def __init__(self):`
        * update the `data_train` and `data_validate` attributes by tokenizing the examples

3. Update the class `Dataset` to set up the data pipeline:

    * Update the class constructor `def __init__(self, batch_size, max_len):`
        * `batch_size` is the batch size for training/validation
        * `max_len` is the maximum number of tokens allowed per example sentence
        * update the `data_train` attribute by performing the following actions:
            * filter out all examples that have either sentence with more than `max_len` tokens
            * cache the dataset to increase performance
            * shuffle the entire dataset
            * split the dataset into padded batches of size `batch_size`
            * prefetch the dataset using `tf.data.experimental.AUTOTUNE` to increase performance
        * update the `data_validate` attribute by performing the following actions:
            * filter out all examples that have either sentence with more than `max_len` tokens
            * split the dataset into padded batches of size `batch_size`

4. Create the function `def create_masks(inputs, target):` that creates all masks for training/validation:

    * `inputs` is a tf.Tensor of shape `(batch_size, seq_len_in)` that contains the input sentence
    * `target` is a tf.Tensor of shape `(batch_size, seq_len_out)` that contains the target sentence
    * This function should only use `tensorflow` operations in order to properly function in the training step
    * Returns: `encoder_mask`, `combined_mask`, `decoder_mask`
        * `encoder_mask` is the `tf.Tensor` padding mask of shape `(batch_size, 1, 1, seq_len_in)` to be applied in the encoder
        * `combined_mask` is the `tf.Tensor` of shape `(batch_size, 1, seq_len_out, seq_len_out)` used in the 1st attention block in the decoder to pad and mask future tokens in the input received by the decoder. It takes the maximum between a look ahead mask and the decoder target padding mask.
        * `decoder_mask` is the `tf.Tensor` padding mask of shape `(batch_size, 1, 1, seq_len_in)` used in the 2nd attention block in the decoder.

5. Write a the function `def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):` that creates and trains a transformer model for machine translation of Portuguese to English using our previously created dataset:

    * `N` the number of blocks in the encoder and decoder
    * `dm` the dimensionality of the model
    * `h` the number of heads
    * `hidden` the number of hidden units in the fully connected layers
    * `max_len` the maximum number of tokens per sequence
    * `batch_size` the batch size for training
    * `epochs` the number of epochs to train for
    * You should use the following imports:
        * `Dataset = __import__('3-dataset').Dataset`
        * `create_masks = __import__('4-create_masks').create_masks`
        * `Transformer = __import__('5-transformer').Transformer`
    * Your model should be trained with Adam optimization with `beta_1=0.9`, `beta_2=0.98`, `epsilon=1e-9`
        * The learning rate should be scheduled using the following equation with `warmup_steps=4000`:
        * `lrate = d^(-0.5)*min(step_num^(-0.5), step_num*warmup_steps^(-1.5))`
    * Your model should use sparse categorical crossentropy loss, ignoring padded tokens
    * Your model should print the following information about the training:
        * Every 50 batches, you should print `Epoch {Epoch number}, batch {batch_number}: loss {training_loss} accuracy {training_accuracy}`
        * Every epoch, you should print `Epoch {Epoch number}: loss {training_loss} accuracy {training_accuracy}`
    * Returns the trained model

## Results

| Filename |
| ------ |
| [0-dataset.py](https://github.com/jhonaRiver/holbertonschool-machine_learning/blob/master/supervised_learning/0x12-transformer_apps/0-dataset.py)|
| [1-dataset.py](https://github.com/jhonaRiver/holbertonschool-machine_learning/blob/master/supervised_learning/0x12-transformer_apps/1-dataset.py)|
| [2-dataset.py](https://github.com/jhonaRiver/holbertonschool-machine_learning/blob/master/supervised_learning/0x12-transformer_apps/2-dataset.py)|
| [3-dataset.py](https://github.com/jhonaRiver/holbertonschool-machine_learning/blob/master/supervised_learning/0x12-transformer_apps/3-dataset.py)|
| [4-create_masks.py](https://github.com/jhonaRiver/holbertonschool-machine_learning/blob/master/supervised_learning/0x12-transformer_apps/4-create_masks.py)|
| [5-train.py](https://github.com/jhonaRiver/holbertonschool-machine_learning/blob/master/supervised_learning/0x12-transformer_apps/5-train.py)|
