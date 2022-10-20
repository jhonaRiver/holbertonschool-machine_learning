# 0x0C Neural style transfer

> Neural style transfer project for Holberton. See [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf)

At the end of this project I was able to answer these conceptual questions:

* What is Neural Style Transfer?
* What is a gram matrix?
* How to calculate style cost
* How to calculate content cost
* What is `Tensorflow`‘s Eager Execution?
* What is Gradient Tape and how do you use it?
* How to perform Neural Style Transfer

## Tasks

0. Create a class `NST` that performs tasks for neural style transfer:

    * Public class attributes:
        * `style_layers` = `['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']`
        * `content_layer` = `'block5_conv2'`
    * Class constructor: `def __init__(self, style_image, content_image, alpha=1e4, beta=1):`
        * `style_image` - the image used as a style reference, stored as a `numpy.ndarray`
        * `content_image` - the image used as a content reference, stored as a `numpy.ndarray`
        * `alpha` - the weight for content cost
        * `beta` - the weight for style cost
        * if `style_image` is not a `np.ndarray` with the shape `(h, w, 3)`, raise a `TypeError` with the message `style_image must be a numpy.ndarray with shape (h, w, 3)`
        * if `content_image` is not a `np.ndarray` with the shape `(h, w, 3)`, raise a `TypeError` with the message `content_image must be a numpy.ndarray with shape (h, w, 3)`
        * if `alpha` is not a non-negative number, raise a `TypeError` with the message `alpha must be a non-negative number`
        * if `beta` is not a non-negative number, raise a `TypeError` with the message `beta must be a non-negative number`
        * Sets Tensorflow to execute eagerly
        * Sets the instance attributes:
            * `style_image` - the preprocessed style image
            * `content_image` - the preprocessed content image
            * `alpha` - the weight for content cost
            * `beta` - the weight for style cost
    * Static Method: `def scale_image(image):` that rescales an image such that its pixels values are between 0 and 1 and its largest side is 512 pixels
        * `image` - a `numpy.ndarray` of shape `(h, w, 3)` containing the image to be scaled
        * if `image` is not a `np.ndarray` with the shape `(h, w, 3)`, raise a `TypeError` with the message `image must be a numpy.ndarray with shape (h, w, 3)`
        * The scaled image should be a `tf.tensor` with the shape `(1, h_new, w_new, 3)` where `max(h_new, w_new)` == `512` and `min(h_new, w_new)` is scaled proportionately
        * The image should be resized using bicubic interpolation
        * After resizing, the image’s pixel values should be rescaled from the range `[0, 255]` to `[0, 1]`.
        * Returns: the scaled image

1. Update the class `NST` to load the model for neural style transfer:

    * Update the class constructor: `def __init__(self, style_image, content_image, alpha=1e4, beta=1):`
        * `style_image` - the image used as a style reference, stored as a `numpy.ndarray`
        * `content_image` - the image used as a content reference, stored as a `numpy.ndarray`
        * `alpha` - the weight for content cost
        * `beta` - the weight for style cost
        * if `style_image` is not a `np.ndarray` with the shape `(h, w, 3)`, raise a `TypeError` with the message `style_image must be a numpy.ndarray with shape (h, w, 3)`
        * if `content_image` is not a `np.ndarray` with the shape `(h, w, 3)`, raise a `TypeError` with the message `content_image must be a numpy.ndarray with shape (h, w, 3)`
        * if `alpha` is not a non-negative number, raise a `TypeError` with the message `alpha must be a non-negative number`
        * if `beta` is not a non-negative number, raise a `TypeError` with the message `beta must be a non-negative number`
        * Sets `Tensorflow` to execute eagerly
        * Sets the instance attributes:
            * `style_image` - the preprocessed style image
            * `content_image` - the preprocessed content image
            * `alpha` - the weight for content cost
            * `beta` - the weight for style cost
            * `model` - the `Keras` model used to calculate cost
    * Public Instance Method: `def load_model(self):`
        * creates the model used to calculate cost
        * the model should use the `VGG19` `Keras` model as a base
        * the model’s input should be the same as the `VGG19` input
        * the model’s output should be a list containing the outputs of the `VGG19` layers listed in `style_layers` followed by `content _layer`
        * saves the model in the instance attribute `model`

2. Update the class `NST` to be able to calculate gram matrices:

    * Static Method: `def gram_matrix(input_layer):`
        * `input_layer` - an instance of `tf.Tensor` or `tf.Variable` of shape `(1, h, w, c)` containing the layer output whose gram matrix should be calculated
        * if `input_layer` is not an instance of `tf.Tensor` or `tf.Variable` of rank 4, raise a `TypeError` with the message `input_layer must be a tensor of rank 4`
        * Returns: a `tf.Tensor` of shape `(1, c, c)` containing the gram matrix of `input_layer`

3. Update the class `NST` to extract the style and content features:

    * Public Instance Method: `def generate_features(self):`
        * extracts the features used to calculate neural style cost
        * Sets the public instance attributes:
            * `gram_style_features` - a list of gram matrices calculated from the style layer outputs of the style image
            * `content_feature` - the content layer output of the content image
    * Update the class constructor: `def __init__(self, style_image, content_image, alpha=1e4, beta=1):`
        * `style_image` - the image used as a style reference, stored as a `numpy.ndarray`
        * `content_image` - the image used as a content reference, stored as a `numpy.ndarray`
        * `alpha` - the weight for content cost
        * `beta` - the weight for style cost
        * if `style_image` is not a `np.ndarray` with the shape `(h, w, 3)`, raise a `TypeError` with the message `style_image must be a numpy.ndarray with shape (h, w, 3)`
        * if `content_image` is not a `np.ndarray` with the shape `(h, w, 3)`, raise a `TypeError` with the message `content_image must be a numpy.ndarray with shape (h, w, 3)`
        * if `alpha` is not a non-negative number, raise a `TypeError` with the message `alpha must be a non-negative number`
        * if `beta` is not a non-negative number, raise a `TypeError` with the message `beta must be a non-negative number`
        * Sets `Tensorflow` to execute eagerly
        * Sets the instance attributes:
            * `style_image` - the preprocessed style image
            * `content_image` - the preprocessed content image
            * `alpha` - the weight for content cost
            * `beta` - the weight for style cost
            * `model` - the Keras model used to calculate cost
            * `gram_style_features` - a list of gram matrices calculated from the style layer outputs of the style image
            * `content_feature` - the content layer output of the content image

4. Update the class `NST` to calculate the style cost for a single layer:

    * Public Instance Method: `def layer_style_cost(self, style_output, gram_target):`
        * Calculates the style cost for a single layer
        * `style_output` - `tf.Tensor` of shape `(1, h, w, c)` containing the layer style output of the generated image
        * `gram_target` - `tf.Tensor` of shape `(1, c, c)` the gram matrix of the target style output for that layer
        * if `style_output` is not an instance of `tf.Tensor` or `tf.Variable` of rank 4, raise a `TypeError` with the message `style_output must be a tensor of rank 4`
        * if `gram_target` is not an instance of `tf.Tensor` or `tf.Variable` with shape `(1, c, c)`, raise a `TypeError` with the message `gram_target must be a tensor of shape [1, {c}, {c}]` where `{c}` is the number of channels in `style_output`
        * Returns: the layer’s style cost

5. Update the class `NST` to calculate the style cost:

    * Public Instance Method: `def style_cost(self, style_outputs):`
        * Calculates the style cost for generated image
        * `style_outputs` - a list of `tf.Tensor` style outputs for the generated image
        * if `style_outputs` is not a list with the same length as `self.style_layers`, raise a `TypeError` with the message `style_outputs must be a list with a length of {l}` where `{l}` is the length of `self.style_layers`
        * each layer should be weighted evenly with all weights summing to `1`
        * Returns: the style cost

6. Update the class `NST` to calculate the content cost:

    * Public Instance Method: `def content_cost(self, content_output):`
        * Calculates the content cost for the generated image
        * `content_output` - a `tf.Tensor` containing the content output for the generated image
        * if `content_output` is not an instance of `tf.Tensor` or `tf.Variable` with the same shape as `self.content_feature`, raise a `TypeError` with the message `content_output must be a tensor of shape {s}` where `{s}` is the shape of `self.content_feature`
        * Returns: the content cost

7. Update the class `NST` to calculate the total cost:

    * Public Instance Method: `def total_cost(self, generated_image):`
        * Calculates the total cost for the generated image
        * `generated_image` - a `tf.Tensor` of shape `(1, nh, nw, 3)` containing the generated image
        * if `generated_image` is not an instance of `tf.Tensor` or `tf.Variable` with the same shape as `self.content_image`, raise a `TypeError` with the message `generated_image must be a tensor of shape {s}` where `{s}` is the shape of `self.content_image`
        * Returns: `(J, J_content, J_style)`
            * `J` is the total cost
            * `J_content` is the content cost
            * `J_style` is the style cost

8. Update the class `NST` to compute the gradients for the generated image:

    * Public Instance Method: `def compute_grads(self, generated_image):`
        * Calculates the gradients for the `tf.Tensor` generated image of shape `(1, nh, nw, 3)`
        * if `generated_image` is not an instance of `tf.Tensor` or `tf.Variable` with the same shape as `self.content_image`, raise a `TypeError` with the message `generated_image must be a tensor of shape {s}` where `{s}` is the shape of `self.content_image`
        * Returns: `gradients, J_total, J_content, J_style`
            * `gradients` is a `tf.Tensor` containing the gradients for the generated image
            * `J_total` is the total cost for the generated image
            * `J_content` is the content cost for the generated image
            * `J_style` is the style cost for the generated image

9. Update the class `NST` to generate the neural style transfered image:

    * Public Instance Method: `def generate_image(self, iterations=1000, step=None, lr=0.01, beta1=0.9, beta2=0.99)`
        * `iterations` - the number of iterations to perform gradient descent over
        * `step` - if not `None`, the step at which you should print information about the training, including the final iteration:
            * print `Cost at iteration {i}: {J_total}, content {J_content}, style {J_style}`
            * `i` is the iteration
            * `J_total` is the total cost
            * `J_content` is the content cost
            * `J_style` is the style cost
        * `lr` - the learning rate for gradient descent
        * `beta1` - the beta1 parameter for gradient descent
        * `beta2` - the beta2 parameter for gradient descent
        * if `iterations` is not an integer, raise a `TypeError` with the message `iterations must be an integer`
        * if `iterations` is not positive, raise a `ValueError` with the message `iterations must be positive`
        * if `step` is not `None` and not an integer, raise a `TypeError` with the message `step must be an integer`
        * if `step` is not `None` and not positive or less than `iterations`, raise a `ValueError` with the message `step must be positive and less than iterations`
        * if `lr` is not a float or an integer, raise a `TypeError` with the message `lr must be a number`
        * if `lr` is not positive, raise a `ValueError` with the message `lr must be positive`
        * if `beta1` is not a float, raise a `TypeError` with the message `beta1 must be a float`
        * if `beta1` is not in the range `[0, 1]`, raise a `ValueError` with the message `beta1 must be in the range [0, 1]`
        * if `beta2` is not a float, raise a `TypeError` with the message `beta2 must be a float`
        * if `beta2` is not in the range `[0, 1]`, raise a `ValueError` with the message `beta2 must be in the range [0, 1]`
        * gradient descent should be performed using Adam optimization
        * the generated image should be initialized as the content image
        * keep track of the best cost and the image associated with that cost
        * Returns: `generated_image, cost`
            * `generated_image` is the best generated image
            * `cost` is the best cost

10. Update the class `NST` to account for variational cost:

    * Update the class constructor: `def __init__(self, style_image, content_image, alpha=1e4, beta=1, var=10):`
        * `style_image` - the image used as a style reference, stored as a `numpy.ndarray`
        * `content_image` - the image used as a content reference, stored as a `numpy.ndarray`
        * `alpha` - the weight for content cost
        * `beta` - the weight for style cost
        * `var` is the weight for the variational cost
        * if `style_image` is not a `np.ndarray` with the shape `(h, w, 3)`, raise a `TypeError` with the message `style_image must be a numpy.ndarray with shape (h, w, 3)`
        * if `content_image` is not a `np.ndarray` with the shape `(h, w, 3)`, raise a `TypeError` with the message `content_image must be a numpy.ndarray with shape (h, w, 3)`
        * if `alpha` is not a non-negative number, raise a `TypeError` with the message `alpha must be a non-negative number`
        * if `beta` is not a non-negative number, raise a `TypeError` with the message `beta must be a non-negative number`
        * if `var` is not a non-negative number, raise a `TypeError` with the message `var must be a non-negative number`
        * Sets `Tensorflow` to execute eagerly
        * Sets the instance attributes:
            * `style_image` - the preprocessed style image
            * `content_image` - the preprocessed content image
            * `alpha` - the weight for content cost
            * `beta` - the weight for style cost
            * `model` - the Keras model used to calculate cost
            * `gram_style_features` - a list of gram matrices calculated from the style layer outputs of the style image
            * `content_feature` - the content layer output of the content image
    * Static Method: `def variational_cost(generated_image):`
        * Calculates the variational cost for the generated image
        * `generated_image` - a `tf.Tensor` of shape `(1, nh, nw, 3)` containing the generated image
        * Returns: the variational cost
    * Public Instance Method: `def total_cost(self, generated_image):`
        * Calculates the total cost for the generated image
        * `generated_image` - a `tf.Tensor` of shape `(1, nh, nw, 3)` containing the generated image
        * Returns: `(J, J_content, J_style, J_var)`
            * `J` is the total cost
            * `J_content` is the content cost
            * `J_style` is the style cost
            * `J_var` is the variational cost
    * Public Instance Method: `def generate_image(self, iterations=1000, step=None, lr=0.01, beta1=0.9, beta2=0.99)`
        * `iterations` - the number of iterations to perform gradient descent over
        * `step` - if not `None`, the step at which you should print information about the training, including the final iteration:
            * print `Cost at iteration {i}: {J_total}, content {J_content}, style {J_style}, var {J_var}`
            * `i` is the iteration
            * `J_total` is the total cost
            * `J_content` is the content cost
            * `J_style` is the style cost
            * `J_var` is the variational cost
        * `lr` - the learning rate for gradient descent
        * `beta1` - the beta1 parameter for gradient descent
        * `beta2` - the beta2 parameter for gradient descent
        * if `iterations` is not an integer, raise a `TypeError` with the message `iterations must be an integer`
        * if `iterations` is not positive, raise a `ValueError` with the message `iterations must be positive`
        * if `step` is not `None` and not an integer, raise a `TypeError` with the message `step must be an integer`
        * if `step` is not `None` and not positive or less than `iterations`, raise a `ValueError` with the message `iterations must be positive and less than iterations`
        * if `lr` is not a float or an integer, raise a `TypeError` with the message `lr must be a number`
        * if `lr` is not positive, raise a `ValueError` with the message `lr must be positive`
        * if `beta1` is not a float, raise a `TypeError` with the message `beta1 must be a float`
        * if `beta1` is not in the range `[0, 1]`, raise a `ValueError` with the message `beta1 must be in the range [0, 1]`
        * if `beta2` is not a float, raise a `TypeError` with the message `beta2 must be a float`
        * if `beta2` is not in the range `[0, 1]`, raise a `ValueError` with the message `beta2 must be in the range [0, 1]`
        * gradient descent should be performed using Adam optimization
        * the generated image should be initialized as the content image
        * keep track of the best cost and the image associated with that cost
        * Returns: `generated_image, cost`
            * `generated_image` is the best generated image
            * `cost` is the best cost

## Results

| Filename |
| ------ |
| [0-neural_style.py]()|
| [1-neural_style.py]()|
| [2-neural_style.py]()|
| [3-neural_style.py]()|
| [4-neural_style.py]()|
| [5-neural_style.py]()|
| [6-neural_style.py]()|
| [7-neural_style.py]()|
| [8-neural_style.py]()|
| [9-neural_style.py]()|
| [10-neural_style.py]()|
