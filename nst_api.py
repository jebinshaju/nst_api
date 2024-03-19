
from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import numpy as np
import PIL.Image
import os

app = Flask(__name__)

# Load pre-trained model
def load_model():
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagnet')
    vgg.trainable = False

    style_extractor = vgg_layers(style_layers)
    extractor = StyleContentModel(style_layers, content_layers)

    return extractor

def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagnet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)

def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def style_transfer(content_image_path, style_image_path, output_path, epochs, steps_per_epoch):
    content_image = load_img(content_image_path)
    style_image = load_img(style_image_path)

    extractor = load_model()
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    image = tf.Variable(content_image)

    opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    style_weight = 1e-2
    content_weight = 1e4

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs)

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    def style_content_loss(outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2) for name in style_outputs.keys()])
        style_loss *= style_weight / len(style_outputs)

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2) for name in content_outputs.keys()])
        content_loss *= content_weight / len(content_outputs)

        loss = style_loss + content_loss
        return loss

    def clip_0_1(image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            train_step(image)

    # Convert the tensor to a NumPy array and remove the batch dimension
    generated_image_array = np.array(image[0].numpy() * 255, dtype=np.uint8)

    # Save the generated image
    generated_image_pil = PIL.Image.fromarray(generated_image_array)
    generated_image_pil.save(output_path)

@app.route('/transfer_style', methods=['POST'])
def transfer_style():
    content_file = request.files['content']
    style_file = request.files['style']

    content_filename = 'contentpic.jpg'
    style_filename = 'stylepic.jpg'
    output_filename = 'generated_image1.jpg'

    content_path = os.path.join('content', content_filename)
    style_path = os.path.join('style', style_filename)
    output_path = os.path.join('generated', output_filename)

    content_file.save(content_path)
    style_file.save(style_path)

    epochs = int(request.form.get('epochs', 1))
    steps_per_epoch = int(request.form.get('steps_per_epoch', 5))

    style_transfer(content_path, style_path, output_path, epochs, steps_per_epoch)

    return jsonify({'result': 'success', 'generated_image': output_filename})

@app.route('/generated_image/<path:image_name>')
def get_generated_image(image_name):
    generated_image_path = os.path.join('generated', image_name)
    if os.path.exists(generated_image_path):
        return send_file(generated_image_path, mimetype='image/jpeg')
    else:
        return jsonify({'message': 'Image not yet generated. Please wait for the process to complete.'})

if __name__ == '__main__':
    app.run(debug=True)