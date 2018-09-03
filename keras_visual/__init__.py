import numpy as np
import keras.applications as applications
from PIL import Image

IMAGE_PATH = './husky.jpg'

# we expect a thousand classes if we are using the same pretrained model
# if we wanted to vary the classes, we would retrain the last few dense layers
model = applications.vgg16.VGG16(include_top=True, weights='imagenet')

for layer in model.layers:
    # this tells all the layer names, and which layers we want to visualise
    # also use model.summary()
    print(layer.name)

# use Pillow to resize the images

img = Image.open('./husky.jpg')
img = img.resize((224, 224))

# img_array is currently Height, Width, Channels
img_array = np.asarray(img)
# this becomes channels, height, width
img_array = np.transpose(img_array, (2, 0, 1))
# turn this into a batch N, channels, height, width
img_array = img_array[np.newaxis, :]
# convert to floating point
img_array = img_array.astype(np.float)
# there appears to be some VGG16 specific preprocessing required
# like mean subtraction to center the variations, also possibly normalisation to stddev (but usually not required due to images being of limited channel depth anyway) (subtract mean and divide by stddev)
img_array = applications.imagenet_utils.preprocess_input(img_array)

deconv = visualise(model, img_array, 'block1_conv2', feature_to_vis, 'max')

# we have to postprocess the image again
# change back to height, width, channels
deconv = np.transpose(deconv, (1, 2, 0))
# minus the minimum value
deconv = deconv - deconv.min()
# i don't know (I think this is intended to scale the image between 0 and 1.0)
# that way we get the full range of an 8 bit depth!
deconv *= 1.0 / (deconv.max() + 1e-8)
# convert to image (0 - 255) for a 8 bit depth
uint8_deconv = (deconv * 255).astype(np.uint8)
img = Image.fromArray(uint8_deconv, 'RGB')

# use matplotlib to show image!


# calling this will attach the deconv stack to each layer
# while running the inference on the data itself
# oh this returns 1 layer name

# i really want this for resnet, it will be really useful

# visualise mode is all, max
# max is the greatest activation
# all will use all values
# for conv layers this is important
def visualise(model, data, layer_name, feature_to_vis, visualise_mode):
    deconv_layers = []
    # for each layer in the model
    for i in range(len(model.layers)):
        if isinstance(model.layers[i], Convolution2D):
            deconv_layers.append(DConvolution2D(model.layers[i]))
            deconv_layers.append(DActivation(model.layers[i]))
        elif isinstance(model.layers[i], MaxPooling2D):
            deconv_layers.append(DPooling(model.layers[i]))
        elif isinstance(model.layers[i], Dense):
            deconv_layers.append(DDense(model.layers[i]))
            deconv_layers.append(DActivation(model.layers[i]))
        elif isinstance(model.layers[i], Flatten):
            deconv_layers.append(DFlatten(model.layers[i]))
        elif isinstance(model.layers[i], InputLayer):
            deconv_layers.append(DInput(model.layers[i]))
        # so we add in the stack according to the position in the original model
        # we don't bother adding more
        # the layer stack has to be according to the layers that we need
        # better way would be to use the functional API to create a graph for visualisation
        if layer_name == model.layers[i].name:
            break

    # how does this work?
    # the UP motion refers to going UP the network
    # because we SAVE the normal layer and the deconvolutional version
    deconv_layers[0].up(data)
    for i in range(1, len(deconv_layers)):
        deconv_layers[i].up(deconv_layers[i - 1].up_data)

    # it seems that you call the up function
    # on each previous layers (output)
    output = deconv_layers[-1].up_data

    # i don't know what feature_to_vis is
    # this means you are indexing the output for some reason
    # is this a filter index? the index of the filters themselves?

    # the ndim being 2 is only possible in the case of the DENSE layer being visualised (because 1x4096)
    if output.ndim == 2:
        feature_map = output[:, feature_to_vis]
    else:
        feature_map = output[:, feature_to_vis, :, :]

    if visualise_mode == 'max':
        # use MAX thresholding
        # this means get the boolean mask of all the locations where it is MAX
        # but this means that we are getting some sort of switches?
        max_activation = feature_map.max()
        temp = feature_map == max_activation
        feature_map = feature_map * temp

    # we get the same shape/dimensions, but all the array is set to 0
    output = np.zeros_like(output)

    if output.ndim == 2:
        # now we set it back according to the feature map
        output[:, feature_to_vis] = feature_map
    else:
        # same here
        output[:, feature_to_vis, :, :] = feature_map

    # now that we have the UP done, we need to do the DOWN path, which is the actual DECONV

    deconv_layers[-1].down(output)
    for i in range(len(deconv_layers) - 2, -1, -1):
        deconv_layers[i].down(deconv_layers[i + 1].down_data)
    deconv = deconv_layers[0].down_data
    # removes all unit dimensions! (removes all unit dimensions)
    deconv = deconv.squeeze()

    # this is apparently not an "actual" deconvolution
    # it's more of a transposed convolution
    # because it does fully invert the convolutional process
    # you can then use it to encode/decode structures

    return deconv





class DConvolution2D():
    pass

class DActivation():
    pass


