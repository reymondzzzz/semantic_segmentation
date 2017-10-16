import argparse
import Models , LoadBatches

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str, default = "model")
parser.add_argument("--train_images", type = str, default = "data/dataset1/images_prepped_train/")
parser.add_argument("--train_annotations", type = str, default = "data/dataset1/annotations_prepped_train/")
parser.add_argument("--n_classes", type=int, default = 10)
parser.add_argument("--input_height", type=int , default = 320)
parser.add_argument("--input_width", type=int , default = 640)

parser.add_argument("--epochs", type = int, default = 5 )
parser.add_argument("--batch_size", type = int, default = 2 )
parser.add_argument("--load_weights", type = str , default = "")

parser.add_argument("--model_name", type = str , default = "vgg_segnet")
parser.add_argument("--optimizer_name", type = str , default = "adadelta")


args = parser.parse_args()

train_images_path = args.train_images
train_segs_path = args.train_annotations
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights

optimizer_name = args.optimizer_name
model_name = args.model_name

models = {'vgg_segnet':Models.VGGSegnet.VGGSegnet, 'vgg_unet':Models.VGGUnet.VGGUnet, 'fcn32':Models.FCN32.FCN32}
model = models[model_name]

model = model(n_classes, input_height=input_height, input_width=input_width)
model.compile(loss='categorical_crossentropy',
      optimizer= optimizer_name,
      metrics=['accuracy'])


if len( load_weights ) > 0:
	model.load_weights(load_weights)


print "Model output shape", model.output_shape

output_height = model.outputHeight
output_width = model.outputWidth

G = LoadBatches.train_lists(train_images_path, train_segs_path, train_batch_size,  n_classes, input_height, input_width, output_height, output_width)


model.fit_generator(G, 512, epochs=epochs)
model.save_weights(save_weights_path + "." + str(model_name))
model.save(save_weights_path + ".model." + str(model_name))
