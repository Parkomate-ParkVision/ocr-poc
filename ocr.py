import torch
from torchvision.transforms import ToTensor
from PIL import Image

# Make sure to have the model architecture implemented in a separate file
from model import Model
# Make sure to have the label converter implemented in a separate file
from utils import AttnLabelConverter, CTCLabelConverter

imgH = 32
imgW = 100
character = '0123456789abcdefghijklmnopqrstuvwxyz'
Prediction = 'CTC'


class DummyOpt:
    def __init__(self, image_folder, workers, batch_size, saved_model, batch_max_length, imgH, imgW, rgb, character, sensitive, PAD, Transformation, FeatureExtraction, SequenceModeling, Prediction, num_fiducial, input_channel, output_channel, hidden_size):
        self.image_folder = image_folder
        self.workers = workers
        self.batch_size = batch_size
        self.saved_model = saved_model
        self.batch_max_length = batch_max_length
        self.imgH = imgH
        self.imgW = imgW
        self.rgb = rgb
        self.character = character
        self.sensitive = sensitive
        self.PAD = PAD
        self.Transformation = Transformation
        self.FeatureExtraction = FeatureExtraction
        self.SequenceModeling = SequenceModeling
        self.Prediction = Prediction
        self.num_fiducial = num_fiducial
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.hidden_size = hidden_size

        if 'CTC' in Prediction:
            self.num_class = len(character)
        else:
            self.num_class = 38


def load_model(model_path, device, opt):
    # Make sure to initialize the model architecture similarly as in the training script
    model = Model(opt)
    model.load_state_dict(torch.load(
        model_path, map_location=device), strict=False)
    model = model.to(device)
    model.eval()
    return model


def inference(model, image_path, converter, opt: DummyOpt):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image_tensor = ToTensor()(image).unsqueeze(0).to(device)
    text_for_pred = torch.LongTensor(
        opt.batch_size, opt.batch_max_length + 1).fill_(0).to(device)

    length_for_pred = torch.IntTensor(
        [opt.batch_max_length]).to(device)

    if 'CTC' in opt.Prediction:
        preds = model(image, text_for_pred, is_train=False)

        # Select max probabilty (greedy decoding) then decode index to character
        preds_size = torch.IntTensor([preds.size(1)])
        _, preds_index = preds.max(2)
        # preds_index = preds_index.view(-1)
        preds_str = converter.decode(preds_index, preds_size)

    else:
        preds = model(image, text_for_pred, is_train=False)

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)

    return preds_str[0]


if __name__ == '__main__':
    # Change this to the path of your trained model file
    model_path = 'models/best_accuracy.pth'
    # Change this to the path of your input image
    image_path = 'images/image.png'

    dummy_opt = DummyOpt(
        image_folder=image_path,
        workers=4,
        batch_size=192,
        saved_model=model_path,
        batch_max_length=25,
        imgH=32,
        imgW=100,
        rgb=True,  # Set to True if using rgb input, else False
        character='0123456789abcdefghijklmnopqrstuvwxyz',
        sensitive=False,  # Set to True for sensitive character mode, else False
        PAD=False,  # Set to True for PAD, else False
        Transformation='TPS',  # Choose 'None' or 'TPS'
        FeatureExtraction='VGG',  # Choose 'VGG', 'RCNN', or 'ResNet'
        SequenceModeling='BiLSTM',  # Choose 'None' or 'BiLSTM'
        Prediction='CTC',  # Choose 'CTC' or 'Attn'
        num_fiducial=20,
        input_channel=1,
        output_channel=512,
        hidden_size=256
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if 'CTC' in dummy_opt.Prediction:
        converter = CTCLabelConverter(dummy_opt.character)
    else:
        converter = AttnLabelConverter(dummy_opt.character)

    model = load_model(dummy_opt.saved_model, device, dummy_opt)
    result = inference(model, image_path, converter, dummy_opt)

    print("Predicted Text:", result)
