import torch
from train import Net


class Convert(object):
    def __init__(self, model_path, gpu=True):
        self.net = Net(gpu)
        self.net.load(model_path)

    def convert_to_onnx(self):

        # Set the model to evaluation mode
        self.net.eval()

        # Export the model to ONNX
        torch.onnx.export(self.net,                    # PyTorch Model
                          torch.rand(1, 3, 39, 135),   # Input tensor
                          "./model.onnx",              # ONNX file path
                          verbose=False,
                          # Input tensor name
                          input_names=['input'],
                          # Output tensor name
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}},
                          export_params=True)


if __name__ == '__main__':
    man = Convert('pretrained')
    man.convert_to_onnx()
