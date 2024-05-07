import torch
from torchvision import models

class FeatureReconstructionLoss:
    def __init__(self) -> None:
        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        if torch.cuda.is_available(): self.model.cuda()


    def get_intermediate_output(self, input_tensor):
        intermediate_output = None
        def hook(module, input, output):
            nonlocal intermediate_output
            intermediate_output = output


        layer_index = 34 # for the block5_conv4 convolution layer
        hook_handle = self.model.features[layer_index].register_forward_hook(hook)
        # print(hook_handle)
        self.model(input_tensor)
        hook_handle.remove()
        return intermediate_output
    

    def reconstruction_loss(self, im1, im2):
        original_features = self.get_intermediate_output(im1)
        generated_features = self.get_intermediate_output(im2)
        return torch.mean(torch.square(original_features-generated_features))



if __name__ == '__main__':
    vgg19 = FeatureReconstructionLoss()
    input_tensor1 = torch.rand(1, 3, 224, 224)
    input_tensor2 = torch.rand(1, 3, 224, 224)

    output_ = vgg19.reconstruction_loss(input_tensor1, input_tensor2)
    print(output_)