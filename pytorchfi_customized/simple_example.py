import torch
import torchvision.models as models
from pytorchfi.core import fault_injection as pfi_core

if __name__=="__main__":
    torch.random.manual_seed(5)
    h = 224
    w = 224
    batch_size = 1
    image = torch.rand((batch_size, 3, h, w))

    softmax = torch.nn.Softmax(dim=1)

    "Models"
    model = models.alexnet(pretrained=True)
    model.eval()

    pfi_model = pfi_core(model, h, w, batch_size)

    "Error-free inference to gather golden value"
    output = model(image)
    golden_softmax = softmax(output)
    golden_label = list(torch.argmax(golden_softmax, dim=1))[0].item()
    print("Error-free label:", golden_label)


    "Single Specified Neuron Injection"
    (b, layer, C, H, W, err_val) = (0, 3, 4, 2, 0, 2000)
    inj = pfi_model.declare_weight_fi(batch=b, conv_num=layer, c=C, h=H, w=W, value=err_val)
    inj_output = inj(image)
    inj_softmax = softmax(inj_output)
    inj_label = list(torch.argmax(inj_softmax, dim=1))[0].item()
    print("[Single Error] PytorchFI label:", inj_label)


    "Multiple Specified Neuron Injections"
    (b, layer, C, H, W, err_val) = ([0,0], [1,3], [5,4], [5,2], [3,4], [20000, 10000])
    inj = pfi_model.declare_neuron_fi(batch=b, conv_num=layer, c=C, h=H, w=W, value=err_val)

    inj_output = inj(image)
    inj_softmax = softmax(inj_output)
    inj_label = list(torch.argmax(inj_softmax, dim=1))[0].item()
    print("[Multiple Errors] PytorchFI label:", inj_label)


    "Function-based Neuron Injection"
    def mul_val_500(conv, input, output):
        if(pfi_model.get_curr_conv() == pfi_model.get_corrupt_conv()):
            prevValue = output[b][C][H][W]
            newValue = abs(prevValue) * 500
            output[b][C][H][W] = newValue
        pfi_model.updateConv()

    (b, layer, C, H, W) = (0, 0, 15, 0, 2)
    inj = pfi_model.declare_neuron_fi(function=mul_val_500, batch=b, conv_num=layer, c=C, h=H, w=W)

    inj_output = inj(image)
    inj_softmax = softmax(inj_output)
    inj_label = list(torch.argmax(inj_softmax, dim=1))[0].item()
    print("[Function] PytorchFI label:", inj_label)


