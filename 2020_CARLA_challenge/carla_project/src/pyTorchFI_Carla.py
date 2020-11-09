from pytorchfi.core import fault_injection as pfi_core
import random
import logging
import torch

class pyTorchFI_Carla_Utils(object):
    def __init__(self):
        pass

    @staticmethod
    def is_conv_weight_layer(name):
        # Note layers 'network.classifier.4.weight' and 'network.classifier.1.weight' are conv layers too
        if "weight" not in name:
            return False
        elif "conv" in name:
            return True
        elif "network.classifier.1.weight" in name:
            return True
        elif "network.classifier.4.weight" in name:
            return True

        return False

    @staticmethod
    def random_weight_location(pfi_model, conv=-1):
        loc = list()

        if conv == -1:
            corrupt_layer = random.randint(0, pfi_model.get_total_conv() - 1)
        else:
            corrupt_layer = conv
        loc.append(corrupt_layer)

        curr_layer = 0
        for name, param in pfi_model.get_original_model().named_parameters():
            if pyTorchFI_Carla_Utils.is_conv_weight_layer(name):
                if curr_layer == corrupt_layer:
                    for dim in param.size():
                        loc.append(random.randint(0, dim - 1))
                curr_layer += 1

        assert curr_layer == pfi_model.get_total_conv()
        assert len(loc) == 5

        return tuple(loc)

    @staticmethod
    def print_pfi_model(pfi_model):
        count = 0
        
        for name,param in pfi_model.get_original_model().named_parameters():
            if pyTorchFI_Carla_Utils.is_conv_weight_layer(name):
                print(name + ' <== Conv')
                count+=1
            else:
                print(name)

        print("Total Conv:",pfi_model.get_total_conv())
        print("Total elements counted:", count)