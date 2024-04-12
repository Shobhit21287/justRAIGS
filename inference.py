import random
import torch
from torchvision import transforms
import torchvision
from torchvision import models
import numpy
from PIL import Image
from helper import DEFAULT_GLAUCOMATOUS_FEATURES, inference_tasks
import torch.nn as nn


def run():
    _show_torch_cuda_info()

    test_transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


#model=models.resnet18(pretrained=True)
    model=models.densenet201(pretrained=False)
    for parameter in model.parameters():
        parameter.requires_grad=False
    model.classifier = nn.Identity()
    custom_layer = nn.Sequential(
        nn.Linear(1920,1920),
        nn.Dropout(0.25),
        nn.ReLU(),
        nn.Linear(1920,512),
        nn.Dropout(0.25),
        nn.ReLU(),
        nn.Linear(512,256),
        nn.Dropout(0.25),
        nn.ReLU(),
        nn.Linear(256,256),
        nn.ReLU(),
        nn.Linear(256,128),
        nn.ReLU(),
        nn.Linear(128,2)
    )
    model = nn.Sequential(model, custom_layer)
    #pth = "C:\\Users\\Shobhit\\Desktop\\IIITacad\\Sem6\\ML_project\\sample\\model.pth"
    pth = "/opt/algorithm/model.pth"
    model.load_state_dict(torch.load(pth,map_location=torch.device('cpu')))
    model.eval()

    model2=models.resnet18(pretrained=False)
    for parameter in model2.parameters():
        parameter.requires_grad=True
    features_last_fc = model2.fc.in_features
    model2.fc=torch.nn.Sequential(
        nn.Linear(features_last_fc,features_last_fc),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(features_last_fc,features_last_fc),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(features_last_fc,256),
        nn.ReLU(),
        nn.Linear(256,128),
        nn.ReLU(),
        nn.Linear(128,10),
    )
    #model2.load_state_dict(torch.load("C:\\Users\\Shobhit\\Desktop\\IIITacad\\Sem6\\ML_project\\sample\\model_one_hot.pth")) 
    model2.load_state_dict(torch.load("/opt/algorithm/model_one_hot.pth",map_location=torch.device('cpu')))
    model2.eval()
    

    # model_one_hot_encode = torch.load("model_multilabel.pth")
    # model_one_hot_encode.eval()

    for jpg_image_file_name, save_prediction in inference_tasks():
        # Do inference, possibly something better performant
        print(f"Running inference on {jpg_image_file_name}")
        # For example: use Pillow to read the jpg file and convert it to a NumPY array:
        image = Image.open(jpg_image_file_name)
        image = test_transform(image)
        image = image.unsqueeze(0)
        is_referable_glaucoma_likelihood = model(image)
        is_referable_glaucoma_likelihood = torch.sigmoid(is_referable_glaucoma_likelihood).detach().numpy()
        is_referable_glaucoma = bool(is_referable_glaucoma_likelihood[0][0] < is_referable_glaucoma_likelihood[0][1])
        is_referable_glaucoma_likelihood = float(is_referable_glaucoma_likelihood[0][1])
        features2 = {}
        if is_referable_glaucoma:
            # features = {
            #     k: random.choice([True, False])
            #     for k, v in DEFAULT_GLAUCOMATOUS_FEATURES.items()
            # }
            features = model2(image)
            probabilities=torch.sigmoid(features)
            predicted_labels = torch.round(probabilities)
            features = predicted_labels.detach().numpy()[0].astype(int).astype(bool)
            print(probabilities,features)
            ct = 0
            for i in DEFAULT_GLAUCOMATOUS_FEATURES.keys():
                features2[i] = features[ct].item()
                ct += 1
        else:
            ct = 0
            for i in DEFAULT_GLAUCOMATOUS_FEATURES.keys():
                features2[i] = False
                ct += 1
        
        # for i in DEFAULT_GLAUCOMATOUS_FEATURES.keys():
        #     features2[i] = False
        # features = model2(image)
        # probabilities=torch.sigmoid(features)
        # predicted_labels = torch.round(probabilities)
        # features = predicted_labels.detach().numpy()[0].astype(int).astype(bool)
        # ct = 0
        # for i in DEFAULT_GLAUCOMATOUS_FEATURES.keys():
        #     features2[i] = features[ct].item()
        #     ct += 1  

        # Finally, save the answer
        save_prediction(
            is_referable_glaucoma,
            is_referable_glaucoma_likelihood,
            features2,
        )
    return 0


def _show_torch_cuda_info():
    print("=+=" * 10)
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())



#git add .
#git commit -m ""
#git push origin main