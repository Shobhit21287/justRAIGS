import json
import tempfile
from pathlib import Path
from pprint import pprint
import os
import SimpleITK as sitk
from PIL import Image
import tifffile
import random
import torch
from torchvision import transforms
import torchvision
from torchvision import models
import numpy
from PIL import Image
import torch.nn as nn

test_transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
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
# pth = "/Users/chaitanyadua/Downloads/justRAIGS-main/model.pth"
pth = "/opt/algorithm/model.pth"
model.load_state_dict(torch.load(pth,map_location=torch.device('cpu')))
#model.load_state_dict(torch.load("/Users/chaitanyadua/Downloads/justRAIGS-main/model.pth",map_location=torch.device('cpu'))) 
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
#model2.load_state_dict(torch.load("/Users/chaitanyadua/Downloads/justRAIGS-main/model_one_hot.pth",map_location=torch.device('cpu'))) 
model2.load_state_dict(torch.load("/opt/algorithm/model_one_hot.pth",map_location=torch.device('cpu')))
model2.eval()

DEFAULT_GLAUCOMATOUS_FEATURES = {
    "appearance neuroretinal rim superiorly": None,
    "appearance neuroretinal rim inferiorly": None,
    "retinal nerve fiber layer defect superiorly": None,
    "retinal nerve fiber layer defect inferiorly": None,
    "baring of the circumlinear vessel superiorly": None,
    "baring of the circumlinear vessel inferiorly": None,
    "nasalization of the vessel trunk": None,
    "disc hemorrhages": None,
    "laminar dots": None,
    "large cup": None,
}
is_referable_glaucoma_stacked = []
is_referable_glaucoma_likelihood_stacked = []
glaucomatous_features_stacked = []
def inference_tasks():
    #test = "C:/Users/Shobhit/Desktop/IIITacad/sem6/ML_project/sample/Example algorithm/test"
    #test = "/Users/chaitanyadua/Downloads/justRAIGS-main/Example algorithm/test"
    #input_files = [x for x in Path(test + "/input").rglob("*") if x.is_file()]
    input_files = [x for x in Path("/input").rglob("*") if x.is_file()]
    print("Input Files:")
    pprint(input_files)
    def save_prediction(
            is_referable_glaucoma,
            likelihood_referable_glaucoma,
            glaucomatous_features=None,
    ):
        is_referable_glaucoma_stacked.append(is_referable_glaucoma)
        is_referable_glaucoma_likelihood_stacked.append(likelihood_referable_glaucoma)
        if glaucomatous_features is not None:
            glaucomatous_features_stacked.append({**DEFAULT_GLAUCOMATOUS_FEATURES, **glaucomatous_features})
        else:
            glaucomatous_features_stacked.append(DEFAULT_GLAUCOMATOUS_FEATURES)

    for file_path in input_files:
        if file_path.suffix == ".mha":  # A single image
            single_file_inference(image_file=file_path)
            write_referable_glaucoma_decision(is_referable_glaucoma_stacked)
            write_referable_glaucoma_decision_likelihood(
            is_referable_glaucoma_likelihood_stacked
            )
            write_glaucomatous_features(glaucomatous_features_stacked)
        elif file_path.suffix == ".tif" or file_path.suffix == ".tiff":  # A stack of images
            stack_inference(stack=file_path)
            write_referable_glaucoma_decision(is_referable_glaucoma_stacked)
            write_referable_glaucoma_decision_likelihood(
            is_referable_glaucoma_likelihood_stacked
            )
            write_glaucomatous_features(glaucomatous_features_stacked)


def single_file_inference(image_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        image = sitk.ReadImage(image_file)
        if image.GetDimension()==3:
            image = sitk.Resample(image, size=(80, 80, image.GetDepth()), interpolator=sitk.sitkLinear)
            slice = image[:, :, image.GetDepth() // 2]
            slice = sitk.Cast(slice, sitk.sitkUInt8)
            # Save the middle slice back to an image
            output_path = Path(temp_dir) / "image.jpg"
            sitk.WriteImage(slice, str(output_path))
        # Define the output file path
        else:
            image = sitk.Resample(image, size=(80, 80), interpolator=sitk.sitkLinear)
            output_path = Path(temp_dir) / "image.jpg"

            # Save the 2D slice as a JPG file
            sitk.WriteImage(image, str(output_path))
        jpg_image_file_name=output_path
        image = Image.open(jpg_image_file_name)
        image_array=numpy.array(image)
        if(image_array.ndim==2):
            # pil_image = Image.fromarray(image_array)
            # image=test_transform_2(pil_image)
            rgb_image = numpy.zeros((3,image_array.shape[0], image_array.shape[1]))
            rgb_image[0, :, :] = image_array
            rgb_image[1, :, :] = image_array
            rgb_image[2, :, :] = image_array
            pil_image = Image.fromarray(rgb_image,mode='RGB')
            image = test_transform(pil_image)
        else:
            image = test_transform(image)
        image = image.unsqueeze(0)
        # print(image.shape)
        # image = image.unsqueeze(0)
        is_referable_glaucoma_likelihood = model(image)
        is_referable_glaucoma_likelihood = torch.sigmoid(is_referable_glaucoma_likelihood).detach().numpy()
        is_referable_glaucoma = bool(is_referable_glaucoma_likelihood[0][0] < is_referable_glaucoma_likelihood[0][1])
        is_referable_glaucoma_likelihood = float(is_referable_glaucoma_likelihood[0][1])
        features2 = {}
        features = model2(image)
        probabilities=torch.sigmoid(features)
        predicted_labels = torch.round(probabilities)
        features = predicted_labels.detach().numpy()[0].astype(int).astype(bool)
        #print(probabilities,features)
        ct = 0
        for i in DEFAULT_GLAUCOMATOUS_FEATURES.keys():
            features2[i] = features[ct].item()
            ct += 1
        is_referable_glaucoma_stacked.append(is_referable_glaucoma)
        is_referable_glaucoma_likelihood_stacked.append(is_referable_glaucoma_likelihood)
        if features2 is not None:
            glaucomatous_features_stacked.append({**DEFAULT_GLAUCOMATOUS_FEATURES, **features2})
        else:
            glaucomatous_features_stacked.append(DEFAULT_GLAUCOMATOUS_FEATURES)
        # yield output_path,callback
        del image


def stack_inference(stack):
    # de_stacked_images = []
    # Unpack the stack
    with tempfile.TemporaryDirectory() as temp_dir:
        # with Image.open(stack) as tiff_image:
        with tifffile.TiffFile(stack) as tiff_image:
            # mode = 'RGB'  # or any other mode you want to use
            # size = tiff_image.size  # Use the size of the original image
            # tiff_image = Image.new(mode, size)

            # Iterate through all pages
            # for page_num in range(tiff_image.n_frames):
            for page_num in range(len(tiff_image.pages)):
                # Select the current page
                # tiff_image.seek(page_num)
                tiff_page=tiff_image.pages[page_num]
                image=Image.fromarray(tiff_page.asarray())
                if max(image.size) >200:
                # Resize the image to reduce its dimensions
                    image = image.resize((200, 200))
                # Define the output file path
                print(f"image_{page_num+1}")
                output_path = Path(temp_dir) / f"image_{page_num + 1}.jpg"
                image.save(output_path, format="JPEG")

                # de_stacked_images.append(output_path)

                print(f"De-Stacked {output_path}")
                jpg_image_file_name=output_path
                image = Image.open(jpg_image_file_name)
                image_array=numpy.array(image)
                if(image_array.ndim==2):
                    # pil_image = Image.fromarray(image_array)
                    # image=test_transform_2(pil_image)
                    rgb_image = numpy.zeros((3,image_array.shape[0], image_array.shape[1]))
                    rgb_image[0, :, :] = image_array
                    rgb_image[1, :, :] = image_array
                    rgb_image[2, :, :] = image_array
                    pil_image = Image.fromarray(rgb_image,mode='RGB')
                    image = test_transform(pil_image)
                else:
                    image = test_transform(image)
                image = image.unsqueeze(0)
                # print(image.shape)
                # image = image.unsqueeze(0)
                is_referable_glaucoma_likelihood = model(image)
                is_referable_glaucoma_likelihood = torch.sigmoid(is_referable_glaucoma_likelihood).detach().numpy()
                is_referable_glaucoma = bool(is_referable_glaucoma_likelihood[0][0] < is_referable_glaucoma_likelihood[0][1])
                is_referable_glaucoma_likelihood = float(is_referable_glaucoma_likelihood[0][1])
                features2 = {}
                features = model2(image)
                probabilities=torch.sigmoid(features)
                predicted_labels = torch.round(probabilities)
                features = predicted_labels.detach().numpy()[0].astype(int).astype(bool)
                #print(probabilities,features)
                ct = 0
                for i in DEFAULT_GLAUCOMATOUS_FEATURES.keys():
                    features2[i] = features[ct].item()
                    ct += 1

                is_referable_glaucoma_stacked.append(is_referable_glaucoma)
                is_referable_glaucoma_likelihood_stacked.append(is_referable_glaucoma_likelihood)
                if features2 is not None:
                    glaucomatous_features_stacked.append({**DEFAULT_GLAUCOMATOUS_FEATURES, **features2})
                else:
                    glaucomatous_features_stacked.append(DEFAULT_GLAUCOMATOUS_FEATURES)
                # yield output_path,callback


# current_dir = os.getcwd()

#test = "C:/Users/Shobhit/Desktop/IIITacad/sem6/ML_project/sample/Example algorithm/test"
#test = "/Users/chaitanyadua/Downloads/justRAIGS-main/Example algorithm/test"
def write_referable_glaucoma_decision(result):
    #with open(test + "/output/multiple-referable-glaucoma-binary.json", "w+") as f:
    with open("/output/multiple-referable-glaucoma-binary.json", "w") as f:
        f.write(json.dumps(result))
    # write_json(test+"/output/multiple-referable-glaucoma-binary.json",result)


def write_referable_glaucoma_decision_likelihood(result):
    #with open(test + "/output/multiple-referable-glaucoma-likelihoods.json", "w+") as f:
    with open("/output/multiple-referable-glaucoma-likelihoods.json", "w") as f:
        f.write(json.dumps(result))


def write_glaucomatous_features(result):
    #with open(test + "/output/stacked-referable-glaucomatous-features.json", "w+") as f:
    with open("/output/stacked-referable-glaucomatous-features.json", "w") as f:
        f.write(json.dumps(result))