import random
import torch
from torchvision import transforms
import numpy
from PIL import Image
from helper import DEFAULT_GLAUCOMATOUS_FEATURES, inference_tasks


def run():
    _show_torch_cuda_info()

    for jpg_image_file_name, save_prediction in inference_tasks():
        # Do inference, possibly something better performant
        test_transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        model_classification = torch.load('model.pth')
        model_classification.eval()


        model_one_hot_encode = torch.load("model_multilabel.pth")
        model_one_hot_encode.eval()
        print(f"Running inference on {jpg_image_file_name}")

        # For example: use Pillow to read the jpg file and convert it to a NumPY array:
        image = Image.open(jpg_image_file_name)
        image = test_transform(image)

        is_referable_glaucoma_likelihood = model_classification(image)[0]
        is_referable_glaucoma_likelihood = torch.sigmoid(is_referable_glaucoma_likelihood)
        is_referable_glaucoma = is_referable_glaucoma_likelihood > 0.5
        if is_referable_glaucoma:
            features = model_one_hot_encode(image)
            dict = {}
            for i in len(features):
                dict[DEFAULT_GLAUCOMATOUS_FEATURES[i]] = features[i]
        else:
            features = None
        

        # Finally, save the answer
        save_prediction(
            is_referable_glaucoma,
            is_referable_glaucoma_likelihood,
            features,
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