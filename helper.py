import json
import tempfile
from pathlib import Path
from pprint import pprint
import os
import SimpleITK as sitk
from PIL import Image
import tifffile

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

def inference_tasks():
    #test = "C:/Users/Shobhit/Desktop/IIITacad/sem6/ML_project/sample/Example algorithm/test"
    # test = "/Users/chaitanyadua/Downloads/justRAIGS-main/Example algorithm/test"
    # input_files = [x for x in Path(test + "/input").rglob("*") if x.is_file()]
    input_files = [x for x in Path("/input").rglob("*") if x.is_file()]

    print("Input Files:")
    pprint(input_files)
    is_referable_glaucoma_stacked = []
    is_referable_glaucoma_likelihood_stacked = []
    glaucomatous_features_stacked = []
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
            yield from single_file_inference(image_file=file_path, callback=save_prediction)
        elif file_path.suffix == ".tif" or file_path.suffix == ".tiff":  # A stack of images
            yield from stack_inference(stack=file_path, callback=save_prediction)

    write_referable_glaucoma_decision(is_referable_glaucoma_stacked)
    write_referable_glaucoma_decision_likelihood(
        is_referable_glaucoma_likelihood_stacked
    )
    write_glaucomatous_features(glaucomatous_features_stacked)


def single_file_inference(image_file, callback):
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
        del image
        # Call back that saves the result
        def save_prediction(
            is_referable_glaucoma,
            likelihood_referable_glaucoma,
            glaucomatous_features=None,
        ):
            glaucomatous_features = (
                glaucomatous_features or DEFAULT_GLAUCOMATOUS_FEATURES
            )
            write_referable_glaucoma_decision([is_referable_glaucoma])
            write_referable_glaucoma_decision_likelihood(
                [likelihood_referable_glaucoma]
            )
            write_glaucomatous_features(
                [{**DEFAULT_GLAUCOMATOUS_FEATURES, **glaucomatous_features}]
            )

        yield output_path, callback


def stack_inference(stack, callback):
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
                if max(image.size) >64:
                # Resize the image to reduce its dimensions
                    image = image.resize((64, 64))
                # Define the output file path
                print(f"image_{page_num+1}")
                output_path = Path(temp_dir) / f"image_{page_num + 1}.jpg"
                image.save(output_path, format="JPEG")

                # de_stacked_images.append(output_path)

                print(f"De-Stacked {output_path}")
                yield output_path,callback

        # Loop over the images, and generate the actual tasks
        # for index, image in enumerate(de_stacked_images):
            # Call back that saves the result
            # yield image, callback

# current_dir = os.getcwd()

#test = "C:/Users/Shobhit/Desktop/IIITacad/sem6/ML_project/sample/Example algorithm/test"
# test = "/Users/chaitanyadua/Downloads/justRAIGS-main/Example algorithm/test"
def write_referable_glaucoma_decision(result):
    # with open(test + "/output/multiple-referable-glaucoma-binary.json", "w") as f:
    with open("/output/multiple-referable-glaucoma-binary.json", "w") as f:
        f.write(json.dumps(result))


def write_referable_glaucoma_decision_likelihood(result):
    # with open(test + "/output/multiple-referable-glaucoma-likelihoods.json", "w") as f:
    with open("/output/multiple-referable-glaucoma-likelihoods.json", "w") as f:
        f.write(json.dumps(result))


def write_glaucomatous_features(result):
    # with open(test + "/output/stacked-referable-glaucomatous-features.json", "w") as f:
    with open("/output/stacked-referable-glaucomatous-features.json", "w") as f:
        f.write(json.dumps(result))