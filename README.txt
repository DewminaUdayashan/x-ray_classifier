1. Spinal X-rays (Primary Dataset)

Dataset Name: AOSpine Knowledge Forum Spinal Deformity

Purpose in our Project: The core dataset containing full spinal X-rays of patients with and without scoliosis. This will be used for all three models.

Download Link: The Cancer Imaging Archive (TCIA) - AOSpine Collection

Instructions & Notes:

On the TCIA page, you will find instructions for accessing the data. You will need to use their NBIA Data Retriever software to download the image sets.

Access requires agreeing to the TCIA data usage policy. This is a standard procedure to ensure the data is used ethically for research purposes.

Download the relevant image series and place them into the data/raw/xray_spine_aos/ directory.

2. Chest X-rays

Dataset Name: NIH ChestX-ray14

Purpose in our Project: To provide a large set of X-rays that are not of the spine. This is essential for training Model 2 (is_full_spine) to learn the difference.

Download Link: NIH ChestX-ray Official Download Page

Instructions & Notes:

This dataset is very large (over 40 GB). The link above leads to a Box folder where the images are split into multiple compressed tar.gz files (e.g., images_01.tar.gz, images_02.tar.gz, etc.).

You will need to download all of these files.

Once downloaded, extract the images from all the archives and place them together into the data/raw/xray_chest/ directory.

3. Musculoskeletal X-rays

Dataset Name: MURA (musculoskeletal radiographs)

Purpose in our Project: Similar to the chest X-rays, this provides more variety of non-spinal X-rays (arms, legs, shoulders) to make Model 2 more robust.

Download Link: Stanford ML Group - MURA Dataset

Instructions & Notes:

Click the "Download the dataset" link on the page.

You will be required to fill out a form and agree to a research-use agreement. Access is typically granted immediately upon submission.

Download the data and place the images into the data/raw/xray_mura/ directory.

4. Non-X-ray Images

Dataset Name: ImageNet (or a subset like Imagenette)

Purpose in our Project: To provide a diverse set of everyday images (animals, objects, scenes) to teach Model 1 (is_xray) what is definitively not an X-ray.

Download Method: Programmatic (Recommended)

Instructions & Notes:

Manually downloading the full ImageNet dataset (over 1TB) is not necessary or recommended. The best practice is to use a library like tensorflow_datasets to download a manageable subset.

I strongly recommend this approach. You can use a small script like the one below to download ~10,000 images directly into the correct folder. Imagenette is a popular 1.5GB subset of ImageNet that is perfect for our use case.