
# Malaria_detection_in_bllod_smear
---
AI-Based Early Diagnosis of Malaria in Blood Smear for Resource-Limited Settings in Africa

### Abstract
Malaria is a deadly infectious disease, mainly affecting children and pregnant women in Africa, with hundreds of thousands of deaths each year. Rapid diagnosis is crucial, but traditional methods such as microscopic examination of blood slides require resources and skills technicians, which are often scarce in rural areas. The main challenge for doctors is decision-making, particularly due to the complexity and variability of symptoms. Therefore computer-aided diagnosis systems assist to prioritize highrisk cases and reduce diagnostic errors. To overcome these situations, a computer-aided diagnosis system is developed to automatically identifying trophozoite stages of P. falciparum Malaria as early identification species, white bloood cell(WBC) and negative cells. A CNN was used as the backbone network for training the artificial intelligence algorithm model architecture to classify whether a cell is infected or not. To achieve high accurancy and recall, YoloV11m and DDQ-Detr(Deformable Dynamic Query DETR) were both combined through NMS ensemble learning techniques to detect and classify trophozoite and WBC. These algorithms were integrated into an end-to-end mobile application, specifically designed to meet diagnostic needs in low-resource settings in Africa. The results showed that the proposed model is capable of detecting the trophozoite stage of Malaria with an accuracy of 0.927 and 0.99 to classify whether a cell is infected or uninfected. Our method outperforms existing approaches in terms of speed and validation on data from.

### Problem Statement
Malaria is endemic in 85 countries, with 627,000 deaths and 246 million cases in 2023.
The World Health Organization (WHO) reports that since 2000, 2.2 billion cases and 12.7 million deaths have been averted worldwide [23]. Traditionally, malaria is diagnosed by examining blood samples under a microscope for parasite-infected red blood cells. This method, however, faces several challenges, including a lack of trained parasitologists and poor facilities in malaria-endemic regions [5]. [9] One of the major obstacles to effective malaria diagnosis is human error, as doctors must process large volumes of data, often leading to inconsistencies in diagnosis. The interpretation of test results is a cognitively demanding task requiring utmost precision. Without expert supervision, microscopy and traditional diagnostic methods can result in incorrect diagnoses, leading to inappropriate treatment. This challenge is particularly significant in regions with limited resources and ### non-professional involvement [3]. Therefore, this study focuses on a systematic review of how malaria diagnosis by light microscopy can effectively improve using artificial intelligence techniques, such as Deep Learning (DL) approaches.

### Study Area, Data, and Methods
##### Data
The dataset used in the study consists of microscopic blood smear images, collected from
patients in malaria-endemic in East Africa regions. These images include both infected and healthy blood samples, focusing on the early-stage malaria parasites (trophozoites). The images in the dataset were captured by placing a smartphone over a microscope to capture the Field of View (FOV) of the blood slide through the eyepiece of the microscope. Along with the image, the slide from which the image was captured, the stage micrometer readings of the microscope, and the objective lens settings were recorded, and a maximum of 40 images was captured from each slide. The dataset is sourced from UGANDA, MAKERERE Artificial Intelligence Health Lab. ### There are 2 747 images in the train and 1178 images in the test. The images were annotated by experts, using bounding boxes to label the life stages of each parasite.
##### Methods

1. Preprocessing

As part of this work, an important step was
to transform the categorical labels into numer-
ical values. As part of the data preparation for
training the YOLO model, a key step was to
filter out negative images. We removed dupli-
cates in the Image-D column. The dataset is
splitted into two separate subsets to allow for
model training and validation. This division is
done in a stratified manner using the class col-
umn to maintain the proportions of the classes
in the two subsets, important when the classes
are unbalanced.

2. Post-processing

We optimized detection results using Non-
Maximum Suppression (NMS) to eliminate re-
dundant bounding boxes. In our ensemble
learning approach, NMS effectively combined
outputs from two detection models, retaining
only the highest-scoring boxes.