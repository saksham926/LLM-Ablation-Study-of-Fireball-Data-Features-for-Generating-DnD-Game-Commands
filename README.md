### CMSC 691 - Team 5 Arya, Patty and Saksham
### Project LLM Ablation Study of Fireball Data Fetures for Generating DnD Game Commands
#### Project Code Site 
https://drive.google.com/drive/folders/1sy1K0Xc5yqQwbMLECIOzEf2OBqd5hGJL?usp=drive_link

Recommended environment:
Linux with NVIDIA GPU with minimum of 16gb GPU memory.  Batch size should be set at 1 for GPU and 16 for higher end GPUs with additional memory.  An AI Workstation with two A6000 with 48GB of GPU Memory was used to train each of the models which took about 5 to 7 hours each. 

#### Model Training Notes

To reduce compute for Llama-3.1-8b-Instruct model, we leveraged the NVIDIA NeMo framework which provided a streamlined path to use PEFT Lora.  We specifically adapted this NeMo Lora tutorial https://github.com/NVIDIA/NeMo/blob/main/tutorials/llm/llama-3/sdg-law-title-generation/llama3-sdg-lora-nemofw.ipynb for this project. 

### Install Instructions for Development Environment 
#### make project directory and add project files
~mkdir nvdata
cd nvdata 

Because of the size of the data, we have placed them on a Google Drive. You can access the data files at https://drive.google.com/drive/folders/1lwn1PD5JLFbKGncgUu_IfGHIWa-O9ZPd?usp=drive_link.   We recommend you download as a zip and upload into the  data folder. 


Because of the size of the data and model files, we have placed the project at https://drive.google.com/drive/folders/17s6qX-p0Js-V97sjHus2YpNrl4CPUMRx?usp=sharing.  We recommend you download as a zip and upload into the nvdata folder. Contains Llama3.1 model and limited GPT-2 model trained on ScienceWorld tasks. Also includes supporting infrastructure, utilitiy scripts, and logs for runs of these models.

The trained lora weights for each of the five fine-tuned variations are available at https://drive.google.com/drive/folders/1sy1K0Xc5yqQwbMLECIOzEf2OBqd5hGJL?usp=sharing. 

The source Llama model file can be downloaded from https://drive.google.com/file/d/1kbmJ3BlDMHjvpePMJ-mN_m5Lsn_up5wk/view?usp=drive_link.  It is in the NeMo format and is 15 GB. 

#### NeMo Docker Container
We recommend that you use the NeMo docker container as it contains the transformers, NeMo, pytorch and bulk of the libraries needed for this project.  The NeMo container is freely available at https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo.  You will just need to create a free developer account.  We used the latest container nvcr.io/nvidia/nemo:24.07.

This command will pull and run the docker container and mount the nvdata project directory

docker run --gpus all --runtime=nvidia -it --rm -v --shm-size=16g -p 8888:8888 -p 6006:6006 \
 -v ~/nvdata:/workspace/nvdata \
--ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/nemo:24.07

Once you are the #sign inside the container, run the following commands
pip install ipywidgets
pip install jupyter_contrib_nbextensions

jupyter nbextension enable --py widgetsnbextension
pip install -U "huggingface_hub[cli]"


#### Jupyter-lab
Finally, run the following command to spin up a Jupyter-Lab session.
jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='' --no-browser

Navigate to a browser http://<ip or localhost>:8888/lab to view and run the code.

The data files should be placed in a data folder  

#### Notebook files these should be placed at the root of your project 
The following notebooks are the ones available to train and evaluate the models:  
- 691_DnD_Project_LlamaTraining_fb_m1.ipynp  tuned on before_utterances and current_actor.
- 691_DnD_Project_LlamaTraining_fb_m2.ipynp  tuned on before_utterances, combat_state_before, and current_actor
- 691_DnD_Project_LlamaTraining_fb_m3.ipynp  tuned on combat_state_before and current_actor
- 691_DnD_Project_LlamaTraining_fb_m4.ipynp  tuned on current_actor
- 691_DnD_Project_LlamaTraining_fb_m5.ipynp  tuned on before_utterances
- 1_691_DnD_Project_DataPrep_final.ipynb  this contains the logic to pull in the fireball dataset from HuggingFace and transform into the formats needed for training. You can directly download the transformed datasets from https://drive.google.com/drive/folders/1lwn1PD5JLFbKGncgUu_IfGHIWa-O9ZPd?usp=drive_link
