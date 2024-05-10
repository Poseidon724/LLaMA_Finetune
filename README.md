This repository contains all the files that I have worked on for the project of finetuning LLMs like LLaMA and mistral for Legal Text Recognition, Classification and Generation of Discourse Trees based on the text.

Copy_of_finetune__LLaMA3_with_LLaMA_Factory is a .ipynb file which with changing the dataset_info.json file, and adding data_02.json and data_02_test.json to the data folder of LLaMA_Factory folder and then running would train the selected base model on specified arguments/parameters, to build a new finetuned model which can be used in our applications by saving the model's adapters, weights and biases.

finetune_03.py is a python file, written for the task of finetuning LLaMA on given custom dataset, llama_data.txt. The inout output pairs are fed with a detailed prompt in this case for better results. 

finetune_sh.sh is a shell file which is to be schduled as a job when run on HPC, high performance computer. This file calls forth funetune_03.py to be run.

llama.txt is the file which contains the custom dataset, which on our case is legal text as input and discourse tree constrcuted from the text as output. These pairs are fed as input-output pairs to the model to be finetuned.

mistral_finetune.py is similar to finetune_03.py, with the only difference being the base model which is used this time being mistral instead of LLaMA.
