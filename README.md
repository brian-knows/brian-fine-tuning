# 🧠 Brian Fine-Tuning

## ❓ Introduction

This repository contains the code developed at the [AugmentHack](https://www.augmenthack.xyz/) hackathon in Paris, France, from the 18th of July to the 19th.

## 📚 What is this?

The Brian "Builds" feature was built at the ETHPrague hackathon, but it still had one "flaw": it was dependent on OpenAI and GPT-3.

During the AugmentHack hackathon we wanted to test and change this approach using open-source, decentralized solutions that are also more sustainable in terms of cost without losing performance.

## 📝 What did we achieve?

This is a list of what we achieved during the hackathon:

- Implemented open-source [Hugging Face](https://hugginface.co) model replacing the current GPT we use;
- Applied LoRA (Low-Rank Adaptation of Large Language Models) which allowed us to accelerate the training of large models while consuming less memory. Specifically, we used flan-t5 for the LoRA part and Falcon as the model for the prompt analysis;
- We generated and obtained more than 600 custom prompts in order to fine-tune the model, with the goal of allowing the model to better recognize the intent of a user. For example the command "Can you swap 100 USDC for ETH" can be said in many different ways such as "Can you buy 100 USDC for ETH?";
- Fine-tuned the above model with our 600+ custom prompts using [Bacalhau](https://bacalhau.org) for decentralized computing power. This allowed us to tow the model sustainably (also for the future) by not having higher costs and solving the previous vendor lock-in.

## 📂 Code structure

The repository structure is divided in 3 main folders:

- `data/`: contains the data used for the fine-tuning process (the prompts);
- `model/`: contains the various models and classes we defined for the fine-tuning process (eg. LoRA);
- `services/`: contains a simple dataloader class, used to load the prompts into HuggingFace Datasets.

## 📦 How to run

In order to run the code you need to install the dependencies:

```bash
pip install -r requirements.txt # or pip3 install -r requirements.txt
```

Then you can run the code using the following command:

```bash
python main.py # or python3 main.py
```

Since the models used in the fine-tuning are pretty big, we chose to execute our training process by using Bacalhau.

### 🐟 Bacalhau

[Bacalhau](https://bacalhau.org) is a decentralized computing platform that allows you to run your code on a decentralized network of computers.

In order to run the code on Bacalhau, you need to install the Bacalhau CLI:

```bash
curl -sL https://get.bacalhau.org/install.sh | bash
```

Then, you have two choices:

- Build the Docker image yourself using the `Dockerfile` in the root of this repository;
- Use the publicly available Docker fine-tuning image from the Brian team (available [here](https://hub.docker.com/r/brianknows/fine-tuning)).

#### Building the Docker image

In order to build the Docker image, you need to run the following command:

```bash
docker build -t brianknows/fine-tuning .
```

Once is built, you can run the image with the following command (make sure):

```bash
mkdir results
docker run -it -v $PWD/results:results brianknows/fine-tuning
```

The output of the training will be available in the `results` folder on your machine.

#### Executing on Bacalhau

In order to execute the code on Bacalhau, you need to run the following command:

```bash
bacalhau docker run brianknows/fine-tuning
```

To check the status of the job, you can run the following command:

```bash
bacalhau list
```

And if you want to get more information about the job, or if you want to download it, run the following command:

```bash
bacalhau describe <job_id> # describes the job info
bacalhau get <job_id> # downloads the job
```
