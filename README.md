# MBGen
Official implementation of MBGen

Download datasets from [here](https://drive.google.com/drive/folders/1G7tvIT1wvGZC2GmI-8Okbn9HrGQnNbfu?usp=sharing) and put the datasets into `./data/raw_dataset/`

Generated item IDs are located in `./tokenizer/ID/`. To generate a new set of item IDs, encode the prior of items into embeddings (through pre-trained embedding models/embedding table of pre-trained sequential recommendation models) and save the generated embedding as a tensor of size `num_items * embedding_dim` into `./tokenizer/embedding.pkl`.

To train the model with Retail dataset
```
python run.py --config=./config/main/retail/main.yaml
```

To train the model with IJCAI dataset
```
python run.py --config=./config/main/ijcai/main.yaml --dataset=ijcai
```