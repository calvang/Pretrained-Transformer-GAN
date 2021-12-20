from sys import argv
import pandas as pd
from datetime import datetime
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelWithLMHead


# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")

decoder_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

# encoder_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

base_dir = "./"
batch_size = 4

class GANDataset(Dataset):
    def __init__(self, df, col_key="text", 
                 padding="max_length", max_length=128):

        decoder_encodings = decoder_tokenizer(
            df[col_key].values.tolist(),
            padding=padding,
            max_length=max_length,
            return_tensors="pt")

        self.decoder_input_ids = decoder_encodings["input_ids"]
        self.decoder_attn_masks = decoder_encodings["attention_mask"]
    
    def __len__(self):
        return len(self.decoder_input_ids)

    def __getitem__(self, idx):
        return self.decoder_input_ids[idx], self.decoder_attn_masks[idx]
    

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        # self.model = GPT2LMHeadModel.from_pretrained("../gpt2", local_files_only=True).to(device)
        self.model = AutoModelWithLMHead.from_pretrained("distilgpt2").to(device)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.config.max_length = 129
        self.with_grad = True

    def disable_grad(self):
        self.with_grad = False
        self.model.eval()
    
    def enable_grad(self):
        self.with_grad = False
        self.model.train()

    def forward(self, x, attn_masks):
        out = self.model(
            input_ids=x,
            labels=x,
            attention_mask=attn_masks)
        
        # gen_out = self.model.generate(
        #     do_sample=True,
        #     top_k=50,
        #     top_p=0.95, 
        #     max_length=129,
        #     num_return_sequences=batch_size)

        return out.loss
    
    def generate(self, 
                do_sample=True,
                top_k=50,
                top_p=0.95, 
                max_length=129,
                num_return_sequences=batch_size):
        out = self.model.generate(
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p, 
            max_length=max_length,
            num_return_sequences=num_return_sequences)
        return [decoder_tokenizer.decode(el[1:]) for el in out]

    
def train_gpt(gen, epochs, train_loader, val_loader, gen_lr=5e-5):
    gen_train_losses = []
    gen_val_losses = []

    gen_optimizer = torch.optim.Adam(params=gen.parameters(), lr=gen_lr)

    try:
        for epoch in range(epochs):
            running_gen_train_loss = 0.
            running_gen_val_loss = 0.

            
            # normal training
            for i, data in tqdm(enumerate(train_loader)):
                real_gen, masks = data

                real_labels = torch.ones(batch_size,2).to(device)

                gen.enable_grad()
                gen_optimizer.zero_grad()

                gen_train_loss = gen(
                    real_gen.to(device),
                    masks.to(device))
                
                gen_train_loss.backward()
                gen_optimizer.step()

                running_gen_train_loss += torch.mean(gen_train_loss).item()

            gen_train_losses.append(running_gen_train_loss)
            
            torch.save({
                "gen_state_dict": gen.state_dict(),
                "gen_optimizer_state_dict": gen_optimizer.state_dict(),
            }, base_dir + f"Checkpoints/GPT_{epoch}_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.pth")

            with torch.no_grad():
                for i, data in tqdm(enumerate(val_loader)):
                    real_gen, masks = data

                    real_labels = torch.ones(batch_size,2).to(device)

                    gen_loss = gen(
                        real_gen.to(device),
                        masks.to(device))       

                    running_gen_val_loss += torch.mean(gen_loss).item()

                gen_val_losses.append(running_gen_val_loss)

            torch.save({
                "gen_train_losses": gen_train_losses,
                "gen_val_losses": gen_val_losses,
            }, base_dir + f"Checkpoints/GPT_{epoch}_losses_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.pth")
        
    except Exception as e:
        print(e)
        return {
            "gen_state_dict": gen.state_dict(),
            "gen_optimizer_state_dict": gen_optimizer.state_dict(),
            "gen_train_losses": gen_train_losses,
            "gen_val_losses": gen_val_losses,
        }

    return {
        "gen_train_losses": gen_train_losses,
        "gen_val_losses": gen_val_losses,
    }


def test_gan(gen, test_df):
    real = test_df["text"].values.tolist()
    fake = []

    with torch.no_grad():
        # fake = gen.generate(num_return_sequences=len(real)) 
        for i in range(len(real)):
            f = gen.generate(
                    do_sample=True,
                    top_k=50,
                    top_p=0.95, 
                    max_length=129,
                    num_return_sequences=1)
            fake.append(f[0])
    
    ref = [ r.strip().split() for r in real ]
    test = [ s.strip().split() for s in fake ]
    score = corpus_bleu(ref, test)
    
    print("Corpus BLEU score:", score)
    
    results = pd.DataFrame({
        "real": real,  
        "fake": fake })
    results.to_csv(base_dir + f"Output/GPT_out_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.csv")
    return score


if __name__ == "__main__":
    if len(argv) > 1:
        if argv[1] == "train":

            train_df = pd.read_csv(base_dir + "train_20k_filtered.csv")
            train_data = GANDataset(train_df)
            train_loader = DataLoader(
                train_data, 
                sampler = RandomSampler(train_data),
                batch_size = 4)
            
            val_df = pd.read_csv(base_dir + "val_1k_filtered.csv")
            val_data = GANDataset(val_df)
            val_loader = DataLoader(
                val_data, 
                sampler = SequentialSampler(val_data), 
                batch_size = 4)
            
            gen = Generator()
            results = train_gpt(gen, 4, train_loader, val_loader)
        elif argv[1] == "test" and len(argv) > 2:
            gen = Generator()
            checkpoint = torch.load(base_dir + str(argv[2]))
            gen.load_state_dict(checkpoint["gen_state_dict"])
            
            test_df = pd.read_csv(base_dir + "test_10k_filtered.csv")
            eval_results = test_gan(gen, test_df)
        elif argv[1] == "test":
            gen = Generator()
            
            test_df = pd.read_csv(base_dir + "test_10k_filtered.csv")
            eval_results = test_gan(gen, test_df)
    