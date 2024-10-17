import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from torch.utils.data import DataLoader
import re

def create_context_window(text, target_string, window_size=200):

    target_index = text.find(target_string)
    #print(target_index)

    if target_index != -1:
        start_index = max(0, target_index - window_size)
        end_index = min(len(text), target_index + len(target_string) + window_size)
        context_window = text[start_index:end_index]
        return context_window

    return None

def encode_texts(texts):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    encodings = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    return encodings
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, text_encodings, candidate_encodings, labels):
        self.text_encodings = text_encodings
        self.candidate_encodings = candidate_encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx): #it works fine for training
        text_input_ids = self.text_encodings['input_ids'][idx]
        text_attention_mask = self.text_encodings['attention_mask'][idx]
        candidate_input_ids = self.candidate_encodings['input_ids'][idx]
        candidate_attention_mask = self.candidate_encodings['attention_mask'][idx]
        label = torch.tensor(self.labels[idx])

        return text_input_ids, text_attention_mask, candidate_input_ids, candidate_attention_mask, label


def prediction(text):
    data_dict={}
    data_dict[0] = {'Issue ID': 1,'Issue Body': text}
    df = pd.DataFrame.from_dict(data_dict, "index")
    df_unique = df.drop_duplicates(subset='Issue ID', keep='first')
    df = df_unique
    print(df.head())
    print(df.shape)
    excel_data = pd.read_excel('../dataset/Secret-Regular-Expression.xlsx')

    # Read the values of the file in the dataframe
    regex = pd.DataFrame(excel_data, columns=[
    'Pattern_ID','Secret Type',	'Regular Expression','Source'])

    data_dict={}
    for j in df.index:
            # if df["id"][j] != "1165939311":
            #         continue
            input_string =    str(df["Issue Body"][j])    
            input_string = re.sub(r'[\'"\│]', '', input_string)
            dir_list_clean = re.sub(r'drwx[-\s]*\d+\s+\w+\s+\w+\s+\d+\s+\w+\s+\d+\s+[0-9a-fA-F-]+.*','',input_string)
            shell_code_free_text = re.sub(r'```shell([^`]+)```','',dir_list_clean,flags=re.IGNORECASE)
            shell_code_free_text = re.sub(r'```Shell\s*"([^"]*)"\s*```','',shell_code_free_text,flags=re.IGNORECASE)
            # saved_game_free_text = re.sub(r'```([^`]+)```','',shell_code_free_text) #etay jhamela hobe
            saved_game_free_text = re.sub(r'<details><summary>Saved game</summary>\n\n```(.*?)```', '', shell_code_free_text)
            remove_packages = re.sub(r'(\w+\.)+\w+','',saved_game_free_text)
            java_exp_free_text = re.sub(r'at\s[\w.$]+\.([\w]+)\(([^:]+:\d+)\)','',remove_packages)
            # url_free_text= re.sub(https?://[^\s#]+#[A-Za-z0-9\-]+,'', java_exp_free_text, flags=re.IGNORECASE)
            url_with_fragment_text= re.sub(r'https?://[^\s#]+#[A-Za-z0-9\-\=\+]+','', java_exp_free_text, flags=re.IGNORECASE)
            url_free_text= re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',url_with_fragment_text)
            commit_free_text= re.sub(r'commit[ ]?(?:id)?[ ]?[:]?[ ]?([0-9a-f]{40})\b', '', url_free_text, flags=re.IGNORECASE)
            file_path_free_text = re.sub(r"/[\w/. :-]+",'',commit_free_text)
            file_path_free_text = re.sub( r'(/[^/\s]+)+','',file_path_free_text)
            sha256_free_text = re.sub(r'sha256\s*[:]?[=]?\s*[a-fA-F0-9]{64}','',file_path_free_text)
            sha1_free_text = re.sub(r'git-tree-sha1\s*=\s*[a-fA-F0-9]+','',sha256_free_text)
            build_id_free_text = re.sub(r'build-id\s*[:]?[=]?\s*([a-fA-F0-9]+)','',sha1_free_text)
            guids_free_text = re.sub(r'GUIDs:\s+([0-9a-fA-F-]+\s+[0-9a-fA-F-]+\s+[0-9a-fA-F-]+)','',build_id_free_text)
            uuids_free_text = re.sub(r'([0-9a-fA-F-]+\s*,\s*[0-9a-fA-F-]+\s*,\s*[0-9a-fA-F-]+)','',guids_free_text)
            event_id_free_text = re.sub(r'<([^>]+)>','',uuids_free_text)
            UUID_free_text = re.sub(r'(?:UUID|GUID|version|id)[\\=:"\'\s]*\b[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}\b'
    ,'',event_id_free_text,flags=re.IGNORECASE) ##without the prefix so many false positives can be omitted
            hex_free_text = re.sub(r'(?:data|address|id)[\\=:"\'\s]*\b0x[0-9a-fA-F]+\b','',UUID_free_text,flags=re.IGNORECASE) ## deleting hex ids directly can cause issues
            ss_free_text = re.sub(r'Screenshot_(\d{4}[_-]\d{2}[_-]\d{2}[_-]\d{2}[_-]\d{2}[_-]\d{2}[_-]\d{2}[_-]\w+)','',hex_free_text,flags=re.IGNORECASE)
            cleaned_text = ss_free_text
            # file_path = "output.txt"

            # with open(file_path, 'w') as file:
            #                 file.write(cleaned_text)
            data_dict[j] = {'Issue ID':df['Issue ID'][j],'Issue Body':cleaned_text}
            # idx = idx+1
        


    cleaned_text_data = pd.DataFrame.from_dict(data_dict, "index")
    cleaned_text_data

    idx = 0
    data_dict={}
    # start = iter*100000
    # end = (iter+1)*100000
    for i in regex.index:

        #print(i,regex['Secret Type'][i]) #, regex['Regular Expression'][i])
        # if i%100==0:
        #     print("checkpoint")
        p = re.compile(regex['Regular Expression'][i])
        
        # print("=====================================================================")
        
        for j in df.index:
            cleaned_text = cleaned_text_data.loc[j, 'Issue Body']
            # Now you can use 'cleaned_text' for further processing
            matches = re.findall(p,cleaned_text)
            for match in set(matches):
                data_dict[idx] = {'Type': regex['Secret Type'][i], 'Issue ID':df['Issue ID'][j],'Candidate String':match} #,'Entropy':shannon_entropy(match)}
                idx = idx+1
        


    data = pd.DataFrame.from_dict(data_dict, "index")
    # return if data is empty
    if len(data) == 0:
        return False,[]
    
    data=data.drop_duplicates(subset=["Issue ID", "Candidate String"], keep='first')
    print(data.shape)
    data.to_csv('crawled_issue/issues-with-candidate-strings.csv')

    data = data.rename(columns={'Issue ID': 'Issue_id'})
    print(data.shape)
    print(data.head())
    merged_df = df.merge(data, left_on='Issue ID', right_on='Issue_id')
    print(merged_df.shape)
    columns_to_remove = ['Issue_id']
    merged_df.drop(columns=columns_to_remove, inplace=True)
    print(merged_df.columns)

    merged_df['modified_text'] = merged_df.apply(lambda row: create_context_window(row['Issue Body'], row['Candidate String']), axis=1)
    print(merged_df.shape)
    print(merged_df.head())


    X_issue_ids = merged_df['Issue ID'].tolist()
    X_text_test = merged_df['Issue Body'].tolist()  # Convert the 'text' column to a list of strings
    X_candidate_test = merged_df['Candidate String'].tolist()  # Convert the 'candidate_string' column to a list of strings

    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    model_path = "../models/adamW_cntxt200_data25k_pre.pth"
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode for inference

    text_body_encodings_test = encode_texts(X_text_test)
    candidate_encodings_test = encode_texts(X_candidate_test)

    print(len(X_text_test))
    Y_labels = [0] * len(X_text_test)
    Y = np.array(Y_labels)
    Y_ =Y.astype(int)
    print(Y_)

    test_dataset = CustomDataset(text_body_encodings_test, candidate_encodings_test, Y_)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    c=0
    predicted_labels_list = []
    with torch.no_grad():

        for batch in test_loader:
            print("Batch %d"%c)
            c+=1

            text_input_ids, text_attention_mask, candidate_input_ids, candidate_attention_mask, labels = batch

            # Move tensors to the device
            text_input_ids, text_attention_mask, candidate_input_ids, candidate_attention_mask, labels = (
                text_input_ids,
                text_attention_mask,
                candidate_input_ids,
                candidate_attention_mask,
                labels.to
            )

            # Perform inference

            outputs = model(input_ids=text_input_ids.type(torch.LongTensor), attention_mask=text_attention_mask.type(torch.LongTensor))
            predicted_labels = torch.argmax(outputs.logits, dim=1)

            # print(f"predicted_labels: {predicted_labels}")
            predicted_labels_list.append(predicted_labels[0])

    predicted_labels_list_output = [f.cpu().numpy().tolist() for f in predicted_labels_list]
    print(predicted_labels_list_output)

    secret_has = False
    secrets = []
    
    for i in range(len(predicted_labels_list_output)):
        if(predicted_labels_list_output[i]==1):
            secret_has=True
            secrets.append(X_candidate_test[i])
    
    return secret_has,secrets

if __name__ == "__main__":
    print(prediction("a"))
