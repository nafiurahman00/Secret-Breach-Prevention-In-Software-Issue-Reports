import json
import csv
import pandas as pd

#read report.json file and return the json object
def read_report_json():
    with open('report.json') as json_file:
        data = json.load(json_file)
        return data
    
data = read_report_json()

# Save to CSV
with open('output.csv', 'w', newline='') as csvfile:
    fieldnames = ['Issue ID','Issue Body']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    cnt=0
    writer.writeheader()
    for item in data:
        if item['body'] is not None:
            writer.writerow({'Issue ID':item['id'],'Issue Body': item['body']})
            print(cnt)
            cnt+=1
    
#sort the csv file by Issue ID
df = pd.read_csv('output.csv')
df.sort_values(by=['Issue ID'], inplace=True)
df.to_csv('output.csv', index=False)


        


