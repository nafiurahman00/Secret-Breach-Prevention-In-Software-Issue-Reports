import os
import requests

from flask import Flask, request
from github import Github, GithubIntegration

from secret_finding_for_tool import prediction

import pandas as pd

app = Flask(__name__)
# MAKE SURE TO CHANGE TO YOUR APP NUMBER!!!!!
app_id = 829793
# Read the bot certificate
with open(
        os.path.normpath(os.path.expanduser(r"secret-detection-tool.2024-02-15.private-key.pem")),
        'r'
) as cert_file:
    app_key = cert_file.read()

# Create a GitHub integration instance
git_integration = GithubIntegration(
    app_id,
    app_key,
)

def contains_vowels(text):
    vowels = 'aeiouAEIOU'
    return any(char in vowels for char in text)

@app.route("/", methods=['POST'])
def bot():
   
    # Get the event payload
    payload = request.json

    # Check if the event is a GitHub PR creation event
    if 'pull_request' in payload and payload['action'] == 'opened':
        owner = payload['repository']['owner']['login']
        repo_name = payload['repository']['name']

        # Get a git connection as our bot
        git_connection = Github(
            login_or_token=git_integration.get_access_token(
                git_integration.get_installation(owner, repo_name).id
            ).token
        )
        repo = git_connection.get_repo(f"{owner}/{repo_name}")

        issue = repo.get_issue(number=payload['pull_request']['number'])

        pr_description = payload['pull_request']['body']
        if not pr_description:
            issue.create_comment("The pull request description is empty.")
            return "ok"
        
        dict ={}
        dict[0] = {'pull_request': pr_description}
        data = pd.DataFrame.from_dict(dict, "index")
        data.to_csv('pullreq.csv', index=False)
            
        df = pd.read_csv('pullreq.csv')
        content = df['pull_request'][0]

        # Check if the pull request description contains vowels
        if prediction(pr_description):
            issue.create_comment("Contains secret")
        else:
            issue.create_comment("You're safe")

    # Check if the event is a GitHub issue comment event
    elif 'issue' in payload and payload['action'] == 'created':
        print("here")
        owner = payload['repository']['owner']['login']
        repo_name = payload['repository']['name']
        issue_number = payload['issue']['number']
        comment_text = payload['comment']['body']
        comment_author = payload['comment']['user']['login']

        print(comment_author)
        # Get a git connection as our bot
        git_connection = Github(
            login_or_token=git_integration.get_access_token(
                git_integration.get_installation(owner, repo_name).id
            ).token
        )
        repo = git_connection.get_repo(f"{owner}/{repo_name}")
        issue = repo.get_issue(number=issue_number)
        
        # Check if the comment is made by our bot
        if comment_author != "secret-detection-tool[bot]":  # Replace "your_bot_username" with your bot's username
            # Check if the comment contains vowels
            # 
            # with open('out.txt', 'wb') as f:
            #     f.write(comment_text.encode('utf-8'))
            # c = comment_text.encode('utf-8')
            # read the file and save to a variable content
            # content=""
            # with open('out.txt', 'r') as f:
            #     content = f.read()
            dict ={}
            dict[0] = {'comment_or_body': comment_text}
            data = pd.DataFrame.from_dict(dict, "index")
            data.to_csv('comment.csv', index=False)
            
            df = pd.read_csv('comment.csv')
            content = df['comment_or_body'][0]

            if prediction(content):
                issue.create_comment("Contains secret")
            else:
                issue.create_comment("You're safe")

    return "ok"

if __name__ == "__main__":
    app.run(debug=True, port=5000)
