import os
import requests

from flask import Flask, request
from github import Github, GithubIntegration


app = Flask(__name__)
# MAKE SURE TO CHANGE TO YOUR APP NUMBER!!!!!
app_id = 829793
# Read the bot certificate
with open(
        os.path.normpath(os.path.expanduser(r"secret-detection-tool.2024-02-15.private-key.pem")),
        'r'
) as cert_file:
    app_key = cert_file.read()

# Create an GitHub integration instance
git_integration = GithubIntegration(
    app_id,
    app_key,
)

def contains_vowels(text):
    vowels = 'aeiouAEIOU'
    return any(char in vowels for char in text)


@app.route("/", methods=['POST'])
def bot():
    print("here")
    # Get the event payload
    payload = request.json

    # Check if the event is a GitHub PR creation event
    if not all(k in payload.keys() for k in ['action', 'pull_request']) and \
            payload['action'] == 'opened':
        return "ok"

    owner = payload['repository']['owner']['login']
    repo_name = payload['repository']['name']

    # Get a git connection as our bot
    # Here is where we are getting the permission to talk as our bot and not
    # as a Python webservice
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

   
    # Check if the pull request description contains vowels
    if contains_vowels(pr_description):
        issue.create_comment("Yay! The pull request description contains vowels.")
    else:
        issue.create_comment("Alas! The pull request description does not contain vowels.")


    return "ok"


if __name__ == "__main__":
    app.run(debug=True, port=5000)