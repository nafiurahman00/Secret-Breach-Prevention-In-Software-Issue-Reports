import os
import requests

from flask import Flask, request
from github import Github, GithubIntegration

app = Flask(__name__)

# MAKE SURE TO CHANGE TO YOUR APP NUMBER!!!!!
app_id = 807924

# Read the bot certificate
with open(
        os.path.normpath(os.path.expanduser('"C:\Users\Asus\Downloads\secret-detection-tool.2024-01-25.private-key.pem"')),
        'r'
) as cert_file:
    app_key = cert_file.read()

# Create a GitHub integration instance
git_integration = GithubIntegration(
    app_id,
    app_key,
)

@app.route("/", methods=['POST'])
def bot():
    # Get the event payload
    payload = request.json

    # Check if the event is a GitHub issue event
    if 'issue' not in payload or payload['action'] != 'opened':
        return "ok"

    owner = payload['repository']['owner']['login']
    repo_name = payload['repository']['name']

    # Get a GitHub connection as our bot
    # Here is where we are getting the permission to talk as our bot and not
    # as a Python webservice
    git_connection = Github(
        login_or_token=git_integration.get_access_token(
            git_integration.get_installation(owner, repo_name).id
        ).token
    )
    repo = git_connection.get_repo(f"{owner}/{repo_name}")

    issue = repo.get_issue(number=payload['issue']['number'])

    # Now you can access information about the issue, such as title and body
    issue_title = issue.title
    issue_body = issue.body

    # Your logic for secret detection can go here
    # Replace the following line with your actual implementation
    if detect_secrets(issue_title) or detect_secrets(issue_body):
        # Your logic for warning the user can go here
        # Replace the following line with your actual implementation
        issue.create_comment("Warning: This issue contains secrets!")

    return "ok"


def detect_secrets(text):
    # Your logic for secret detection using your ML model can go here
    # Replace the following line with your actual implementation
    return False

if __name__ == "__main__":
    app.run(debug=True, port=5000)
    
