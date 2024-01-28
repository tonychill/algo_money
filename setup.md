### Edit the .zshrc File:

### Open your .zshrc file in a text editor. You can use a command like open -e ~/.zshrc or nano ~/.zshrc.
### Update the PATH:

### Adjust the PATH export to include the directory /opt/homebrew/bin (which it seems you already have). You don't need to add the specific ### Python path since /opt/homebrew/bin should already be in your PATH.
export PATH="/opt/homebrew/bin:$PATH"
### Set Aliases (Optional):

### If you want to use python or python3 to specifically refer to Python 3.10, set aliases as follows:
alias python='/opt/homebrew/bin/python3.10'
alias python3='/opt/homebrew/bin/python3.10'

### Apply the Changes:

### After making these changes in the .zshrc file, save the file and exit the text editor.
### Apply the changes by sourcing your .zshrc file:
source ~/.zshrc

### Then, instruct Poetry to use this version. You can explicitly set the Python version for Poetry by using the poetry env use command  ### followed by the path to the Python executable. Since you have Python 3.10.13 set as the default, the command will be:
poetry env use python3.10

### You can list all virtual environments managed by Poetry with:
poetry env list

### On replit, pytorch is to large to install in the poetry lock file, so it must be preinstalled

poetry add torch==1.13.1

### Install poetry dependencies from pyproject.toml
poetry install

### You can run your Python scripts directly in the environment managed by Poetry. For example:
poetry run python stock_machine_learning_lstm.py
poetry run python machine_learning_pytorch_lstm_colab.py

### If you're using Jupyter notebooks, you can start Jupyter Notebook or JupyterLab in the Poetry-managed environment:
poetry run jupyter notebook

## Add these secrets to the secrets in replit
{
  "POLYGON_API_KEY": "value",
  "ALPACA_API_KEY": "value",
  "ALPACA_API_SECRET": "value",
  "IS_LIVE": "True",
  "ALPACA_IS_PAPER": "True"
}

# Create Replit
1. Inside Replit, click the "Create Replit" button
2. Click the "Import from GitHub" button
3. Add the GitHub account if not already listed.
4. Add Secrets
5. Deploy
## If you get this error
[Errno 122] Disk quota exceeded: '/home/runner/858f45ce-b2ed-4b2a-b336-ad43f75309d7/.pythonlibs/lib/python3.10/site-packages/plotly/validators/treemap/stream/_token.py'
### Is because on replit, pytorch is to large to install in the poetry lock file, so it must be preinstalled
### Execute this command from the shell
poetry add torch==1.13.1