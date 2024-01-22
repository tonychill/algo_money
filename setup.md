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

### You can run your Python scripts directly in the environment managed by Poetry. For example:
poetry run python stock_machine_learning_lstm.py
poetry run python machine_learning_pytorch_lstm_colab.py

### If you're using Jupyter notebooks, you can start Jupyter Notebook or JupyterLab in the Poetry-managed environment:
poetry run jupyter notebook
