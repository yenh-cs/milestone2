# Git Guidelines
This project follows a hybrid of trunk-base and feature branching. [git strategies](https://tilburgsciencehub.com/topics/automation/version-control/advanced-git/git-branching-strategies/)
Specifically, .py modules will be treated as feature "branches" and the main .ipynb will be treated as "main/master".
## Example for Visualization

- Create visualization script in .py module
- when visualization is done, cleaned, documented, and ready to be incorporated into main notebook:
  1. import into main notebook
  2. write text associated with plot
  3. write text associated with code

### .py Module:
```python
# Visualizations.py
# Module for creating visualization 

import matplotlib.pyplot as plt

def nice_looking_plot(*args, **kwargs):
    # make nice plot
    return None
```

### .ipynb Main:
```python
# main.ipynb
# Main Notebook
from visualization import nice_looking_plot

# =========== Text Block ==================
"""
We did x and plotted x, y. Blah, blah, etc. Describe code, if needed, more blahs and etcs.
"""
# =========================================

# ========== code block ===================
nice_looking_plot()
# =====================================
```