
from dataprob.plot.appearance import default_styles

def test_default_styles():

    # silly test. code ran and dictionary validated at this level on import 
    # anyway. 
    for a in default_styles:
        for b in default_styles[a]:
            default_styles[a][b]