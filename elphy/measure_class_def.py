"""
Created on Thu Apr 04 14:00:00 2019

@author: vane1601
"""

__author__ = """Elijah EW Van Houten"""
__email__ = 'eew.vanhouten@usherbrooke.ca'
__version__ = '0.1.0'

class measure:
    def __init__(self, *args, **kwargs):
        self.id = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
        self.data=None
        self.annex=None

class elphy_measure(measure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)