class geometry:
    def __init__(self, *args, **kwargs):
        self.id = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
        self.data=None
        self.annex=None
        
    def Add_Data(self,data_file=None,dimension=None):
        if data_file is None:
