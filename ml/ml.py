
class ML:
    def __init__(self, models):
        self.models = models

    def run(self):
        """Run models"""

        for model in self.models:
            try:
                model.train()
                model.predict()
            except Exception as e:
                raise e

        return 1
