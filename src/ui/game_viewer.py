class GameViewer:
    def __init__(self, environment):
        self.environment = environment

    def display(self):
        self.environment.render()