from cute_popup import show_cute_popup

class AssistanceLayer:

    def __init__(self):
        self.frustration_counter = 0
        self.cooldown = 0

    def update(self, state):

        if self.cooldown > 0:
            self.cooldown -= 1

        if state == "frustrated":
            self.frustration_counter += 1
        else:
            self.frustration_counter = 0

        if self.frustration_counter >= 3 and self.cooldown == 0:
            print("💛 Triggering Assistance Popup...")
            show_cute_popup()
            self.frustration_counter = 0
            self.cooldown = 10