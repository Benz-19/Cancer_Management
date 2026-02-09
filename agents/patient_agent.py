class PatientAgent:
    def __init__(self, patient_id):
        self.patient_id = patient_id
        self.data = {}

    def update_data(self, new_data):
        self.data.update(new_data)

    def get_data(self):
        return self.data
