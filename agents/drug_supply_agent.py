class DrugSupplyAgent:
    def treat(self, patient_ids):
        for pid in patient_ids:
            print(f"Drugs given to patient {pid}")
