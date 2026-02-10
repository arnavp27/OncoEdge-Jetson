"""
Clinical Decision Tree

Risk stratification based on vision scores and patient metadata.
"""


class ClinicalDecisionTree:
    """
    Clinical risk assessment using decision tree logic.

    Combines AI vision scores with patient risk factors (age, tobacco use,
    lesion duration) to provide clinical risk stratification.
    """

    def assess_risk(self, vision_score, age, tobacco_years, lesion_duration):
        """
        Assess clinical risk level based on vision score and patient metadata.

        Risk Calculation:
            - Base score: vision_score × 10
            - Age > 40: +1.5
            - Tobacco use > 10 years: +2.0
            - Lesion duration > 6 weeks: +1.5

        Classification:
            - HIGH (>8.0): Urgent referral within 2 weeks
            - MEDIUM (>5.0): Non-urgent referral within 4 weeks
            - LOW (≤5.0): Routine monitoring

        Args:
            vision_score: Fused AI vision score (0-1)
            age: Patient age in years
            tobacco_years: Years of tobacco use
            lesion_duration: Lesion duration category
                ("< 2 weeks", "2-6 weeks", "> 6 weeks")

        Returns:
            dict: Risk assessment containing:
                - 'level': Risk level (str) - "HIGH", "MEDIUM", or "LOW"
                - 'recommendation': Clinical recommendation (str)
                - 'score': Computed risk score (float)
        """
        # Base risk score from vision
        risk_score = vision_score * 10

        # Add clinical risk factors
        if age > 40:
            risk_score += 1.5

        if tobacco_years > 10:
            risk_score += 2.0

        if lesion_duration == "> 6 weeks":
            risk_score += 1.5

        # Classify risk level and provide recommendation
        if risk_score > 8.0:
            return {
                'level': 'HIGH',
                'recommendation': 'Urgent referral to specialist within 2 weeks',
                'score': risk_score
            }
        elif risk_score > 5.0:
            return {
                'level': 'MEDIUM',
                'recommendation': 'Non-urgent referral within 4 weeks',
                'score': risk_score
            }
        else:
            return {
                'level': 'LOW',
                'recommendation': 'Routine monitoring recommended',
                'score': risk_score
            }

    def get_risk_factors(self, age, tobacco_years, lesion_duration):
        """
        Identify active risk factors for a patient.

        Args:
            age: Patient age
            tobacco_years: Years of tobacco use
            lesion_duration: Lesion duration category

        Returns:
            list: List of active risk factor descriptions
        """
        risk_factors = []

        if age > 40:
            risk_factors.append(f"Age over 40 ({age} years)")

        if tobacco_years > 10:
            risk_factors.append(f"Heavy tobacco use ({tobacco_years} years)")

        if lesion_duration == "> 6 weeks":
            risk_factors.append("Persistent lesion (> 6 weeks)")

        return risk_factors
