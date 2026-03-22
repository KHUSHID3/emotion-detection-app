def recommend_task(stress_level):
    if stress_level == "Low Stress":
        return "Assign challenging and creative tasks."
    elif stress_level == "Medium Stress":
        return "Assign priority tasks with manageable workload."
    else:
        return "Assign light tasks and recommend a break."

# Test
stress = "High Stress"
print("Stress Level:", stress)
print("Task Recommendation:", recommend_task(stress))
