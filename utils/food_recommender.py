def recommend_food(glucose_level):
    if glucose_level > 180:
        return [("Broccoli", 100), ("Quinoa", 80)]
    elif glucose_level < 80:
        return [("Banana", 120), ("Oatmeal", 150)]
    else:
        return [("Grilled Chicken", 150), ("Sweet Potato", 100)]