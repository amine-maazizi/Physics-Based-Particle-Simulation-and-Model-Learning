def elasticity_to_color(elasticity, channel=0):
    color = [255, 255, 255]
    for i in range(3):
        if i != channel:
            color[i] *= elasticity
    return color