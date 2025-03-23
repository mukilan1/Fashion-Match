"""
Dynamic color combination generator for fashion advice
"""
import random
import colorsys

# Color category definitions with example colors in each category
COLOR_CATEGORIES = {
    "warm": ["red", "orange", "yellow", "coral", "amber", "gold", "rust", "terracotta"],
    "cool": ["blue", "green", "purple", "teal", "mint", "turquoise", "lavender", "indigo"],
    "neutral": ["black", "white", "gray", "navy", "beige", "cream", "tan", "brown", "khaki"],
    "pastel": ["light pink", "baby blue", "mint green", "lavender", "peach", "pale yellow", "lilac"],
    "bold": ["magenta", "electric blue", "lime green", "bright orange", "hot pink", "cobalt", "emerald"]
}

# Color wheel position approximations (in degrees 0-360)
COLOR_WHEEL = {
    "red": 0,
    "orange": 30,
    "yellow": 60,
    "chartreuse": 90,
    "green": 120,
    "spring green": 150,
    "cyan": 180,
    "azure": 210,
    "blue": 240,
    "violet": 270,
    "magenta": 300,
    "rose": 330
}

# Expanded color relationships for fashion
COLOR_RELATIONSHIPS = {
    "complementary": "Colors directly opposite on the color wheel create high contrast and vibrant looks",
    "analogous": "Colors next to each other on the wheel for a harmonious, cohesive look",
    "triadic": "Three colors equally spaced around the wheel for a balanced yet vibrant combination",
    "split-complementary": "One color plus two colors adjacent to its complement for a dynamic but less intense contrast",
    "tetradic": "Four colors arranged in two complementary pairs for rich, varied combinations",
    "monochromatic": "Different shades, tints, and tones of a single color for a sophisticated, cohesive look"
}

def generate_color_advice(base_color=None):
    """Generate dynamic, tailored color combination advice"""
    # If no base color provided, recommend a good starting color
    if not base_color:
        base_color = random.choice(["navy", "black", "white", "camel", "gray"])
        starting_advice = f"{base_color.capitalize()} is an excellent foundation color as it's versatile and timeless. "
    else:
        base_color = base_color.lower()
        starting_advice = f"Starting with {base_color} as your base color, "
    
    # Determine color category of base color
    base_category = None
    for category, colors in COLOR_CATEGORIES.items():
        if base_color in colors:
            base_category = category
            break
    
    # Pick a color relationship approach
    relationship = random.choice(list(COLOR_RELATIONSHIPS.keys()))
    
    # Generate specific color combinations based on the relationship
    if relationship == "complementary":
        if base_color in COLOR_WHEEL:
            # Find the opposite color on the wheel
            opposite_deg = (COLOR_WHEEL[base_color] + 180) % 360
            complement = min(COLOR_WHEEL.keys(), key=lambda c: abs(COLOR_WHEEL[c] - opposite_deg))
            advice = f"{starting_advice}try pairing it with {complement} for a bold complementary contrast. {COLOR_RELATIONSHIPS[relationship]}. For the ideal ratio, use about 70% {base_color} and 30% {complement} to avoid overwhelming the eye."
        else:
            advice = f"{starting_advice}try pairing it with a complementary color that creates contrast. {COLOR_RELATIONSHIPS[relationship]}. A good ratio is 70-30 with your {base_color} as the dominant color."
            
    elif relationship == "analogous":
        if base_color in COLOR_WHEEL:
            # Find adjacent colors
            base_deg = COLOR_WHEEL[base_color]
            adjacent1_deg = (base_deg + 30) % 360
            adjacent2_deg = (base_deg - 30 + 360) % 360
            adjacent1 = min(COLOR_WHEEL.keys(), key=lambda c: abs(COLOR_WHEEL[c] - adjacent1_deg))
            adjacent2 = min(COLOR_WHEEL.keys(), key=lambda c: abs(COLOR_WHEEL[c] - adjacent2_deg))
            advice = f"{starting_advice}create an analogous palette with {adjacent1} and {adjacent2}. {COLOR_RELATIONSHIPS[relationship]}. For best results, use a 60-30-10 ratio of these three colors."
        else:
            # Suggest general analogous approach
            advice = f"{starting_advice}look for colors that sit next to each other on the color wheel. {COLOR_RELATIONSHIPS[relationship]}. Try a 60-30-10 distribution of your three chosen colors."
    
    elif relationship == "monochromatic":
        advice = f"{starting_advice}explore different shades and tints of {base_color}. {COLOR_RELATIONSHIPS[relationship]}. For visual interest, use at least three different intensities of {base_color} with a 60-30-10 ratio from lightest to darkest."
    
    elif relationship == "triadic":
        advice = f"{starting_advice}try a triadic color scheme by choosing two other colors equally spaced around the color wheel. {COLOR_RELATIONSHIPS[relationship]}. For balance, use your {base_color} as 60% of the outfit, with 30% and 10% for the other two colors."
    
    else:  # Other relationships
        advice = f"{starting_advice}consider a {relationship} color scheme. {COLOR_RELATIONSHIPS[relationship]}. The ideal distribution is to use your main color for 60% of the outfit, with accent colors making up the remaining 40%."
    
    # Add a practical fashion tip based on the base color
    if base_category == "neutral":
        advice += f" {base_color.capitalize()} as a neutral provides an excellent canvas for adding pops of vibrant colors through accessories."
    elif base_category == "bold":
        advice += f" Since {base_color} is quite vibrant, balance it with neutrals like black, white, or beige to let it shine without overwhelming."
    elif base_category == "warm":
        advice += f" {base_color.capitalize()} brings warmth to your outfit and pairs beautifully with gold accessories and earthy tones."
    elif base_category == "cool":
        advice += f" {base_color.capitalize()} creates a cool, calm feeling and works well with silver accessories and other cool tones."
    
    return advice

def get_trending_color_combinations():
    """Generate information about currently trending color combinations"""
    # Simulate trending combinations (these would ideally come from a regularly updated source)
    trending = [
        {"name": "Neo Mint & Lavender", "description": "Fresh mint green paired with soft lavender creates a soothing, contemporary palette that's on trend this season."},
        {"name": "Burnt Orange & Navy", "description": "This rich, contrasting combination offers warmth and depth, perfect for transitional seasons."},
        {"name": "Sage Green & Terracotta", "description": "Nature-inspired hues that bring an earthy, grounded quality to outfits with a modern edge."},
        {"name": "Buttercream & Chocolate Brown", "description": "A soft, neutral pairing that's replacing stark black and white for a more gentle contrast."},
        {"name": "Cerulean Blue & Coral", "description": "Vibrant yet balanced, this combination references coastal aesthetics with a contemporary twist."}
    ]
    return random.sample(trending, 2)  # Return 2 random trending combinations
