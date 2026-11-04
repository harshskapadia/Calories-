"""Map Food Recognition 2022 (MyFoodRepo, 498 classes) names onto the app's 103 FoodSeg103 classes.

Ordered keyword rules: the first rule whose pattern matches wins. "skip" = zero-calorie items (water, ice) that are better
left as background; anything unmatched becomes "other ingredients". Run this file to print the mapping for review.
"""

from __future__ import annotations

import re

RULES: list[tuple[str, str]] = [
    # ---- leave as background (no energy, not food on a plate)
    (r"^water($|-mineral|-with)|ice-cubes|^salt$|^sweetener", "skip"),
    # ---- names that would otherwise be caught by a later, wrong keyword
    (r"^(mixed-salad|salad-leaf|salad-rocket|salad-lambs|greek-salad|witloof)", "lettuce"),
    (r"^potato-salad", "potato"), (r"^chicken-curry", "chicken duck"),
    (r"^sauce|-sauce$|tartar-sauce|dressing|ketchup|mayonnaise|mustard|vinegar|^dips?$|pesto|gravy", "sauce"),
    (r"^soup|bouillon|broth", "soup"),
    (r"tofu|tempeh|meat-substitute|seitan|quorn", "tofu"),
    # ---- drinks
    (r"wine|prosecco|champagne|sekt|aperitif|cocktail|beer|cider", "wine"),
    (r"coffee|espresso|ristretto|cappuccino|latte|macchiato", "coffee"),
    (r"^tea|-tea|herbal", "tea"),
    (r"juice|coca-cola|^cola|lemonade|soft-drink|syrup|glucose", "juice"),
    (r"milkshake|smoothie|chocolate-milk|kefir", "milkshake"),
    (r"^(milk|oat-milk|almond-milk|soy-milk|soya-drink.*)$", "milk"),
    # ---- sweets & bakery
    (r"ice-cream|sorbet|gelato", "ice cream"),
    (r"mousse|pudding|panna-cotta|tiramisu|custard|creme-dessert", "pudding"),
    (r"cake|brownie|muffin|tart$|tart-|^tart|black-forest|macaroon|meringue|cupcake|donut|doughnut|pancake", "cake"),
    (r"^pie(-|$)|-pie(-|$)|quiche|empanada", "pie"),
    (r"cookie|biscuit|cracker|grissini|crisp-bread|rusk|wafer|waffel|pretzel", "biscuit"),
    (r"chocolate|praline|nutella|m-m-s|cocoa", "chocolate"),
    (r"gummi|candy|candies|sweets|lollipop|marzipan|sugar", "candy"),
    (r"popcorn", "popcorn"),
    (r"fries|rosti|hash-brown|wedges|crisps|paprika-chips|tortilla-chips", "french fries"),
    # ---- legumes, nuts, seeds, dried fruit
    (r"bean-seeds|beans-kidney|beans-white|chickpea|lentil|hummus|falafel|chili-con|dahl|dhal", "red beans"),
    (r"edamame|^soy", "soy"),
    (r"peanut", "peanut"),
    (r"almond", "almond"),
    (r"cashew", "cashew"),
    (r"walnut|pecan|hazelnut|brazil-nut|pistachio|pine-nuts|macadamia|chestnut|^nuts|mixed-nuts|seeds|linseed|chia", "walnut"),
    (r"raisin|cranberr|dried-fruit|-dried$|dates|prune", "dried cranberries"),
    # ---- eggs
    (r"^egg($|-|s$)|omelette", "egg"),
    # ---- fruit
    (r"^apple|applesauce", "apple"), (r"apricot", "apricot"), (r"avocado|guacamole", "avocado"), (r"banana", "banana"),
    (r"strawberr", "strawberry"), (r"cherr", "cherry"), (r"blueberr|blackberr|^berries|currant", "blueberry"),
    (r"raspberr", "raspberry"), (r"mango", "mango"), (r"olive", "olives"), (r"peach|nectarine", "peach"),
    (r"^lemon|^lime", "lemon"), (r"^pears?$|^pear-", "pear"), (r"^figs?$", "fig"), (r"pineapple", "pineapple"),
    (r"^grapes?$", "grape"), (r"kiwi", "kiwi"), (r"watermelon", "watermelon"), (r"melon", "melon"),
    (r"orange|mandarin|clementine|grapefruit|pomelo|kaki|persimmon", "orange"),
    (r"plum|damson|pomegranate|^fruit|papaya|passion|lychee|^coconut|compote|coulis", "apple"),
    # ---- seafood, meat
    (r"shrimp|prawn|scampi", "shrimp"), (r"crab|lobster", "crab"), (r"mussel|clam|squid|octopus|scallop", "shellfish"),
    (r"salmon|tuna|cod$|^cod|perch|trout|fish|anchov|sardine|mackerel|sushi", "fish"),
    (r"pizza|focaccia|dough", "pizza"),
    (r"hamburger|burger|kebab", "hamburg"),
    (r"nugget|schnitzel|cordon-bleu|breaded|fried-chicken", "fried meat"),
    (r"chicken|turkey|duck|poultry", "chicken duck"),
    (r"sausage|cervelat|wienerli|salami|chorizo|frankfurter|^ham|bacon|charcuterie|dried-meat|terrine|pate$|hot-dog", "sausage"),
    (r"^lamb|mutton|^goat-average", "lamb"),
    (r"pork", "pork"),
    (r"beef|steak|veal|meat|bolognaise", "steak"),
    # ---- starches & dishes
    (r"dumpling|ravioli|tortellini|tortelloni|gnocchi|wonton|gyoza|spring-roll", "wonton dumplings"),
    (r"noodle|vermicelli|ramen|udon", "noodles"),
    (r"pasta|spaghetti|penne|macaroni|farfalle|linguin|tagliatelle|hornli|couscous|taboule|lasagne|spaetzle", "pasta"),
    (r"rice|risotto|paella|quinoa|bulgur|buckwheat|ebly|millet|polenta|grits|porridge|muesli|oat|cereal|corn-flakes", "rice"),
    (r"bread|toast|bun$|roll|croissant|brioche|bagel|baguette|pita|naan|tortilla|fajita|wrap|crepe|waffle|croutons|loaf|"
     r"pastry|sandwich|croque", "bread"),
    (r"potato|mashed|gratin|dauphinois", "potato"),
    (r"corn|maize", "corn"),
    (r"curry|ratatouille|stew$", "soup"),
    # ---- dairy (after dishes, so "butterfly pasta" or "risotto without cheese" are not dairy)
    (r"butter|margarine|cheese|brie|cheddar|gruyere|emmental|mozzarella|parmesan|feta|halloumi|raclette|fondue|philadelphia|tomme|"
     r"tete-de-moine|curd|quark|cream|yaourt|yogurt|yoghurt|faux-mage|tzatziki", "cheese butter"),
    (r"jam|honey|spread|cenovis|marmite|oil$|sauce", "sauce"),
    # ---- vegetables
    (r"tomato|caprese", "tomato"), (r"eggplant|aubergine", "eggplant"), (r"cauliflower", "cauliflower"),
    (r"broccoli|romanesco", "broccoli"), (r"carrot", "carrot"), (r"cucumber|zucchini|courgette|gherkin|pickle", "cucumber"),
    (r"asparagus", "asparagus"), (r"celery|celeriac|fennel", "celery stick"), (r"pumpkin|squash|butternut", "pumpkin"),
    (r"pepper|paprika|capsicum", "pepper"), (r"onion|leek|chive|shallot|scallion", "onion"),
    (r"garlic", "garlic"), (r"ginger", "ginger"), (r"mushroom|champignon|chanterelle", "white button mushroom"),
    (r"french-beans|green-bean|haricot", "French beans"), (r"^peas|snow-pea|mangetout", "snow peas"),
    (r"alfa|bean-sprout", "bean sprouts"), (r"shoots", "bamboo shoots"),
    (r"brussel|cabbage|sauerkraut|kohl|kolhrabi|savoy", "cabbage"),
    (r"spinach|chard|kale|rocket|arugula|lettuce|salad|endive|chicory|watercress", "lettuce"),
    (r"radish|beetroot|turnip", "white radish"),
    (r"basil|parsley|coriander|cilantro|mint|dill|herb|bouquet", "cilantro mint"),
    (r"artichoke|okra", "okra"), (r"seaweed|nori|algae", "seaweed"),
    (r"vegetable|veggie|caper", "salad"),
]


def map_name(name: str) -> str:
    n = name.lower().strip()
    for pattern, target in RULES:
        if re.search(pattern, n):
            return target
    return "other ingredients"


if __name__ == "__main__":
    import json
    import sys

    names = json.load(open(sys.argv[1], encoding="utf-8")) if len(sys.argv) > 1 else {}
    for n in sorted(names):
        print(f"{n:60s} -> {map_name(n)}")
