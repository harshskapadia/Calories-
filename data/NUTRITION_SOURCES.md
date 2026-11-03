# Nutrition data: sources and method

`data/nutrition.csv` gives, for each of the 139 recognisable dishes (101 Food-101 + 38 Indian; 5 Indian labels merge into Food-101 classes):

| column | meaning |
|---|---|
| `kcal_per_100g`, `protein_per_100g`, `carbs_per_100g`, `fat_per_100g` | energy (kcal) and macros (g) per 100 g |
| `serving_g` | typical serving weight used as the "medium" portion |
| `match` | how well the USDA entry represents the dish (see below) |
| `fdc_ids` | FoodData Central ids and their weight shares, e.g. `2705853:75\|2705730:10` |
| `fndds_description` | USDA description(s) of the component foods |
| `notes` | assumptions, proxies and serving overrides |

## Source

**USDA FoodData Central, FNDDS 2021–2023** ("Food and Nutrient Database for Dietary Studies",
the database behind the *What We Eat in America* survey). Bulk CSV release **2024-10-31**,
public domain (CC0): <https://fdc.nal.usda.gov/download-datasets>.

FNDDS was chosen over SR Legacy / Foundation Foods because it describes **prepared, mixed
dishes as eaten** (e.g. "Pad Thai, NFS", "Pizza, cheese, from restaurant") and includes
household **portion weights**.

## Method (reproducible: `make nutrition`)

1. `data/nutrition_mapping.csv` maps each class to one or more FNDDS foods, chosen by hand
   after keyword search of `food.csv`.
2. `data/build_nutrition.py` downloads the FNDDS bundle (~3 MB), reads nutrients
   208 (energy, kcal), 203 (protein), 205 (carbohydrate by difference) and 204 (total fat), and
   computes the weight-weighted average for composite recipes.
3. **Serving weight** = the FNDDS portion "Quantity not specified" of the first component (USDA's
   typical amount eaten), unless the mapping sets `serving_g`. That is the case for every composite
   and for 15 classes where the FNDDS default is a single piece that doesn't match a
   plated photo (e.g. one falafel = 28 g → 4 patties = 120 g). Each override is recorded in `notes`.

### Match quality

| match | count | meaning |
|---|---|---|
| `exact` | 74 | FNDDS has the dish itself (incl. "chappatti or roti", idli, masala dosa, pakora, palak paneer, fish curry, white rice) |
| `close` | 20 | a near-equivalent dish (e.g. cannoli → "Pastry, Italian, with cheese"; biryani → "Biryani with chicken"; kheer → "Firni, Indian pudding") |
| `composite` | 40 | no FNDDS entry; built from ingredients with a documented recipe (e.g. caprese = 45% mozzarella, 45% tomato, 7% olive oil, 3% basil; dal makhani = lentils + kidney beans + butter + cream) |
| `proxy` | 5 | a known-imperfect stand-in: foie gras → liver pâté; macarons → coconut macaroon; dhokla → idli; ghevar → funnel cake; kulfi → ice cream |

Indian dishes use Indian-typical portions (1 katori ≈ 150–200 g of curry, 2 chapatis = 80 g, 3 idlis = 120 g).
The [Indian Food Composition Tables 2017](https://ifct2017.github.io) (NIN) list raw ingredients rather than
cooked dishes, so FNDDS (which has cooked Indian dishes) was used for consistency with the rest of the table.

### Search table
`data/food_search.csv` contains **all 5,431 FNDDS foods** (per-100 g macros + typical serving), so the app can
give an estimate for anything the image model doesn't recognise.

### Sanity checks (enforced in `tests/test_nutrition.py`)
- Every Food-101 class and every Indian dataset label (after merging) has a row: 139 in total.
- 4·protein + 4·carbs + 9·fat matches the stated kcal within 15% for every class.
- Serving weights are between 10 g and 1 kg.

## Caveats
Values describe a *typical US recipe*. Restaurant portions, oil, sauces and dressings vary
widely, so treat every number as an **estimate**, not medical or dietary advice.
