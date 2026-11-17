# Your own training photos

Put photos of a dish in a folder named after the dish, then train (v6 or later config):

```
data/my_photos/
  Halal snack pack/      <- the name as the app shows it in search (or a class key like halal_snack_pack)
    IMG_0001.jpg
    IMG_0002.jpg
    ...
```

- **15+ photos per dish** (30-50 is much better). Different days, plates, angles and lighting; the dish filling most of
  the photo. JPEG, PNG or WebP.
- The folder name must be a dish the app already has nutrition for: search it in the app first. If the name is not
  found, training prints a warning and skips that folder.
- Every 5th photo is kept aside to test the model, the rest are used for training.
- These photos stay on your computer: `.gitignore` keeps everything in this folder except this README out of git.
- Photos you download from the internet are fine for training your own copy, but don't commit or share them.
