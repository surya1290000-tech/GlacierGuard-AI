with open('scratch/build_nb07.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace the unescaped docstring in make_augmented_gold
text = text.replace('    """Merge GLO gold with environmental features.', '    \\"\\"\\"Merge GLO gold with environmental features.')
text = text.replace('    augmented DataFrame with all GLO + environmental columns\n    """', '    augmented DataFrame with all GLO + environmental columns\n    \\"\\"\\"')

with open('scratch/build_nb07.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Fixed quotes in build_nb07.py!')
