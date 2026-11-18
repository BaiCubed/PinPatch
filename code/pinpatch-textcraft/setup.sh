#!/usr/bin/env bash
# Fetch the TextCraft environment (ADaPT, MIT) and the Minecraft 1.16.5 recipes into third_party/.
# Everything stays inside this project folder.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p third_party && cd third_party
if [ ! -d ADaPT ]; then
  git clone https://github.com/archiki/ADaPT.git
  git -C ADaPT checkout ecdc4ab0030b4be9be122622d8ea78f8c59c44c4
fi
if [ ! -d mcassets ]; then
  git clone --depth 1 --filter=blob:none --sparse -b 1.16.5 https://github.com/InventivetalentDev/minecraft-assets.git mcassets
  git -C mcassets sparse-checkout set data/minecraft/recipes
fi
mkdir -p textcraft/recipes
cp ADaPT/TextCraft/env.py ADaPT/TextCraft/crafting_tree.py ADaPT/TextCraft/utils.py textcraft/
# Pin recipe loading order (the tree depends on it; os.listdir order differs across filesystems).
python3 - <<'PY'
p = 'textcraft/crafting_tree.py'
s = open(p).read()
a = "for f in os.listdir(os.path.join(minecraft_dir, 'recipes/')):"
b = "for f in (json.load(open(os.environ['TEXTCRAFT_RECIPE_ORDER'])) if os.environ.get('TEXTCRAFT_RECIPE_ORDER') else os.listdir(os.path.join(minecraft_dir, 'recipes/'))):"
assert a in s
open(p, 'w').write(s.replace(a, b))
PY
touch textcraft/__init__.py
cp mcassets/data/minecraft/recipes/*.json textcraft/recipes/
n=$(ls textcraft/recipes | wc -l)
echo "recipes: $n (expected 860)"
[ "$n" = "860" ] || { echo "unexpected recipe count"; exit 1; }
python3 -c "import numpy" 2>/dev/null || pip install numpy
echo "setup done"
