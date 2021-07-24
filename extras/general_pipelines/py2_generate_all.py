#! /usr/bin/env python3
import os
os.chdir(os.path.split(__file__)[0])

py2_pipeline_alternatives_file = \
    open("../../demo_app/py2_pipeline_alternatives.inc", 'w', encoding='utf-8')

warps_choices = (4, 6, 8, 10, 12, 16, 32)
tile_dim_choices = (8, 10, 12, 14, 16, 20, 24)

for warps in warps_choices:
    for tile_dim in tile_dim_choices:
        os.system(f"./py2_generate_source.py {warps} {tile_dim} {tile_dim}")
        name = f"py2_{warps}_{tile_dim}_{tile_dim}"
        py2_pipeline_alternatives_file.write('{"%s", {"%s"}},\n' % (name, name))
        os.system(f"git add ./{name}/general_pipeline_alternative.glsl")
        os.system(f"git add ./{name}/{name}.cpp")
