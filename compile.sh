#!/bin/bash
for source in $(find . -maxdepth 1 -type f ! -name "*.spv" ! -name "*.sh" ! -name "*.h"); do 
    spv=${source##*.}.spv
    echo "Compiling $source to $spv"
    glslc --target-spv=spv1.5 $source -o $spv 
    glslangValidator --target-env vulkan1.2 -o $spv $source
done
