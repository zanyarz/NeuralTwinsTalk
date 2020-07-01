
docker run --runtime=nvidia --name NTT -it \
        $(ls /dev/nvidia* | xargs -I{} echo '--device={}') \
        $(ls /usr/lib/{libcuda,libnvidia}* | xargs -I{} echo '-v {}:{}:ro') \
        -v $MAGI_ZZ_I:/workspace/neuralbabytalk/data/coco/images \
        -v $MAGI_ZZ_A:/workspace/neuralbabytalk/data/coco/annotations \
        --shm-size 32G -p 8888:8888 ZZ_NTT /bin/bash \
        $CMD $2 $3 $4 $5
~                                                                                                                                                
~                                                                                                                                                
~                                                                                                                                                
~                                                                                                                                                
~                                                                                                                                                
~                                                                                                                                                
~                                                                                                                                                
~                                                                                                                                                
~                                                                                                                                                
~                                                                                                                                                
~                                                                                                                                                
~                       
